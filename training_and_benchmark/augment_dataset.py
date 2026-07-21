#!/usr/bin/env python3
"""
Adds deeper, more substantive examples for existing citations to the existing dataset.

Feeds DeepSeek each citation's paragraph from citation_lookup.jsonl for grounding.
Explicitly requests non-compliant and partially compliant scenarios.

Run build_citation_lookup.py first if you haven't already.
"""
import asyncio
import os
import json
import re
import sys
import httpx

DATASET_FILE = "nerc_cip_phi4_dataset.jsonl"
LOOKUP_FILE = "citation_lookup.jsonl"
OUTPUT_FILE = "nerc_cip_phi4_dataset_augmented.jsonl"
CHECKPOINT_FILE = "augmentation_checkpoint.json"
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

SAMPLES_PER_AUGMENT = 3  # additional examples to generate per existing citation
SEMAPHORE_LIMIT = 1

STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)

# Extracts the numeric identifier from a citation string
CORE_ID_PATTERNS = [
    re.compile(r"attachment\s+\d+,?\s+section\s+[\d.]+", re.IGNORECASE),
    re.compile(r"section\s+[\d.]+", re.IGNORECASE),
    re.compile(r"part\s+[\d.]+", re.IGNORECASE),
    re.compile(r"r\d+(?:\.\d+)*", re.IGNORECASE),
    re.compile(r"requirement\s+\d+(?:\.\d+)*", re.IGNORECASE),
]


def normalize_key(citation: str):
    """Returns (standard, core_id) so citations with different phrasing but the
    same underlying standard+requirement/part/section match each other."""
    if not citation:
        return None
    citation_lower = citation.lower()

    standard_match = STANDARD_PATTERN.search(citation)
    standard = standard_match.group(1).upper() if standard_match else None

    core_id = None
    for pattern in CORE_ID_PATTERNS:
        match = pattern.search(citation_lower)
        if match:
            core_id = re.sub(r"[,.]", "", match.group(0)).strip()
            break

    if not standard and not core_id:
        return None
    return (standard, core_id)


def build_lookup_index(lookup: dict):
    """Builds a secondary index keyed by normalize_key() for fallback matching
    when exact dataset and lookup table citation strings fail to match."""
    index = {}
    for entry in lookup.values():
        key = normalize_key(entry["citation"])
        if key and key not in index:
            index[key] = entry
    return index


def find_context(citation: str, exact_lookup: dict, normalized_index: dict):
    exact = exact_lookup.get(citation.lower().strip())
    if exact:
        return exact
    key = normalize_key(citation)
    if key:
        return normalized_index.get(key)
    return None

AUGMENT_PROMPT_TEMPLATE = """You are an expert in NERC CIP compliance auditing.

You are given the SPECIFIC source text for one citation: "{citation}"

Source text for this citation:
\"\"\"{context}\"\"\"

Generate exactly {n} DISTINCT, realistic audit training examples for this exact citation.

CRITICAL INSTRUCTIONS:
1. SUBSTANCE OVER RESTATEMENT: Each "output" must reference the actual concrete criteria,
   thresholds, timeframes, or methods stated in the source text above -- not a generic
   paraphrase of the topic. Quote or closely reflect specific numbers/timeframes/methods
   that appear in the source text.
2. VARY COMPLIANCE OUTCOMES: Do not make all {n} examples affirmative. Include a mix --
   at least one should describe a partial gap or plausible non-compliant scenario, written
   the way an auditor would document a finding (e.g. "The entity's process does not specify
   X" or "Evidence was incomplete regarding Y"). Each example needs a "compliance_status"
   field: exactly one of "compliant", "partial", "non_compliant".
3. STAY ON THIS EXACT CITATION -- THIS IS CRITICAL: The source text above may include
   sibling or parallel sections/parts (e.g. if this is "Section 1", the text may also show
   "Section 2" and "Section 3" nearby as related sub-items of the same requirement). That is
   normal document structure -- use it ONLY to understand what makes "{citation}" distinct
   from its siblings, but your {n} examples must describe ONLY what "{citation}" itself
   requires. Do NOT mention, restate, summarize, or contrast against what any sibling
   section/part/requirement covers, even briefly. If you find yourself writing a sentence
   about a different number than "{citation}", delete it.
4. The "output" MUST explicitly state the citation: "{citation}"
5. Do not invent facts not supported by the source text above.
6. After thinking, respond with ONLY a JSON array in a ```json code block.

Target Schema:
[
  {{
    "instruction": "...",
    "input": "...",
    "output": "... including specific criteria from the source and the citation ...",
    "citation": "{citation}",
    "compliance_status": "compliant"
  }},
  ...
]
"""


def clean_and_parse_json_array(raw_text: str) -> list:
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    markdown_match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
    if markdown_match:
        try:
            data = json.loads(markdown_match.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    bracket_match = re.search(r'(\[.*\])', cleaned, re.DOTALL)
    if bracket_match:
        try:
            data = json.loads(bracket_match.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not find a valid JSON array in the LLM's response.")


def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_citations", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed_citations": sorted(completed)}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


NUMBERED_REF_PATTERNS = [
    ("section", re.compile(r"(?:attachment\s+\d+,?\s+)?section\s+([\d.]+)", re.IGNORECASE)),
    ("part", re.compile(r"part\s+([\d.]+)", re.IGNORECASE)),
    ("r", re.compile(r"\br(\d+(?:\.\d+)*)\b", re.IGNORECASE)),
    ("requirement", re.compile(r"requirement\s+(\d+(?:\.\d+)*)", re.IGNORECASE)),
]


def find_all_refs(text: str) -> set:
    """Returns a set of (category, number) pairs found in text.
    The attachment number is deliberately ignored."""
    refs = set()
    for category, pattern in NUMBERED_REF_PATTERNS:
        for match in pattern.finditer(text):
            number = match.group(1).rstrip(".")
            refs.add((category, number))
    return refs


def output_matches_citation(output: str, expected_citation: str) -> bool:
    expected_refs = find_all_refs(expected_citation)
    if not expected_refs:
        return True

    output_refs = find_all_refs(output)
    for category, number in output_refs:
        expected_numbers_for_category = {n for c, n in expected_refs if c == category}
        if expected_numbers_for_category and number not in expected_numbers_for_category:
            return False  # same category, different number -- drift
    return True


def validate_augmented(samples: list, source_context: str, expected_citation: str, source_file: str) -> list:
    valid = []
    dropped = 0
    seen_statuses = set()
    for item in samples:
        citation = item.get("citation", "").strip()
        output = item.get("output", "")
        instruction = item.get("instruction", "")
        status = item.get("compliance_status", "").strip().lower()

        if not citation or not output or not instruction:
            dropped += 1
            continue
        if citation.lower() != expected_citation.lower():
            dropped += 1
            continue
        if not output_matches_citation(output, expected_citation):
            dropped += 1
            print(f"[Validation] Dropped a sample for '{expected_citation}' -- "
                  f"output text referenced a different section/part/requirement number.")
            continue
        if status not in {"compliant", "partial", "non_compliant"}:
            status = "compliant"
        item["compliance_status"] = status
        item["source_file"] = source_file
        seen_statuses.add(status)
        valid.append(item)

    if dropped:
        print(f"[Validation] Dropped {dropped} sample(s) for citation '{expected_citation}'.")
    if valid and len(seen_statuses) == 1 and "compliant" in seen_statuses:
        print(f"[Warning] All kept samples for '{expected_citation}' are 'compliant' -- "
              f"the diversity instruction may not have been followed for this citation.")
    return valid


async def augment_citation(citation: str, context: str, source_file: str, client: httpx.AsyncClient, sem: asyncio.Semaphore):
    async with sem:
        prompt = AUGMENT_PROMPT_TEMPLATE.format(citation=citation, context=context, n=SAMPLES_PER_AUGMENT)
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3, "num_ctx": 8192},
        }
        full_response = ""
        try:
            async with client.stream("POST", OLLAMA_API_URL, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    print(f"[Error] {response.status_code} for '{citation}': {error_body.decode(errors='replace')}")
                    return None
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            print(token, end="", flush=True)
                            full_response += token
                        except json.JSONDecodeError:
                            continue
            print(f"\n[Generation Complete for '{citation}']")
            parsed = clean_and_parse_json_array(full_response)
            return validate_augmented(parsed, context, citation, source_file)
        except Exception as e:
            print(f"\n[Failed] '{citation}': {e}", file=sys.stderr)
            return None


async def main():
    dataset = load_jsonl(DATASET_FILE)
    lookup_raw = load_jsonl(LOOKUP_FILE)
    lookup = {entry["citation"].lower().strip(): entry for entry in lookup_raw}

    if not dataset:
        print(f"[Fatal] {DATASET_FILE} not found or empty.")
        sys.exit(1)
    if not lookup:
        print(f"[Fatal] {LOOKUP_FILE} not found or empty. Run build_citation_lookup.py first.")
        sys.exit(1)

    unique_citations = sorted(set(d["citation"] for d in dataset if d.get("citation")))
    completed = load_checkpoint()
    remaining = [c for c in unique_citations if c not in completed]

    normalized_index = build_lookup_index(lookup)

    print(f"[Plan] {len(unique_citations)} unique citations in dataset, "
          f"{len(completed)} already augmented, {len(remaining)} remaining.")

    skipped_no_context = 0
    zero_result_citations = []
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(connect=15.0, read=1800.0, write=30.0, pool=15.0)
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    total_saved = 0
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as outfile:
            for i, citation in enumerate(remaining, 1):
                lookup_entry = find_context(citation, lookup, normalized_index)
                if not lookup_entry:
                    print(f"[Skip] No source context found for '{citation}' in {LOOKUP_FILE}.")
                    skipped_no_context += 1
                    completed.add(citation)  # don't retry forever if it'll never resolve
                    save_checkpoint(completed)
                    continue

                print(f"\n--- Augmenting {i}/{len(remaining)}: '{citation}' ---")
                results = await augment_citation(
                    citation, lookup_entry["context"], lookup_entry.get("source_file", ""), client, sem
                )

                # If everything got dropped (e.g. drift), give it ONE retry --
                # temperature/prompt variance means a second attempt sometimes succeeds
                # even when the first one drifted.
                if not results:
                    print(f"[Retry] All samples dropped for '{citation}' -- trying once more.")
                    results = await augment_citation(
                        citation, lookup_entry["context"], lookup_entry.get("source_file", ""), client, sem
                    )

                if results:
                    for sample in results:
                        outfile.write(json.dumps(sample) + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())
                    total_saved += len(results)
                    print(f"[Saved] {len(results)} new sample(s) for '{citation}' "
                          f"(run total: {total_saved}).")
                else:
                    zero_result_citations.append(citation)
                    print(f"[Zero] '{citation}' produced no usable samples after retry.")

                completed.add(citation)
                save_checkpoint(completed)
                await asyncio.sleep(2.0)

    print(f"\n[Done] Saved {total_saved} new augmented samples to {OUTPUT_FILE}.")
    if skipped_no_context:
        print(f"[Note] {skipped_no_context} citation(s) had no matching source context and were skipped.")
    if zero_result_citations:
        print(f"[Note] {len(zero_result_citations)} citation(s) produced zero usable samples even after retry:")
        for c in zero_result_citations:
            print(f"    - {c}")
        print("  These may need a manual look, or a narrower source context window.")
    print(f"[Next] Review {OUTPUT_FILE}, then merge into {DATASET_FILE} with: "
          f"cat {OUTPUT_FILE} >> {DATASET_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Terminated] Stopped by user. Resumable on next run.")
        sys.exit(0)
