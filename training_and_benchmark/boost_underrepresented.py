#!/usr/bin/env python3
"""
Same generation logic as augment_dataset.py (citation-grounded, drift-checked,
compliance-status-varied) but targets only citations from under-
represented standards, with a higher SAMPLES_PER_CITATION to compensate.

Run build_citation_lookup.py first if you haven't.
"""
import asyncio
import os
import json
import re
import sys
import httpx

LOOKUP_FILE = "citation_lookup.jsonl"
OUTPUT_FILE = "nerc_cip_phi4_dataset_boosted.jsonl"
CHECKPOINT_FILE = "boost_checkpoint.json"
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Which standards to boost, and how many examples per citation for each.
TARGET_STANDARDS = {
    "CIP-002-8": 8,
    "CIP-006-7": 12,
}

SEMAPHORE_LIMIT = 1
STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)

PROMPT_TEMPLATE = """You are an expert in NERC CIP compliance auditing.

You are given the SPECIFIC source text for one citation: "{citation}"

Source text for this citation:
\"\"\"{context}\"\"\"

Generate exactly {n} DISTINCT, realistic audit training examples for this exact citation.

CRITICAL INSTRUCTIONS:
1. SUBSTANCE OVER RESTATEMENT: Each "output" must reference the actual concrete criteria,
   thresholds, timeframes, or methods stated in the source text above -- not a generic
   paraphrase of the topic.
2. VARY COMPLIANCE OUTCOMES: Do not make all {n} examples affirmative. Include a mix --
   some compliant, some partial gaps, some non-compliant scenarios, written the way an
   auditor would document a finding. Each example needs a "compliance_status" field:
   exactly one of "compliant", "partial", "non_compliant".
3. STAY ON THIS EXACT CITATION -- THIS IS CRITICAL: The source text above may include
   sibling or parallel sections/parts. That is normal document structure -- use it ONLY to
   understand what makes "{citation}" distinct from its siblings, but your {n} examples
   must describe ONLY what "{citation}" itself requires. Do NOT mention, restate, or
   contrast against what any sibling section/part/requirement covers.
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

NUMBERED_REF_PATTERNS = [
    ("section", re.compile(r"(?:attachment\s+\d+,?\s+)?section\s+([\d.]+)", re.IGNORECASE)),
    ("part", re.compile(r"part\s+([\d.]+)", re.IGNORECASE)),
    ("r", re.compile(r"\br(\d+(?:\.\d+)*)\b", re.IGNORECASE)),
    ("requirement", re.compile(r"requirement\s+(\d+(?:\.\d+)*)", re.IGNORECASE)),
]


def find_all_refs(text: str) -> set:
    refs = set()
    for category, pattern in NUMBERED_REF_PATTERNS:
        for match in pattern.finditer(text):
            refs.add((category, match.group(1).rstrip(".")))
    return refs


def output_matches_citation(output: str, expected_citation: str) -> bool:
    expected_refs = find_all_refs(expected_citation)
    if not expected_refs:
        return True
    output_refs = find_all_refs(output)
    for category, number in output_refs:
        expected_numbers = {n for c, n in expected_refs if c == category}
        if expected_numbers and number not in expected_numbers:
            return False
    return True


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


def validate(samples: list, expected_citation: str, source_file: str) -> list:
    valid = []
    for item in samples:
        citation = item.get("citation", "").strip()
        output = item.get("output", "")
        instruction = item.get("instruction", "")
        status = item.get("compliance_status", "").strip().lower()
        if not citation or not output or not instruction:
            continue
        if citation.lower() != expected_citation.lower():
            continue
        if not output_matches_citation(output, expected_citation):
            continue
        if status not in {"compliant", "partial", "non_compliant"}:
            status = "compliant"
        item["compliance_status"] = status
        item["source_file"] = source_file
        valid.append(item)
    return valid


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


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


async def generate_for_citation(citation, context, n, source_file, client, sem):
    async with sem:
        prompt = PROMPT_TEMPLATE.format(citation=citation, context=context, n=n)
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True,
                   "options": {"temperature": 0.4, "num_ctx": 8192}}
        full_response = ""
        try:
            async with client.stream("POST", OLLAMA_API_URL, json=payload) as response:
                if response.status_code != 200:
                    print(f"[Error] {response.status_code} for '{citation}'")
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
            print(f"\n[Complete] '{citation}'")
            parsed = clean_and_parse_json_array(full_response)
            return validate(parsed, citation, source_file)
        except Exception as e:
            print(f"\n[Failed] '{citation}': {e}", file=sys.stderr)
            return None


async def main():
    lookup = load_jsonl(LOOKUP_FILE)
    if not lookup:
        print(f"[Fatal] {LOOKUP_FILE} not found or empty. Run build_citation_lookup.py first.")
        sys.exit(1)

    targets = []
    for entry in lookup:
        match = STANDARD_PATTERN.search(entry["citation"])
        if match and match.group(1).upper() in TARGET_STANDARDS:
            targets.append(entry)

    print(f"[Plan] Found {len(targets)} citations across target standards: "
          f"{list(TARGET_STANDARDS.keys())}")

    completed = load_checkpoint()
    remaining = [t for t in targets if t["citation"] not in completed]
    print(f"[Plan] {len(completed)} already done, {len(remaining)} remaining.")

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(connect=15.0, read=1800.0, write=30.0, pool=15.0)
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    total_saved = 0
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as outfile:
            for i, entry in enumerate(remaining, 1):
                citation = entry["citation"]
                match = STANDARD_PATTERN.search(citation)
                n = TARGET_STANDARDS.get(match.group(1).upper(), 5)

                print(f"\n--- Boosting {i}/{len(remaining)}: '{citation}' (n={n}) ---")
                results = await generate_for_citation(
                    citation, entry["context"], n, entry.get("source_file", ""), client, sem
                )

                if results:
                    for sample in results:
                        outfile.write(json.dumps(sample) + "\n")
                    outfile.flush()
                    total_saved += len(results)
                    print(f"[Saved] {len(results)} sample(s) for '{citation}' (run total: {total_saved}).")

                completed.add(citation)
                save_checkpoint(completed)
                await asyncio.sleep(2.0)

    print(f"\n[Done] Saved {total_saved} new samples to {OUTPUT_FILE}.")
    print(f"[Next] Merge with: cat {OUTPUT_FILE} >> nerc_cip_phi4_dataset_balanced.jsonl")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Terminated] Resumable on next run.")
        sys.exit(0)
