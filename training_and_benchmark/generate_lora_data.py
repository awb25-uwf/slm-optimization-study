#!/usr/bin/env python3
import asyncio
import os
import json
import re
import sys
import glob
import shutil
import httpx

# --- CONFIGURATION ---
INPUT_DIR = "extracted_txt"
OUTPUT_FILE = "nerc_cip_phi4_dataset.jsonl"
CHECKPOINT_FILE = "generation_checkpoint.json"
MODEL_NAME = "deepseek-r1:32b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Minimum distinct training examples to generate per citation in each chunk
SAMPLES_PER_CITATION = 4

# Concurrency is strictly limited to 1 to conserve the Jetson's memory
SEMAPHORE_LIMIT = 1

CITATION_PATTERN = re.compile(
    r"(Attachment\s+\d+,?\s+Section\s+[\d.]+|Section\s+[\d.]+|R\d+(?:\.\d+)*|Requirement\s+\d+(?:\.\d+)*)",
    re.IGNORECASE
)

PROMPT_TEMPLATE = """You are an expert in NERC CIP compliance.

Step 1: Carefully read the source excerpt below. Identify EVERY distinct, specific citation
it contains (e.g. "Attachment 1, Section 2.1", "Section 2.12", "R1.2", "Requirement 1").
List each unique citation you find.

Step 2: For EACH citation you identified, generate exactly {samples_per_citation} DISTINCT,
diverse, realistic training examples that all correctly reference THAT SAME citation. Vary the
scenario, facility type, phrasing, and style across the examples for a given citation (auditor
verification questions, hypothetical utility infrastructure scenarios, direct technical
implementation/evidence-gathering requests) -- but the citation itself, and the compliance
determination it maps to, must stay factually consistent across all examples for that citation.

CRITICAL INSTRUCTIONS:
1. Every single example's "output" field MUST explicitly state the citation string exactly as
   found in the source text (e.g. "...under Attachment 1, Section 2.1.").
2. Every example MUST also include a separate "citation" field containing ONLY the citation
   string itself (e.g. "Attachment 1, Section 2.1"), with no other text.
3. Do not invent citations that are not present in the source excerpt.
4. SUBSTANCE OVER RESTATEMENT: The "output" must include the actual concrete criteria,
   thresholds, timeframes, or methods stated in the source excerpt for that citation --
   NOT a generic paraphrase of the requirement's title or topic. For example, if the source
   specifies "within 15 calendar days" or "at least once every 15 calendar months", the output
   must include that specific detail, not just say "the organization maintains a process."
   If the source excerpt for a citation lacks enough specific detail to do this, say so
   honestly in the output rather than inventing detail that isn't there.
5. VARY COMPLIANCE OUTCOMES: Across the {samples_per_citation} examples for a given citation,
   do NOT make all of them affirmative/compliant. Include a mix: some examples should describe
   full compliance, but at least one should describe a partial gap, a missing piece of evidence,
   or a plausible non-compliant scenario relevant to that citation, written the way an auditor
   would document a finding. Every example must also include a "compliance_status" field with
   one of exactly: "compliant", "partial", "non_compliant".
6. After thinking, provide your final response as a single valid JSON Array wrapped inside a
   ```json markdown code block. Include ALL examples for ALL citations you found in one array.

Source Excerpt:
\"\"\"{context}\"\"\"

Target Schema for each object inside the ```json block:
[
  {{
    "instruction": "Detailed audit question, technical request, or compliance scenario...",
    "input": "Specific utility context/infrastructure parameters or empty string...",
    "output": "The exact, legally and technically compliant response, including the citation AND concrete criteria/thresholds from the source...",
    "citation": "Attachment 1, Section 2.1",
    "compliance_status": "compliant"
  }},
  ...
]
"""


def clean_and_parse_json_array(raw_text: str) -> list:
    """Extracts and parses a JSON array from the markdown code blocks."""
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


STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)


def disambiguate_citation(citation: str, file_path: str) -> str:
    citation = citation.strip()
    if STANDARD_PATTERN.search(citation):
        return citation

    """If a citation doesn't reference its own standard, prepend the standard
    extracted from file_path regardless of format."""
    match = STANDARD_PATTERN.search(file_path)
    if not match:
        return citation

    return f"{match.group(1).upper()}, {citation}"


def validate_and_filter(samples: list, source_content: str, file_path: str) -> list:
    """Drops samples that lack a citation and those whose citation doesn't appear
    in the source text."""
    valid = []
    dropped = 0
    for item in samples:
        citation = item.get("citation", "").strip()
        output = item.get("output", "")
        instruction = item.get("instruction", "")

        if not citation or not output or not instruction:
            dropped += 1
            continue

        # Citation must genuinely appear in the source excerpt (case-insensitive,
        # tolerant of minor punctuation differences) to guard against hallucination.
        # IMPORTANT: validate against the ORIGINAL bare citation here, since the
        # source text won't contain the disambiguated "STANDARD, R1" form verbatim.
        normalized_citation = re.sub(r'[,.]', '', citation.lower())
        normalized_source = re.sub(r'[,.]', '', source_content.lower())
        if normalized_citation not in normalized_source:
            dropped += 1
            continue

        # Citation should also actually be echoed in the output text itself.
        if citation.lower() not in output.lower():
            dropped += 1
            continue

        # Now disambiguate bare requirement citations (e.g. "R1" -> "CIP-005-8, R1")
        # so different standards' R1/R2/etc. don't collide in the final dataset.
        disambiguated = disambiguate_citation(citation, file_path)
        if disambiguated != citation:
            output = re.sub(re.escape(citation), disambiguated, output, flags=re.IGNORECASE)
        citation = disambiguated

        item["citation"] = citation
        item["output"] = output
        item["source_file"] = os.path.basename(file_path)
        if not item.get("compliance_status"):
            item["compliance_status"] = "compliant"  # default for backward compatibility
        valid.append(item)

    if dropped:
        print(f"[Validation] Dropped {dropped} sample(s) with missing/hallucinated citations.")
    return valid


def load_checkpoint() -> set:
    """Returns the set of source filenames already fully processed in a prior run."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed_files", []))
    except (json.JSONDecodeError, OSError):
        print("[Warning] Checkpoint file was unreadable/corrupt. Starting fresh.")
        return set()


def save_checkpoint(completed_files: set):
    """Persists progress after every successfully processed file."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed_files": sorted(completed_files)}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def count_citations_in_text(content: str) -> int:
    """Quick heuristic count of distinct citations in the source, used only for logging."""
    found = set(m.group(0).strip() for m in CITATION_PATTERN.finditer(content))
    return len(found)


async def process_file(file_path: str, client: httpx.AsyncClient, sem: asyncio.Semaphore):
    async with sem:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"[Warning] Skipped empty file: {file_path}")
                return None

            approx_citations = count_citations_in_text(content)
            print(f"\n[Processing] -> {file_path} (~{approx_citations} distinct citations detected, "
                  f"targeting {SAMPLES_PER_CITATION} samples each)")

            prompt = PROMPT_TEMPLATE.format(context=content, samples_per_citation=SAMPLES_PER_CITATION)

            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.5,
                    "num_ctx": 16384  # bumped up since output arrays are now larger
                }
            }

            full_response = ""
            async with client.stream("POST", OLLAMA_API_URL, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    print(f"[Error] API returned status {response.status_code} for {file_path}")
                    print(f"[Error Detail] {error_body.decode(errors='replace')}")
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

            print("\n[Generation Complete. Extracting JSON Array...]")
            parsed_array = clean_and_parse_json_array(full_response)
            validated = validate_and_filter(parsed_array, content, file_path)
            return validated

        except Exception as e:
            print(f"\n[Failed] Exception processing {file_path}: {e}", file=sys.stderr)
            return None


async def main():
    force_fresh = "--fresh" in sys.argv

    if not os.path.exists(INPUT_DIR):
        print(f"[Fatal] Input directory '{INPUT_DIR}' does not exist.")
        sys.exit(1)

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not all_files:
        print(f"[Fatal] No .txt files found inside '{INPUT_DIR}'.")
        sys.exit(1)

    completed_files = set()

    if force_fresh:
        # Explicit full restart: back up any existing output/checkpoint and wipe them.
        if os.path.exists(OUTPUT_FILE):
            backup_file = OUTPUT_FILE + ".bak"
            print(f"[--fresh] Backing up existing dataset to {backup_file}...")
            shutil.move(OUTPUT_FILE, backup_file)
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        print("[--fresh] Starting a clean run from scratch.")
    else:
        completed_files = load_checkpoint()
        if completed_files:
            print(f"[Resume] Found checkpoint with {len(completed_files)} file(s) already completed. "
                  f"Skipping those and appending new results to {OUTPUT_FILE}.")
        else:
            print("[Start] No checkpoint found -- this will be treated as a fresh run.")

    remaining_files = [f for f in all_files if os.path.basename(f) not in completed_files]

    if not remaining_files:
        print("[Done] All files already completed according to checkpoint. "
              "Run with --fresh to regenerate everything.")
        return

    print(f"[Plan] {len(remaining_files)}/{len(all_files)} chunk(s) remaining. "
          f"Aiming for {SAMPLES_PER_CITATION} examples per distinct citation found in each chunk.")

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(connect=15.0, read=5400.0, write=30.0, pool=15.0)  # 90 min read timeout -- large RSAW files can run long at ~6 t/s
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    total_saved = 0
    # Append mode is essential for resuming -- overwriting ('w') would destroy
    # everything saved by previous runs before this one.
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as outfile:
            for i, file_path in enumerate(remaining_files, 1):
                print(f"\n--- Chunk Progress: {i}/{len(remaining_files)} remaining "
                      f"({len(completed_files)}/{len(all_files)} total done) ---")

                results_list = await process_file(file_path, client, sem)

                if results_list:
                    for sample in results_list:
                        outfile.write(json.dumps(sample) + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())
                    total_saved += len(results_list)
                    print(f"[Saved] Committed {len(results_list)} validated samples "
                          f"(this run's total: {total_saved}).")

                    # Only mark a file complete once its results are safely on disk --
                    # this is what makes resuming after a crash safe.
                    completed_files.add(os.path.basename(file_path))
                    save_checkpoint(completed_files)
                else:
                    print(f"[Skipped] Failed to get valid dataset array for: {file_path} "
                          f"(NOT marked complete -- will retry on next run).")

                await asyncio.sleep(2.0)

    print(f"\n[Done] This run saved {total_saved} new samples. "
          f"{len(completed_files)}/{len(all_files)} files completed overall.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Terminated] Stopped by user. Done.")
        sys.exit(0)
