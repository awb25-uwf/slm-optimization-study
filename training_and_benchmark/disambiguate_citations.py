#!/usr/bin/env python3
"""
Fixes citation ambiguity: citations like "R1", "Section 4", "Part 2.4" mean
different things in different CIP standards, but were saved without their
parent standard. This uses the source_file field (already recorded per-sample)
to qualify them, e.g. "R1" in CIP-005-8_STANDARD_chunks.txt becomes
"CIP-005-8, R1". Any citation that already contains its own "CIP-XXX-Y" token
is left untouched, since it's already unambiguous.
"""
import json
import re

INPUT_FILE = "nerc_cip_phi4_dataset.jsonl"
OUTPUT_FILE = "nerc_cip_phi4_dataset_disambiguated.jsonl"

STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_standard(source_file: str) -> str:
    match = STANDARD_PATTERN.search(source_file)
    return match.group(1).upper() if match else None


def disambiguate_citation(citation: str, source_file: str) -> str:
    """If a citation doesn't already reference its own standard, prepend it,
    regardless of format."""
    citation = citation.strip()
    if STANDARD_PATTERN.search(citation):
        return citation  # already self-contained

    standard = extract_standard(source_file or "")
    if not standard:
        return citation

    return f"{standard}, {citation}"


def main():
    total = 0
    disambiguated = 0
    unresolved = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            citation = item.get("citation")
            source_file = item.get("source_file")

            if citation:
                new_citation = disambiguate_citation(citation, source_file)
                if new_citation != citation:
                    if extract_standard(source_file or ""):
                        disambiguated += 1
                    else:
                        unresolved += 1
                        print(f"[Warning] Could not resolve standard for citation "
                              f"'{citation}' from source_file '{source_file}' -- left unchanged.")
                item["citation"] = new_citation

                # Update the output text if it echoes the citation to keep
                # training text and the citation field stay consistent.
                if item.get("output") and citation.lower() in item["output"].lower() and new_citation != citation:
                    item["output"] = re.sub(
                        re.escape(citation), new_citation, item["output"], flags=re.IGNORECASE
                    )

            out.write(json.dumps(item) + "\n")

    print(f"\n[Done] Processed {total} samples.")
    print(f"[Done] Disambiguated {disambiguated} bare citations.")
    if unresolved:
        print(f"[Warning] {unresolved} citations could not be resolved (no standard found in source_file).")
    print(f"[Done] Wrote {OUTPUT_FILE}. Review it, then replace the original if it looks correct.")


if __name__ == "__main__":
    main()
