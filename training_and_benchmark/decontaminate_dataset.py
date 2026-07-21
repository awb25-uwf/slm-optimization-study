#!/usr/bin/env python3
"""
Two cleanup passes:
1. Removes any sample referencing the fabricated "10 C.F.R. Section 73.54"
   citation.
2. Deduplicates exact-match entries by instruction, output, and citation
   still present after overlapping merge operations.
"""
import json

INPUT_FILE = "nerc_cip_phi4_dataset.jsonl"
OUTPUT_FILE = "nerc_cip_phi4_dataset_decontaminated.jsonl"

CONTAMINATION_MARKERS = ["C.F.R", "CFR", "73.54", "Nuclear Regulatory", "NRC"]


def is_contaminated(item: dict) -> bool:
    text = json.dumps(item)
    return any(marker in text for marker in CONTAMINATION_MARKERS)


def dedup_key(item: dict):
    return (item.get("instruction", "").strip(), item.get("output", "").strip())


def main():
    total = 0
    contaminated = 0
    duplicates = 0
    seen = set()
    kept = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)

            if is_contaminated(item):
                contaminated += 1
                continue

            key = dedup_key(item)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            kept.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for item in kept:
            out.write(json.dumps(item) + "\n")

    print(f"[Done] Total: {total}")
    print(f"[Done] Removed (contaminated): {contaminated}")
    print(f"[Done] Removed (exact duplicates): {duplicates}")
    print(f"[Done] Kept: {len(kept)}")
    print(f"[Done] Wrote {OUTPUT_FILE}. Review it, then swap it in if it looks right.")


if __name__ == "__main__":
    main()
