#!/usr/bin/env python3
"""
Caps any standard's sample count at MAX_PER_STANDARD by randomly downsampling
(seeded for reproducibility). Standards already under the cap are left untouched.
This addresses class imbalance that was causing the model to default to
over-represented standards (e.g. CIP-007-7.1) regardless of actual topic.
"""
import json
import re
import random
from collections import defaultdict

INPUT_FILE = "nerc_cip_phi4_dataset.jsonl"
OUTPUT_FILE = "nerc_cip_phi4_dataset_balanced.jsonl"
MAX_PER_STANDARD = 90
SEED = 42

STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)


def main():
    by_standard = defaultdict(list)
    no_citation = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            citation = item.get("citation", "")
            match = STANDARD_PATTERN.search(citation)
            if match:
                by_standard[match.group(1).upper()].append(line)
            else:
                no_citation.append(line)

    random.seed(SEED)
    kept_lines = []
    print("Before -> After (per standard):")
    for standard in sorted(by_standard.keys(), key=lambda s: -len(by_standard[s])):
        samples = by_standard[standard]
        if len(samples) > MAX_PER_STANDARD:
            sampled = random.sample(samples, MAX_PER_STANDARD)
            print(f"  {standard}: {len(samples)} -> {len(sampled)} (downsampled)")
        else:
            sampled = samples
            print(f"  {standard}: {len(samples)} -> {len(sampled)} (unchanged)")
        kept_lines.extend(sampled)

    kept_lines.extend(no_citation)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in kept_lines:
            out.write(line + "\n")

    print(f"\n[Done] Total before: {sum(len(v) for v in by_standard.values()) + len(no_citation)}")
    print(f"[Done] Total after: {len(kept_lines)}")
    print(f"[Done] Wrote {OUTPUT_FILE}. Review it, then swap it in if it looks right.")


if __name__ == "__main__":
    main()
