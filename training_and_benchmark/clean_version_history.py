#!/usr/bin/env python3
"""
Strips Version History / Associated Documents boilerplate from citation_lookup.jsonl.
"""
import json
import re

INPUT_FILE = "citation_lookup.jsonl"
OUTPUT_FILE = "citation_lookup_cleaned.jsonl"

# The "Version History" header itself often isn't present in the extracted
# window -- citations near the end of a document produce windows that start
# ALREADY inside the changelog table, past where the header would appear.
# These phrases are the table's actual recurring content and reliably signal
# "everything from here on is changelog noise, not substantive rule text."
VERSION_HISTORY_MARKERS = [
    r"Adopted\s+by\s+the\s+NERC\s+Board\s+of\s+Trustees",
    r"Adopted\s+by\s+the\s+Standards\s+Committee",
    r"Docket\s+No\.",
    r"FERC\s+[Oo]rder\s+No\.",
    r"Revised\s+to\s+address",
    r"Revised\s+version\s+addresses",
    r"Replaces\s+the\s+version\s+adopted",
]
VERSION_HISTORY_PATTERN = re.compile("|".join(VERSION_HISTORY_MARKERS), re.IGNORECASE)


def clean(text: str) -> str:
    match = VERSION_HISTORY_PATTERN.search(text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


def main():
    total = 0
    changed = 0
    now_thin = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            entry = json.loads(line)
            original = entry.get("context", "")
            cleaned = clean(original)
            if cleaned != original.strip():
                changed += 1
            if len(cleaned) < 80:
                now_thin += 1
                print(f"[Warning] '{entry.get('citation')}' has thin/empty context after "
                      f"cleaning ({len(cleaned)} chars) -- its entire extracted window "
                      f"was likely version-history table noise.")
            entry["context"] = cleaned
            out.write(json.dumps(entry) + "\n")

    print(f"\n[Done] Processed {total} entries, cleaned {changed}, {now_thin} now thin/empty.")
    print(f"[Done] Wrote {OUTPUT_FILE}. Review it, then swap it in if it looks right.")


if __name__ == "__main__":
    main()
