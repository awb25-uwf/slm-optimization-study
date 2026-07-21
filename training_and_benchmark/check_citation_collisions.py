#!/usr/bin/env python3
"""
Detects citation strings in citation_lookup.jsonl that likely refer to the
same underlying requirement but ended up as separate entries due to
formatting differences (e.g. 'R2 Part 2.6' vs 'Part 2.6', 'R1' vs
'Requirement 1', extra whitespace/newlines in the raw match, etc).

Groups citations by a normalized "signature" (the standard + numeric path,
stripped of label words), then reports any signature with more than one
raw citation string mapped to it.
"""
import json
import re
from collections import defaultdict

def signature(citation: str) -> str:
    """Reduce a citation to just its standard + numeric path, stripping
    label words (R, M, Part, Requirement, Section, Attachment) and
    collapsing whitespace, so equivalent references collapse together."""
    c = citation.lower()
    c = re.sub(r'\s+', ' ', c).strip()
    # Pull out the standard prefix (e.g. "cip-005-8,") separately so it's
    # not accidentally stripped by the label-word removal below.
    standard_match = re.match(r'^(cip-\d{3}-\d+(?:\.\d+)?),?\s*', c)
    standard = standard_match.group(1) if standard_match else ""
    rest = c[standard_match.end():] if standard_match else c
    # Strip label words, keep only numbers/dots/order
    rest_stripped = re.sub(r'\b(r|m|part|requirement|section|attachment)\b\.?', '', rest)
    rest_stripped = re.sub(r'[^\d.]', ' ', rest_stripped)
    rest_stripped = ' '.join(rest_stripped.split())  # collapse whitespace
    return f"{standard}|{rest_stripped}"

def main():
    entries = []
    with open("citation_lookup.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    groups = defaultdict(list)
    for e in entries:
        sig = signature(e["citation"])
        groups[sig].append(e)

    collisions = {sig: es for sig, es in groups.items() if len(es) > 1}

    print(f"[Report] {len(entries)} total citations, {len(groups)} unique signatures, "
          f"{len(collisions)} signature(s) with multiple raw citation strings.\n")

    for sig, es in sorted(collisions.items(), key=lambda x: -len(x[1])):
        print(f"--- signature: {sig!r} ({len(es)} variants) ---")
        for e in es:
            length = len(e["context"])
            print(f"    citation={e['citation']!r:45} len={length:4}  source={e['source_file']}")
        print()

if __name__ == "__main__":
    main()
