#!/usr/bin/env python3
"""
Builds a deterministic map from citation to source-text from your original
extracted_txt/ files for use as a RAG corpus and basis for generating training
data.

For each citation found, it captures a window of surrounding text as the
retrievable "definition" chunk.
"""
import os
import re
import json
import glob

INPUT_DIR = "extracted_txt"
OUTPUT_FILE = "citation_lookup.jsonl"

CITATION_PATTERN = re.compile(
    r"(Attachment\s+\d+,?\s+Section\s+\d[\d.]*|Section\s+\d[\d.]*|"
    r"\bR\d+(?:\.\d+)*\s+Part\s+\d[\d.]*|Part\s+\d[\d.]*|\bR\d+(?:\.\d+)*|\bM\d+(?:\.\d+)*|Requirement\s+\d+(?:\.\d+)*)",
    re.IGNORECASE
)

STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)

# Matches a standard reference without version suffix.
# Detects cross-references to a different standard  we don't mistakenly
# prepend the current file's own standard onto it.
PARTIAL_STANDARD_PATTERN = re.compile(r"\bCIP-\d{3}\b", re.IGNORECASE)

CHUNK_MARKER_PATTERN = re.compile(r"--- CHUNK \d+ ---\n?")

def strip_chunk_markers(text: str) -> str:
    return CHUNK_MARKER_PATTERN.sub("", text)

def disambiguate_citation(citation: str, source_file: str) -> str:
    """If a citation doesn't reference its own standard, prepend the
    standard number of its source file. Anything that already contains a full
    'CIP-XXX-Y' token or partial 'CIP-XXX' reference to a different standard,
    is unchanged."""
    citation = citation.strip()
    if STANDARD_PATTERN.search(citation) or PARTIAL_STANDARD_PATTERN.search(citation):
        return citation

    match = STANDARD_PATTERN.search(source_file or "")
    standard = match.group(1).upper() if match else None
    if not standard:
        return citation

    return f"{standard}, {citation}"

CONTEXT_WINDOW = 600

# RSAW documents are fillable audit worksheets that interleave rule text
# with large blocks of generic form boilerplate. These phrases are the
# form's own instructions to the auditor/entity, not rule content.
BOILERPLATE_PATTERNS = [
    re.compile(r"Registered Entity Response \(Required[^)]*\):", re.IGNORECASE),
    re.compile(r"Registered Entity Evidence \(Required\):", re.IGNORECASE),
    re.compile(r"Audit Team Evidence Reviewed \(This section to be completed by the Compliance Enforcement Authority\):", re.IGNORECASE),
    re.compile(r"Compliance Assessment Approach Specific to [^\n]*", re.IGNORECASE),
    re.compile(r"This section to be completed by the Compliance Enforcement Authority\.?", re.IGNORECASE),
    re.compile(r"Auditor Notes:", re.IGNORECASE),
    re.compile(r"Compliance Narrative:", re.IGNORECASE),
    re.compile(r"Provide a brief explanation, in your own words, of how you comply with this Requirement\.?", re.IGNORECASE),
    re.compile(r"References to supplied evidence, including links to the appropriate page, are recommended\.?", re.IGNORECASE),
    re.compile(r"Subject Matter Experts", re.IGNORECASE),
    re.compile(r"Identify the Subject Matter Expert\(s\) responsible for this Reliability Standard\.?", re.IGNORECASE),
    re.compile(r"\(Insert additional rows if needed\)", re.IGNORECASE),
    re.compile(r"Findings\s*\n", re.IGNORECASE),
    re.compile(r"Version History\s*\n.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"F\.\s*Associated Documents\s*\n.*?(?=\n[A-Z]\.|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Adopted by the NERC Board of Trustees\.?", re.IGNORECASE),
    re.compile(r"FERC order issued approving[^\n]*", re.IGNORECASE),
    re.compile(
        r"[Ss]ystems?\s*,\s*associated\s+with\s+communication\s+networks\s+and\s+data\s+communication\s+links.*?10\s*C\.F\.R\.\s*Section\s*73\.54\.?",
        re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"The systems,\s*structures,\s*and\s+components\s+that\s+are\s+regulated.*?10\s*C\.F\.R\.\s*Section\s*73\.54\.?",
        re.IGNORECASE | re.DOTALL
    ),
]


def strip_boilerplate(text: str) -> str:
    """Removes known RSAW form-template phrases from a context window."""
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)  # collapse gaps left by removals
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def boilerplate_density(text: str) -> float:
    """Proportion of the text's characters that are boilerplate, before stripping.
    Scores candidate context windows -- lower is better."""
    if not text:
        return 1.0
    boilerplate_chars = 0
    for pattern in BOILERPLATE_PATTERNS:
        for match in pattern.finditer(text):
            boilerplate_chars += len(match.group(0))
    return boilerplate_chars / len(text)


def context_score(entry: dict) -> tuple:
    """Higher is better. Prefers STANDARD-source context over RSAW-source,
    then prefers lower boilerplate density, then prefers longer substantive
    (post-strip) content."""
    is_standard_doc = "standard" in entry["source_file"].lower()
    density = boilerplate_density(entry["context"])
    substantive_length = len(strip_boilerplate(entry["context"]))
    return (is_standard_doc, -density, substantive_length)


def extract_citations_with_context(content: str, source_file: str) -> list:
    entries = []
    for match in CITATION_PATTERN.finditer(content):
        raw_citation = match.group(0).strip()
        citation = disambiguate_citation(raw_citation, source_file)

        start = max(0, match.start() - CONTEXT_WINDOW // 2)
        end = min(len(content), match.end() + CONTEXT_WINDOW // 2)

        # Snap outward to the nearest whitespace to avoid slicing mid-word.
        while start > 0 and not content[start - 1].isspace():
            start -= 1
        while end < len(content) and not content[end].isspace():
            end += 1

        context = content[start:end].strip()
        entries.append({
            "citation": citation,
            "context": context,
            "source_file": source_file
        })
    return entries


def normalize_citation_key(citation: str) -> str:
    """Collapses whitespace-only differences and converts 'R# Part #.#' to
    'Part #.#' -- RSAWs tend to reference sub-parts as 'R2 Part 2.6' while
    STANDARD docs use 'Part 2.6' for the same requirement; this ensures both
    forms compete for the same dedup slot instead of coexisting separately."""
    key = citation.lower().strip()
    key = re.sub(r'\s+', ' ', key)  # collapse whitespace runs first
    key = re.sub(
        r'^((?:cip-\d{3}-\d+(?:\.\d+)?,\s*)?)r\d+(?:\.\d+)*\s+(part\s+[\d.]+)$',
        r'\1\2',
        key
    )
    return key


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[Fatal] Input directory '{INPUT_DIR}' does not exist.")
        return

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not all_files:
        print(f"[Fatal] No .txt files found inside '{INPUT_DIR}'.")
        return

    all_entries = []
    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        content = strip_chunk_markers(content)
        if not content:
            continue
        entries = extract_citations_with_context(content, os.path.basename(file_path))
        all_entries.extend(entries)
        print(f"[{os.path.basename(file_path)}] Found {len(entries)} citation mentions.")

    # Flags any entry whose context starts or ends mid-word to reinforce
    # snapping logic.
    def looks_word_truncated(text: str) -> bool:
        if not text:
            return False
        # Checks whether either token looks abnormally short/fragmentary.
        first_word = text.split()[0] if text.split() else ""
        return first_word[:1].islower() and len(first_word) <= 3

    suspect_entries = [e for e in all_entries if looks_word_truncated(e["context"])]
    print(f"\n[Check] {len(suspect_entries)}/{len(all_entries)} context window(s) still look "
          f"potentially truncated after the word-boundary fix.")
    if suspect_entries:
        print("[Check] Sample of suspect entries (showing up to 5):")
        for e in suspect_entries[:5]:
            print(f"    - citation={e['citation']!r} context_start={e['context'][:60]!r}...")

    spot_check_citations = ["CIP-003-11, Section 3.1.3", "CIP-005-8, Part 2.6"]
    print("\n[Check] Spot-checking previously-broken citations:")
    for target in spot_check_citations:
        matches = [e for e in all_entries if e["citation"].lower() == target.lower()]
        if not matches:
            print(f"    - {target!r}: not found in this run's entries")
            continue
        for e in matches:
            print(f"    - {target!r} context now starts: {e['context'][:80]!r}")
    # -------------------------------------------------------------------------

    # Deduplicate: for each citation, pick the best candidate across all files
    best_by_citation = {}
    for entry in all_entries:
        key = normalize_citation_key(entry["citation"])
        if key not in best_by_citation or context_score(entry) > context_score(best_by_citation[key]):
            best_by_citation[key] = entry

    stripped_count = 0
    thin_context_citations = []
    MIN_SUBSTANTIVE_LENGTH = 80

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for entry in best_by_citation.values():
            cleaned_context = strip_boilerplate(entry["context"])
            if cleaned_context != entry["context"].strip():
                stripped_count += 1
            entry["context"] = cleaned_context
            if len(cleaned_context) < MIN_SUBSTANTIVE_LENGTH:
                thin_context_citations.append(entry["citation"])
            outfile.write(json.dumps(entry) + "\n")

    print(f"\n[Done] {len(best_by_citation)} unique citations written to {OUTPUT_FILE}")
    print(f"[Done] Stripped boilerplate from {stripped_count} context window(s).")
    if thin_context_citations:
        print(f"\n[Warning] {len(thin_context_citations)} citation(s) have very little substantive "
              f"context even after stripping boilerplate:")
        for c in thin_context_citations:
            print(f"    - {c}")
    print("This file can be embedded/indexed directly for RAG retrieval.")


if __name__ == "__main__":
    main()
