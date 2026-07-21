#!/usr/bin/env python3
"""
Mechanical scoring for the NERC CIP audit benchmark. Scores a model's raw text
output against a benchmark scenario's ground truth on three independent axes:

  1. citation_correct   -- normalized (standard, core_id) match, same logic
                            used elsewhere in this project (augment_dataset.py,
                            spot_check_dataset.py) so a citation phrased
                            differently but referring to the same requirement
                            still counts as correct.
  2. status_correct      -- exact match on compliant/partial/non_compliant,
                            extracted from the output text via an explicit
                            "Compliance Status:" line that eval prompts should
                            require the model to produce (see run_benchmark.py).
  3. key_fact_recall      -- fraction of required_key_facts found as
                            case-insensitive substrings in the output.
"""
import re
from dataclasses import dataclass, field


STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBERED_REF_PATTERNS = [
    ("section", re.compile(r"(?:attachment\s+\d+,?\s+)?section\s+([\d.]+)", re.IGNORECASE)),
    ("part", re.compile(r"part\s+([\d.]+)", re.IGNORECASE)),
    ("r", re.compile(r"\br(\d+(?:\.\d+)*)\b", re.IGNORECASE)),
    ("requirement", re.compile(r"requirement\s+(\d+(?:\.\d+)*)", re.IGNORECASE)),
]

STATUS_LINE_PATTERN = re.compile(
    r"compliance\s+status\s*:\s*(compliant|partial|non[_\s-]?compliant)",
    re.IGNORECASE
)


def normalize_citation(citation: str):
    """Returns (standard, core_id) for normalized matching."""
    if not citation:
        return None
    standard_match = STANDARD_PATTERN.search(citation)
    standard = standard_match.group(1).upper() if standard_match else None

    core_id = None
    for category, pattern in NUMBERED_REF_PATTERNS:
        match = pattern.search(citation.lower())
        if match:
            core_id = (category, match.group(1).rstrip("."))
            break

    if not standard and not core_id:
        return None
    return (standard, core_id)


def extract_citation_from_output(output_text: str):
    """Finds the citation last referenced in a model's free-text output since
    models often give their actual answer's citation near the end."""
    matches = list(STANDARD_PATTERN.finditer(output_text))
    if not matches:
        return None
    last_match = matches[-1]
    window_start = last_match.start()
    window_end = min(len(output_text), last_match.end() + 60)
    return output_text[window_start:window_end]


def extract_status_from_output(output_text: str):
    """Extracts compliance status. Returns None if not found -- eval
    prompts must request this format for status scoring to work; see
    run_benchmark.py's prompt template."""
    match = STATUS_LINE_PATTERN.search(output_text)
    if not match:
        return None
    raw = match.group(1).lower().replace(" ", "_").replace("-", "_")
    return raw


@dataclass
class ScenarioScore:
    scenario_id: str
    citation_correct: bool
    status_correct: bool
    status_found: bool
    key_fact_recall: float
    key_facts_found: list = field(default_factory=list)
    key_facts_missing: list = field(default_factory=list)
    extracted_citation: str = None
    extracted_status: str = None

    @property
    def full_pass(self) -> bool:
        return self.citation_correct and self.status_correct and self.key_fact_recall == 1.0


def score_response(scenario: dict, model_output: str) -> ScenarioScore:
    extracted_citation = extract_citation_from_output(model_output)
    extracted_norm = normalize_citation(extracted_citation) if extracted_citation else None
    expected_norm = normalize_citation(scenario["ground_truth_citation"])
    citation_correct = bool(extracted_norm and expected_norm and extracted_norm == expected_norm)

    extracted_status = extract_status_from_output(model_output)
    status_found = extracted_status is not None
    expected_status = scenario["ground_truth_compliance_status"].lower().strip()
    status_correct = bool(status_found and extracted_status == expected_status)

    required_facts = scenario.get("required_key_facts", [])
    output_lower = model_output.lower()
    found = [f for f in required_facts if f.lower() in output_lower]
    missing = [f for f in required_facts if f.lower() not in output_lower]
    recall = len(found) / len(required_facts) if required_facts else 1.0

    return ScenarioScore(
        scenario_id=scenario["id"],
        citation_correct=citation_correct,
        status_correct=status_correct,
        status_found=status_found,
        key_fact_recall=recall,
        key_facts_found=found,
        key_facts_missing=missing,
        extracted_citation=extracted_citation,
        extracted_status=extracted_status,
    )


def aggregate_scores(scores: list) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n_scenarios": n,
        "citation_accuracy": sum(s.citation_correct for s in scores) / n,
        "status_accuracy": sum(s.status_correct for s in scores) / n,
        "status_extraction_rate": sum(s.status_found for s in scores) / n,
        "mean_key_fact_recall": sum(s.key_fact_recall for s in scores) / n,
        "full_pass_rate": sum(s.full_pass for s in scores) / n,
    }
