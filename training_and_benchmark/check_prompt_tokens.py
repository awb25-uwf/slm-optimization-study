#!/usr/bin/env python3
"""
Estimates the worst-case prompt token count across all benchmark scenarios,
to pick a safe num_ctx for the Ollama Modelfiles rather than guessing.
Worst case = RAG context (top-3) + KG context (top-3) + scenario_input +
instruction + prompt template boilerplate.

Uses the mini tokenizer as an approximation for token counts -- Phi-4-mini
and Phi-4 share the same tokenizer family, so counts should be very close;
this is meant to size a safety margin, not be exact to the token.
"""
import json
from transformers import AutoTokenizer

BENCHMARK_FILE = "nerc_benchmark_seed.jsonl"
TOKENIZER_SOURCE = "/mnt/ollama_repo/phi4_nerc_cip_lora"

PROMPT_TEMPLATE = """Context:
{context}

Instruction:
{instruction}

Respond with your compliance determination and reasoning. End your response with exactly these two lines:
Citation: <the specific standard and requirement/part/section you are citing>
Compliance Status: <compliant|partial|non_compliant>"""


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE, trust_remote_code=True)

    scenarios = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    print(f"Checking {len(scenarios)} scenarios...\n")

    max_tokens = 0
    max_scenario_id = None

    for scenario in scenarios:
        fake_rag_context = "[Retrieved -- EXAMPLE]: " + ("x " * 150) * 3
        fake_kg_context = "[KG-resolved -- EXAMPLE]: " + ("x " * 150) * 3
        combined_context = f"{scenario['scenario_input']}\n\n{fake_rag_context}\n{fake_kg_context}"

        prompt_text = PROMPT_TEMPLATE.format(
            context=combined_context,
            instruction=scenario["instruction"],
        )
        system_message = "You are a NERC CIP compliance audit assistant."

        full_text = system_message + "\n" + prompt_text
        token_count = len(tokenizer.encode(full_text))

        print(f"  {scenario['id']}: ~{token_count} tokens")

        if token_count > max_tokens:
            max_tokens = token_count
            max_scenario_id = scenario["id"]

    generation_budget = 300  # max_new_tokens / num_predict
    total_worst_case = max_tokens + generation_budget

    print(f"\nWorst case: {max_scenario_id} at ~{max_tokens} prompt tokens "
          f"(+{generation_budget} generation budget = ~{total_worst_case} total)")
    print(f"\nSuggested num_ctx: round up to a comfortable margin above {total_worst_case}, "
          f"e.g. {((total_worst_case // 1024) + 2) * 1024} or the next power-of-2-ish "
          f"value with headroom.")


if __name__ == "__main__":
    main()
