#!/usr/bin/env python3
"""
Merges each LoRA adapter into its base model weights and saves a standalone
HF checkpoint for GGUF conversion with convert_hf_to_gguf.py.

Run once per adapter (mini, then 14b) before converting to GGUF.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MERGE_TARGETS = {
    "phi4mini": {
        "base_model_name": "microsoft/Phi-4-mini-instruct",
        "adapter_dir": "/mnt/ollama_repo/phi4_nerc_cip_lora",
        "output_dir": "/mnt/ollama_repo/phi4mini_lora_merged",
    },
    "phi4_14b": {
        "base_model_name": "microsoft/phi-4",
        "adapter_dir": "/mnt/ollama_repo/phi4_14b_nerc_cip_lora",
        "output_dir": "/mnt/ollama_repo/phi4_14b_lora_merged",
    },
}


def merge_one(size_label):
    target = MERGE_TARGETS[size_label]
    print(f"\n{'=' * 70}")
    print(f"Merging {size_label}: {target['base_model_name']} + {target['adapter_dir']}")
    print(f"{'=' * 70}")

    print("Loading base model (bf16, matching training/inference dtype throughout this study; "
          "fp32 was tested for the mini model and produced identical merge output to bf16, "
          "confirming the base+adapter vs merged divergence is a computation-order artifact, "
          "not a precision issue -- so bf16 is used throughout, which is also required to fit "
          "the 14B model in memory on this hardware)...")
    model = AutoModelForCausalLM.from_pretrained(
        target["base_model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(target["adapter_dir"], trust_remote_code=True)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, target["adapter_dir"])

    print("Merging adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {target['output_dir']}...")
    model.save_pretrained(target["output_dir"])
    tokenizer.save_pretrained(target["output_dir"])

    print(f"[Done] {size_label} merged checkpoint written to {target['output_dir']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=list(MERGE_TARGETS.keys()),
                         default=list(MERGE_TARGETS.keys()),
                         help="Which adapters to merge, e.g. --sizes phi4mini")
    args = parser.parse_args()

    for size_label in args.sizes:
        merge_one(size_label)
        # Free memory before the next (potentially much larger) model.
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
