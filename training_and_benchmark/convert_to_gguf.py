#!/usr/bin/env python3
"""
Converts all four model states to GGUF (bf16, unquantized) using llama.cpp's
convert_hf_to_gguf.py. Base models (repo IDs, not local dirs) are resolved
to their local HF cache snapshot path first via snapshot_download -- this
does NOT re-download anything already cached, it just returns the path.

Requires: llama.cpp cloned with convert_hf_to_gguf.py present (see
CONVERT_SCRIPT_PATH below -- update if your clone lives elsewhere).
"""
import argparse
import os
import subprocess
from huggingface_hub import snapshot_download

CONVERT_SCRIPT_PATH = "/mnt/ollama_repo/llama.cpp/convert_hf_to_gguf.py"
GGUF_OUTPUT_DIR = "/mnt/ollama_repo/gguf"

# Each entry: source is either a local directory (already merged/downloaded)
# or an HF repo ID (resolved to a local snapshot dir before conversion).
CONVERT_TARGETS = {
    "phi4mini_base": {
        "source": "microsoft/Phi-4-mini-instruct",
        "is_repo_id": True,
    },
    "phi4mini_lora": {
        "source": "/mnt/ollama_repo/phi4mini_lora_merged",
        "is_repo_id": False,
    },
    "phi4_14b_base": {
        "source": "microsoft/phi-4",
        "is_repo_id": True,
    },
    "phi4_14b_lora": {
        "source": "/mnt/ollama_repo/phi4_14b_lora_merged",
        "is_repo_id": False,
    },
}


def convert_one(label):
    target = CONVERT_TARGETS[label]
    print(f"\n{'=' * 70}")
    print(f"Converting {label}")
    print(f"{'=' * 70}")

    if target["is_repo_id"]:
        print(f"Resolving local snapshot path for {target['source']} "
              f"(uses existing HF cache -- no re-download expected)...")
        local_dir = snapshot_download(repo_id=target["source"])
        print(f"Resolved to: {local_dir}")
    else:
        local_dir = target["source"]
        if not os.path.isdir(local_dir):
            print(f"[Error] Expected local directory not found: {local_dir}")
            return False

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(GGUF_OUTPUT_DIR, f"{label}.gguf")

    cmd = [
        "python", CONVERT_SCRIPT_PATH,
        local_dir,
        "--outtype", "bf16",
        "--outfile", outfile,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[Error] Conversion failed for {label} (exit code {result.returncode})")
        return False

    print(f"[Done] {label} -> {outfile}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", choices=list(CONVERT_TARGETS.keys()),
                         default=list(CONVERT_TARGETS.keys()),
                         help="Which model states to convert, e.g. --targets phi4mini_base phi4mini_lora")
    args = parser.parse_args()

    if not os.path.isfile(CONVERT_SCRIPT_PATH):
        print(f"[Fatal] convert_hf_to_gguf.py not found at {CONVERT_SCRIPT_PATH}. "
              f"Update CONVERT_SCRIPT_PATH at the top of this script if your llama.cpp "
              f"clone lives elsewhere.")
        return

    results = {}
    for label in args.targets:
        results[label] = convert_one(label)

    print(f"\n{'#' * 70}")
    print("SUMMARY")
    print(f"{'#' * 70}")
    for label, ok in results.items():
        print(f"  {label}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
