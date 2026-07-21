import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "microsoft/phi-4"
ADAPTER_DIR = "/mnt/ollama_repo/phi4_14b_nerc_cip_lora"

# Fixed regression set: these 3 questions originally revealed a systematic bias
# where the model defaulted to citing CIP-007-7.1 regardless of actual topic,
# instead of the correct CIP-003-11. All three should cite CIP-003-11.
BENCHMARK_QUESTIONS = [
    {
        "instruction": "Does the utility company's current cyber security policy cover all required topics for high-impact BCS?",
        "input": "A utility company with high-impact Bulk Electric Systems (BCS)",
        "expected_standard": "CIP-003-11",
    },
    {
        "instruction": "How often should a Responsible Entity review its cyber security policies?",
        "input": "A distribution provider with medium-impact BCS",
        "expected_standard": "CIP-003-11",
    },
    {
        "instruction": "What measures are required for electronic access controls?",
        "input": "A power plant's control system",
        "expected_standard": "CIP-003-11",
    },
]

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("\n" + "=" * 80)
print("BIAS REGRESSION TEST -- all answers should cite CIP-003-11")
print("=" * 80)

results = []
for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
    messages = [
        {"role": "system", "content": "You are a NERC CIP compliance audit assistant."},
        {"role": "user", "content": f"Context:\n{q['input']}\n\nInstruction:\n{q['instruction']}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    correct = q["expected_standard"] in generated_text
    results.append(correct)

    print(f"\n--- QUESTION {i} ---")
    print(f"Instruction: {q['instruction']}")
    print(f"Expected standard: {q['expected_standard']}")
    print(f"Model output: {generated_text}")
    print(f"Correct standard cited: {'YES' if correct else 'NO'}")

print("\n" + "=" * 80)
print(f"RESULT: {sum(results)}/{len(results)} correctly cited {BENCHMARK_QUESTIONS[0]['expected_standard']}")
print("=" * 80)
