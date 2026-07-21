import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- 1. CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
DATASET_PATH = "nerc_cip_phi4_dataset.jsonl"
OUTPUT_DIR = "/mnt/ollama_repo/phi4_nerc_cip_lora"
MAX_SEQ_LENGTH = 4096

# --- 2. LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- 3. PREPARE FOR PEFT ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],  # Phi-4's actual fused layer names
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# --- 4. DATASET ---
full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def formatting_prompts_func(examples):
    texts = []
    for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": "You are a NERC CIP compliance audit assistant."},
            {"role": "user", "content": f"Context:\n{inp}\n\nInstruction:\n{instr}"},
            {"role": "assistant", "content": out}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

full_dataset = full_dataset.map(formatting_prompts_func, batched=True)

# 90/10 train/eval split so we can track generalization, not just memorization.
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

# --- 5. TRAINING ---
# Reduce num_train_epochs if eval_loss increases despite
# training loss decreasing as this trend suggests overfitting.
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_torch",           # bitsandbytes 8-bit optimizer broke gradient flow on this build
    gradient_checkpointing=False,  # also broke gradient flow when combined with 4-bit quant
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# --- 6. EXECUTE ---
# Snapshot one LoRA weight before training to verify updates.
probe_name = "base_model.model.model.layers.25.self_attn.qkv_proj.lora_B.default.weight"
probe_before = dict(model.named_parameters())[probe_name].detach().clone()

trainer.train()

probe_after = dict(model.named_parameters())[probe_name].detach()
diff = (probe_after - probe_before).abs().sum().item()
print(f"\n--- WEIGHT CHANGE CHECK ---")
print(f"Total absolute change in {probe_name}: {diff}")
print(f"Weights changed: {diff > 0}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
