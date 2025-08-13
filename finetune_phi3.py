import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Load and prepare the dataset
logging.info("Loading and preparing the dataset...")
dataset_path = 'output.json'
try:
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    logging.info(f"Successfully loaded dataset from {dataset_path}")
except FileNotFoundError:
    logging.error(f"Dataset file not found at {dataset_path}")
    exit()
except json.JSONDecodeError:
    logging.error(f"Error decoding JSON from {dataset_path}")
    exit()

# Convert to Hugging Face Dataset object
dataset = Dataset.from_list(data)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# 2. Format the dataset
def formatting_prompts_func(example):
    text = f"### Instruction:\n{example['instruction']}\nPlease provide the output in JSON format.\n\n### Input:\n{example['input']}\n\n### Response:\n{json.dumps(example['output'])}"
    return text

# 3. Model and Tokenizer setup
model_name = "microsoft/Phi-3-mini-4k-instruct"
logging.info(f"Setting up model and tokenizer for '{model_name}'...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj"],
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Training setup
training_arguments = TrainingArguments(
    output_dir="./results_phi3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=1000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    do_eval=True,
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_arguments,
)

# 5. Start training
logging.info("Starting model training...")
trainer.train()
logging.info("Model training completed.")

# 6. Save the fine-tuned model
trainer.save_model("./fine-tuned-model-phi3")
logging.info("Model saved to './fine-tuned-model-phi3'.")

logging.info("Fine-tuning process completed successfully.")
