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

# Configure logging to print timestamped messages
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1. Load and prepare the dataset
logging.info("Starting dataset loading and preparation...")
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
logging.info("Converting data to Hugging Face Dataset object...")
dataset = Dataset.from_list(data)
logging.info(f"Dataset created with {len(dataset)} samples.")

# Split the dataset
logging.info("Splitting dataset into training and validation sets (90% train, 10% validation)...")
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
logging.info(f"Training set size: {len(train_dataset)}")
logging.info(f"Validation set size: {len(eval_dataset)}")

# 2. Format the dataset into the required prompt format
logging.info("Defining the prompt formatting function...")
def formatting_prompts_func(example):
    # The instruction, input, and output fields are extracted from the example dictionary.
    # The output field, which is a JSON object, is serialized into a string representation using json.dumps().
    instruction = example['instruction']
    input_text = example['input']
    output_text = json.dumps(example['output'])
    
    # A formatted string is constructed using an f-string.
    # This string follows a specific template with placeholders for the instruction, input, and response.
    text = f"""### Instruction:
{instruction}
Please provide the output in JSON format.

### Input:
{input_text}

### Response:
{output_text}"""
    
    # The function returns the formatted string.
    # This string will be used as the input for the language model during training.
    return text

logging.info("Prompt formatting function defined.")

# 3. Model and Tokenizer setup
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
logging.info(f"Setting up model and tokenizer for '{model_name}'...")

# Quantization configuration
logging.info("Configuring BitsAndBytes for 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
logging.info("BitsAndBytes configuration created.")

# LoRA configuration
logging.info("Configuring LoRA...")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
logging.info("LoRA configuration created.")

# Load model
logging.info(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
logging.info("Model loaded successfully.")

# Load tokenizer
logging.info(f"Loading tokenizer for: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
logging.info("Tokenizer loaded successfully.")

# 4. Training setup
logging.info("Setting up training arguments...")
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
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
logging.info("Training arguments configured.")

logging.info("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_arguments,
)
logging.info("SFTTrainer initialized.")

# 5. Start training
logging.info("Starting model training...")
trainer.train()
logging.info("Model training completed.")

# 6. Save the fine-tuned model
logging.info("Saving the fine-tuned model...")
trainer.save_model("./fine-tuned-model")
logging.info("Model saved to './fine-tuned-model'.")

# 7. Evaluate the model
logging.info("Evaluating the model...")
eval_results = trainer.evaluate()
logging.info(f"Evaluation results: {eval_results}")
perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
logging.info(f"Perplexity: {perplexity}")
print(f"Final Perplexity: {perplexity}")

logging.info("Fine-tuning process completed successfully.")
