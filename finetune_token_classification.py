import json
import logging
import torch
from datasets import Dataset, Features, ClassLabel, Value, Sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
import numpy as np

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

# Get all unique PII identifiers to create label mapping
all_pii_types = sorted(list(set(pii for item in data for pii in item['output'].get('pii_identifiers', []))))
labels = ["O"] + [f"B-{pii}" for pii in all_pii_types] + [f"I-{pii}" for pii in all_pii_types]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 2. Pre-process the data for token classification
model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["input"], truncation=True, is_split_into_words=False)
    
    labels = []
    for i, doc in enumerate(examples["input"]):
        doc_labels = np.full(len(tokenized_inputs.tokens(i)), -100) # -100 is ignored by the loss function
        
        pii_values = examples["output"][i].get("pii_values", [])
        pii_types = examples["output"][i].get("pii_identifiers", [])

        for pii_val, pii_type in zip(pii_values, pii_types):
            # Find all occurrences of the PII value in the input text
            for match in re.finditer(re.escape(pii_val), doc):
                start, end = match.span()
                
                # Convert character spans to token spans
                if end > 0:
                    token_start = tokenized_inputs.char_to_token(i, start)
                    token_end = tokenized_inputs.char_to_token(i, end - 1)

                    if token_start is not None and token_end is not None:
                        doc_labels[token_start] = label2id[f"B-{pii_type}"]
                        for token_idx in range(token_start + 1, token_end + 1):
                            doc_labels[token_idx] = label2id[f"I-{pii_type}"]
        
        # Set "O" label for all other tokens
        doc_labels[doc_labels == -100] = label2id["O"]
        labels.append(doc_labels.tolist())

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

import re

# Convert data to a Hugging Face Dataset
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42).values()

# 3. Model setup
logging.info(f"Setting up model: {model_name}")
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# 4. Training setup
logging.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results_token_classification",
    do_eval=True,
    eval_steps=500,
    logging_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 5. Start training
logging.info("Starting model training...")
trainer.train()
logging.info("Model training completed.")

# 6. Save the fine-tuned model
logging.info("Saving the fine-tuned model...")
trainer.save_model("./fine-tuned-model-token-classification")
logging.info("Model saved to './fine-tuned-model-token-classification'.")

logging.info("Fine-tuning process completed successfully.")
