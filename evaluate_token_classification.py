import json
import logging
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

# 2. Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model-token-classification"
logging.info(f"Loading model from: {model_path}")
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

# 3. Pre-process the data for evaluation
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["input"], truncation=True, is_split_into_words=False)
    
    labels = []
    for i, doc in enumerate(examples["input"]):
        doc_labels = np.full(len(tokenized_inputs.tokens(i)), -100)
        
        pii_values = examples["output"][i].get("pii_values", [])
        pii_types = examples["output"][i].get("pii_identifiers", [])

        for pii_val, pii_type in zip(pii_values, pii_types):
            for match in re.finditer(re.escape(pii_val), doc):
                start, end = match.span()
                
                if end > 0:
                    token_start = tokenized_inputs.char_to_token(i, start)
                    token_end = tokenized_inputs.char_to_token(i, end - 1)

                    if token_start is not None and token_end is not None:
                        doc_labels[token_start] = label2id[f"B-{pii_type}"]
                        for token_idx in range(token_start + 1, token_end + 1):
                            doc_labels[token_idx] = label2id[f"I-{pii_type}"]
        
        doc_labels[doc_labels == -100] = label2id["O"]
        labels.append(doc_labels.tolist())

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

import re

dataset = Dataset.from_list(data)
# For evaluation, we'll use the same split as in training
_, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# 4. Evaluate the model
logging.info("Evaluating the model...")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

predictions, labels, _ = trainer.predict(tokenized_eval_dataset)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (-100)
true_predictions = [
    [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# --- Start of Generation Metrics Calculation ---

def reconstruct_json_from_tokens(tokens, labels):
    pii_identifiers = []
    pii_values = []
    anonymized_parts = []
    
    current_pii_val = ""
    current_pii_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_pii_val:
                anonymized_parts.append(f"[{current_pii_type}]")
            current_pii_type = label[2:]
            current_pii_val = token
        elif label.startswith("I-"):
            current_pii_val += token.replace("##", "")
        else: # O label
            if current_pii_val:
                anonymized_parts.append(f"[{current_pii_type}]")
                pii_identifiers.append(current_pii_type)
                pii_values.append(current_pii_val)
                current_pii_val = ""
                current_pii_type = None
            anonymized_parts.append(token)
            
    if current_pii_val:
        anonymized_parts.append(f"[{current_pii_type}]")
        pii_identifiers.append(current_pii_type)
        pii_values.append(current_pii_val)

    return {
        "anonymized": " ".join(anonymized_parts),
        "need_anonymization": "yes" if pii_identifiers else "no",
        "pii_identifiers": pii_identifiers,
        "pii_values": pii_values
    }

reconstructed_predictions = []
for i in range(len(true_predictions)):
    tokens = tokenizer.convert_ids_to_tokens(tokenized_eval_dataset[i]['input_ids'])
    reconstructed_predictions.append(reconstruct_json_from_tokens(tokens, true_predictions[i]))

# Now calculate the original metrics
exact_matches = 0
y_true_need_anonymization = []
y_pred_need_anonymization = []
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
bleu_scores = []

ground_truths = [item['output'] for item in eval_dataset]

for pred, true in zip(reconstructed_predictions, ground_truths):
    if pred == true:
        exact_matches += 1
    
    y_true_need_anonymization.append(1 if true.get('need_anonymization') == 'yes' else 0)
    y_pred_need_anonymization.append(1 if pred.get('need_anonymization') == 'yes' else 0)

    true_anonymized = true.get('anonymized', '')
    pred_anonymized = pred.get('anonymized', '')

    if true_anonymized and pred_anonymized:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(true_anonymized, pred_anonymized)
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)

        reference = [true_anonymized.split()]
        candidate = pred_anonymized.split()
        smoothie = SmoothingFunction().method4
        bleu_scores.append(sentence_bleu(reference, candidate, smoothing_function=smoothie))

exact_match_ratio = exact_matches / len(ground_truths) if ground_truths else 0
precision_anon, recall_anon, f1_anon, _ = precision_recall_fscore_support(y_true_need_anonymization, y_pred_need_anonymization, average='binary', zero_division=0)
accuracy_anon = accuracy_score(y_true_need_anonymization, y_pred_need_anonymization)

avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0
avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0
avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
avg_bleu = np.mean(bleu_scores) if bleu_scores else 0

# PII Identifier metrics from reconstructed JSON
y_true_pii_identifiers = []
y_pred_pii_identifiers = []
for pred, true in zip(reconstructed_predictions, ground_truths):
    true_pii = set(filter(None, true.get('pii_identifiers', [])))
    pred_pii = set(filter(None, pred.get('pii_identifiers', [])))
    all_pii = sorted(list(true_pii.union(pred_pii)))
    y_true_pii_identifiers.append([1 if pii in true_pii else 0 for pii in all_pii])
    y_pred_pii_identifiers.append([1 if pii in pred_pii else 0 for pii in all_pii])

y_true_pii_flat = [item for sublist in y_true_pii_identifiers for item in sublist]
y_pred_pii_flat = [item for sublist in y_pred_pii_identifiers for item in sublist]

precision_pii, recall_pii, f1_pii, _ = precision_recall_fscore_support(y_true_pii_flat, y_pred_pii_flat, average='binary', zero_division=0)
accuracy_pii = accuracy_score(y_true_pii_flat, y_pred_pii_flat)


# --- End of Generation Metrics Calculation ---

# Flatten the lists for token classification report
flat_true_predictions = [item for sublist in true_predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]

token_report = classification_report(flat_true_labels, flat_true_predictions, zero_division=0)

# Combine reports
full_report = f"""
--- Token Classification Evaluation Report ---
{token_report}
--------------------------------------------

--- Comparative Generation Metrics ---
Exact Match Ratio: {exact_match_ratio:.4f}

--- Anonymization Detection ---
Accuracy: {accuracy_anon:.4f}
Precision: {precision_anon:.4f}
Recall: {recall_anon:.4f}
F1-Score: {f1_anon:.4f}

--- PII Identifier Detection (from reconstructed JSON) ---
Accuracy: {accuracy_pii:.4f}
Precision: {precision_pii:.4f}
Recall: {recall_pii:.4f}
F1-Score: {f1_pii:.4f}

--- Text Generation Metrics (on 'anonymized' field) ---
Average ROUGE-1: {avg_rouge1:.4f}
Average ROUGE-2: {avg_rouge2:.4f}
Average ROUGE-L: {avg_rougeL:.4f}
Average BLEU Score: {avg_bleu:.4f}
------------------------------------
"""

print(full_report)

# Save report to file
report_path = "evaluation_report_token_classification_combined.txt"
with open(report_path, "w") as f:
    f.write(full_report)
logging.info(f"Combined evaluation report saved to {report_path}")
