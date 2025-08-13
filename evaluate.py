import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

dataset = Dataset.from_list(data)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = train_test_split['test']
logging.info(f"Evaluation dataset size: {len(eval_dataset)}")

# 2. Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model"
logging.info(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
logging.info("Model and tokenizer loaded successfully.")

# 3. Generate predictions
logging.info("Generating predictions...")
predictions = []
ground_truths = []

for example in tqdm(eval_dataset):
    instruction = example['instruction']
    input_text = example['input']
    prompt = f"### Instruction:\n{instruction}\nPlease provide the output in JSON format.\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.1)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the JSON part of the response
    try:
        # Use a more robust regex to find something that looks like a JSON object
        json_match = re.search(r'{\s*".*"\s*:\s*.*}', predicted_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Clean the string as much as possible
            json_str = json_str.replace('\n', ' ').replace('\\', '\\\\').replace('", "', '", "')
            
            # Try to parse the JSON
            try:
                predicted_json = json.loads(json_str)
                predictions.append(predicted_json)
                ground_truths.append(example['output'])
            except json.JSONDecodeError as e:
                # If parsing fails, try to find the largest valid JSON object within the string
                logging.warning(f"Initial JSON parsing failed. Attempting to recover. Error: {e}")
                max_len = 0
                best_json = None
                for i in range(len(json_str)):
                    for j in range(i, len(json_str)):
                        sub_str = json_str[i:j+1]
                        try:
                            parsed = json.loads(sub_str)
                            if len(sub_str) > max_len:
                                max_len = len(sub_str)
                                best_json = parsed
                        except json.JSONDecodeError:
                            continue
                if best_json:
                    predictions.append(best_json)
                    ground_truths.append(example['output'])
                else:
                    raise json.JSONDecodeError("Could not recover a valid JSON object", json_str, 0)
        else:
            raise json.JSONDecodeError("No JSON object found in response", predicted_text, 0)
    except (json.JSONDecodeError, IndexError) as e:
        logging.warning(f"Could not parse JSON from prediction: {predicted_text}. Error: {e}")
        # Add a dummy prediction to maintain alignment
        predictions.append({})
        ground_truths.append(example['output'])

logging.info("Predictions generated.")

# 4. Calculate metrics
logging.info("Calculating metrics...")

exact_matches = 0
y_true_need_anonymization = []
y_pred_need_anonymization = []
y_true_pii_identifiers = []
y_pred_pii_identifiers = []
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
bleu_scores = []

for pred, true in zip(predictions, ground_truths):
    try:
        # Exact match
        if pred == true:
            exact_matches += 1

        # 'need_anonymization' metric
        if 'need_anonymization' in pred and 'need_anonymization' in true:
            y_true_need_anonymization.append(1 if true.get('need_anonymization') == 'yes' else 0)
            y_pred_need_anonymization.append(1 if pred.get('need_anonymization') == 'yes' else 0)

        # PII identifiers metrics
        if 'pii_identifiers' in pred and 'pii_identifiers' in true:
            true_pii = set(true.get('pii_identifiers', []))
            pred_pii = set(pred.get('pii_identifiers', []))
            
            all_pii = sorted(list(true_pii.union(pred_pii)))
            
            y_true_pii_identifiers.append([1 if pii in true_pii else 0 for pii in all_pii])
            y_pred_pii_identifiers.append([1 if pii in pred_pii else 0 for pii in all_pii])

        # ROUGE and BLEU scores for the 'anonymized' text
        if 'anonymized' in pred and 'anonymized' in true:
            true_anonymized = true.get('anonymized', '')
            pred_anonymized = pred.get('anonymized', '')

            if true_anonymized and pred_anonymized:
                # ROUGE
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(true_anonymized, pred_anonymized)
                for key in rouge_scores.keys():
                    rouge_scores[key].append(scores[key].fmeasure)

                # BLEU
                reference = [true_anonymized.split()]
                candidate = pred_anonymized.split()
                smoothie = SmoothingFunction().method4
                bleu_scores.append(sentence_bleu(reference, candidate, smoothing_function=smoothie))
    except Exception as e:
        logging.error(f"Error processing prediction: {pred}. Error: {e}")
        continue


exact_match_ratio = exact_matches / len(ground_truths)

# Anonymization detection metrics
precision_anon, recall_anon, f1_anon, _ = precision_recall_fscore_support(
    y_true_need_anonymization, y_pred_need_anonymization, average='binary'
)
accuracy_anon = accuracy_score(y_true_need_anonymization, y_pred_need_anonymization)

# PII Identifier metrics
# Flatten the lists for sklearn metrics
y_true_pii_flat = [item for sublist in y_true_pii_identifiers for item in sublist]
y_pred_pii_flat = [item for sublist in y_pred_pii_identifiers for item in sublist]

precision_pii, recall_pii, f1_pii, _ = precision_recall_fscore_support(
    y_true_pii_flat, y_pred_pii_flat, average='binary'
)
accuracy_pii = accuracy_score(y_true_pii_flat, y_pred_pii_flat)


# Calculate average ROUGE and BLEU scores
avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0
avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0
avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
avg_bleu = np.mean(bleu_scores) if bleu_scores else 0

report = f"""
--- Evaluation Report ---
Exact Match Ratio: {exact_match_ratio:.4f}

--- Anonymization Detection ---
Accuracy: {accuracy_anon:.4f}
Precision: {precision_anon:.4f}
Recall: {recall_anon:.4f}
F1-Score: {f1_anon:.4f}

--- PII Identifier Detection ---
Accuracy: {accuracy_pii:.4f}
Precision: {precision_pii:.4f}
Recall: {recall_pii:.4f}
F1-Score: {f1_pii:.4f}

--- Text Generation Metrics (on 'anonymized' field) ---
Average ROUGE-1: {avg_rouge1:.4f}
Average ROUGE-2: {avg_rouge2:.4f}
Average ROUGE-L: {avg_rougeL:.4f}
Average BLEU Score: {avg_bleu:.4f}
-------------------------
"""

print(report)

# Save report to file
report_path = "evaluation_report.txt"
with open(report_path, "w") as f:
    f.write(report)
logging.info(f"Evaluation report saved to {report_path}")
