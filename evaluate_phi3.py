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

# 2. Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model-phi3"
logging.info(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. Generate predictions
logging.info("Generating predictions...")
predictions = []
ground_truths = []

for example in tqdm(eval_dataset):
    instruction = example['instruction']
    input_text = example['input']
    prompt = f"### Instruction:\n{instruction}\nPlease provide the output in JSON format.\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.1, use_cache=False)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        json_match = re.search(r'\{.*\}', predicted_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace('\n', ' ').replace('\\', '\\\\')
            predicted_json = json.loads(json_str)
            predictions.append(predicted_json)
            ground_truths.append(example['output'])
        else:
            raise json.JSONDecodeError("No JSON object found", predicted_text, 0)
    except (json.JSONDecodeError, IndexError) as e:
        logging.warning(f"Could not parse JSON from prediction: {predicted_text}. Error: {e}")
        predictions.append({})
        ground_truths.append(example['output'])

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
        if pred == true:
            exact_matches += 1

        if 'need_anonymization' in pred and 'need_anonymization' in true:
            y_true_need_anonymization.append(1 if true.get('need_anonymization') == 'yes' else 0)
            y_pred_need_anonymization.append(1 if pred.get('need_anonymization') == 'yes' else 0)

        if 'pii_identifiers' in pred and 'pii_identifiers' in true:
            true_pii = set(filter(None, true.get('pii_identifiers', [])))
            pred_pii = set(filter(None, pred.get('pii_identifiers', [])))
            all_pii = sorted(list(true_pii.union(pred_pii)))
            y_true_pii_identifiers.append([1 if pii in true_pii else 0 for pii in all_pii])
            y_pred_pii_identifiers.append([1 if pii in pred_pii else 0 for pii in all_pii])

        if 'anonymized' in pred and 'anonymized' in true:
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
    except Exception as e:
        logging.error(f"Error processing prediction: {pred}. Error: {e}")
        continue

exact_match_ratio = exact_matches / len(ground_truths) if ground_truths else 0
precision_anon, recall_anon, f1_anon, _ = precision_recall_fscore_support(y_true_need_anonymization, y_pred_need_anonymization, average='binary', zero_division=0)
accuracy_anon = accuracy_score(y_true_need_anonymization, y_pred_need_anonymization)

y_true_pii_flat = [item for sublist in y_true_pii_identifiers for item in sublist]
y_pred_pii_flat = [item for sublist in y_pred_pii_identifiers for item in sublist]
precision_pii, recall_pii, f1_pii, _ = precision_recall_fscore_support(y_true_pii_flat, y_pred_pii_flat, average='binary', zero_division=0)
accuracy_pii = accuracy_score(y_true_pii_flat, y_pred_pii_flat)

avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0
avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0
avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
avg_bleu = np.mean(bleu_scores) if bleu_scores else 0

report = f"""
--- Evaluation Report for microsoft/Phi-3-mini-4k-instruct ---
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
------------------------------------
"""

print(report)

report_path = "evaluation_report_phi3.txt"
with open(report_path, "w") as f:
    f.write(report)
logging.info(f"Evaluation report saved to {report_path}")
