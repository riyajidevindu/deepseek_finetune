import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
import time
import json
import logging
from datasets import Dataset
import numpy as np
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions ---
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # In MB
    return 0

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # In MB

def benchmark_model(model_path, model_type, tokenizer_path=None, num_samples=50):
    logging.info(f"--- Benchmarking {model_path} ---")
    
    # --- Memory Usage ---
    torch.cuda.reset_peak_memory_stats()
    start_ram = get_ram_usage()
    
    if model_type == 'causal_lm':
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
    elif model_type == 'token_classification':
        model = AutoModelForTokenClassification.from_pretrained(model_path, device_map="auto")
    else:
        raise ValueError("Invalid model_type specified.")
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peak_gpu_mem_load = torch.cuda.max_memory_allocated() / 1024**2
    end_ram = get_ram_usage()
    ram_usage = end_ram - start_ram
    
    logging.info(f"Peak GPU memory usage during loading: {peak_gpu_mem_load:.2f} MB")
    logging.info(f"RAM usage for model loading: {ram_usage:.2f} MB")

    # --- Inference Speed ---
    # Load sample data
    with open('output.json', 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data).select(range(num_samples))

    inference_times = []
    for example in dataset:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            if model_type == 'causal_lm':
                model.generate(**inputs, max_new_tokens=100, use_cache=False)
            else: # token_classification
                model(**inputs)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    avg_inference_time = np.mean(inference_times)
    logging.info(f"Average inference time per sample: {avg_inference_time * 1000:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "model": model_path,
        "peak_gpu_memory_mb": peak_gpu_mem_load,
        "ram_usage_mb": ram_usage,
        "avg_inference_ms": avg_inference_time * 1000
    }

# --- Main Execution ---
if __name__ == "__main__":
    results = []
    
    # Benchmark DeepSeek Coder
    results.append(benchmark_model("./fine-tuned-model", "causal_lm"))
    
    # Benchmark DistilBERT
    results.append(benchmark_model(
        "./fine-tuned-model-token-classification", 
        "token_classification",
        tokenizer_path="distilbert-base-cased"
    ))
    
    # Benchmark Phi-3 Mini
    results.append(benchmark_model("./fine-tuned-model-phi3", "causal_lm"))

    # --- Print Report ---
    print("\n--- Benchmark Report ---")
    print(f"{'Model':<45} | {'Peak GPU Mem (MB)':<20} | {'RAM Usage (MB)':<20} | {'Avg Inference (ms)':<20}")
    print("-" * 110)
    for res in results:
        print(f"{res['model']:<45} | {res['peak_gpu_memory_mb']:.2f}{'':<15} | {res['ram_usage_mb']:.2f}{'':<15} | {res['avg_inference_ms']:.2f}")
    print("------------------------\n")

    # Save report to file
    report_path = "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("--- Benchmark Report ---\n")
        f.write(f"{'Model':<45} | {'Peak GPU Mem (MB)':<20} | {'RAM Usage (MB)':<20} | {'Avg Inference (ms)':<20}\n")
        f.write("-" * 110 + "\n")
        for res in results:
            f.write(f"{res['model']:<45} | {res['peak_gpu_memory_mb']:.2f}{'':<15} | {res['ram_usage_mb']:.2f}{'':<15} | {res['avg_inference_ms']:.2f}\n")
        f.write("------------------------\n")
    logging.info(f"Benchmark report saved to {report_path}")
