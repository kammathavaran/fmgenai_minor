import csv
import requests
import os
from datetime import datetime
import tiktoken

LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
CSV_PATH = "prompts/passphrase.csv"
RESULTS_DIR = "results"
PASSCODE = "KINGFISH2233"
MODEL_NAME = "llama-3"
REPEATS = 3

ENCODING = tiktoken.get_encoding("cl100k_base")

def run_lmstudio_completion(prompt):
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    response = requests.post(LMSTUDIO_API_URL, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def extract_final_code(output):
    for line in output.splitlines():
        if "Final:" in line:
            return line.split("Final:")[-1].strip()
    return None

def find_needle_token_idx_tiktoken(context, secret, encoding):
    idx = context.find(secret)
    if idx == -1:
        return -1
    before_secret = context[:idx]
    return len(encoding.encode(before_secret))

def run_tests(rows, use_sliding_window=False, repeats=3):
    summary_rows = []
    for row in rows:
        version = row["version"]
        position = row["position"]
        context = row["content"]
        tokens_total = len(ENCODING.encode(context))
        needle_token_idx = find_needle_token_idx_tiktoken(context, PASSCODE, ENCODING)
        for repeat in range(repeats):
            prompt = context + "\n\nWhat is the passphrase? Output your answer in the format: Final: <code>"
            output = run_lmstudio_completion(prompt)
            pred = extract_final_code(output)
            correct = int(pred == PASSCODE)
            if correct == 0 and use_sliding_window:
                if sliding_window_search(context):
                    pred = PASSCODE
                    correct = 1
            summary_rows.append({
                "version": version,
                "position": position,
                "tokens_total": tokens_total,
                "needle_token_idx": needle_token_idx,
                "prediction": pred if pred is not None else "",
                "correct": correct
            })
    return summary_rows

def sliding_window_search(context, window_size=1200, step_size=400):
    context_length = len(context)
    for start in range(0, context_length, step_size):
        window = context[start:start+window_size]
        prompt = window + "\n\nWhat is the passphrase? Output your answer in the format: Final: <code>"
        output = run_lmstudio_completion(prompt)
        code = extract_final_code(output)
        if code == PASSCODE:
            return True
    return False

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as csvfile:
        return list(csv.DictReader(csvfile))

def print_summary_table(rows):
    # Print without prompt, as requested
    headers = ["version", "position", "tokens_total", "needle_token_idx", "prediction", "correct"]
    print(",".join(headers))
    for r in rows:
        print("{version},{position},{tokens_total},{needle_token_idx},{prediction},{correct}".format(**r))

if __name__ == "__main__":
    rows = read_csv(CSV_PATH)

    table_all = []
    # Baseline
    table_all.extend(run_tests(rows, use_sliding_window=False, repeats=REPEATS))
    # Sliding window
    table_all.extend(run_tests(rows, use_sliding_window=True, repeats=REPEATS))

    print_summary_table(table_all)
