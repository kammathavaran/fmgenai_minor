import requests
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
from difflib import SequenceMatcher
from tqdm import tqdm
from scipy.stats import bootstrap

LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
PROMPT_CSV_PATH = "prompts/multi-language-questions-set.csv"
RESULTS_DIR = "results"
RAW_RESULTS_PATH = os.path.join(RESULTS_DIR, "raw_predictions.csv")
SUMMARY_REPORT_PATH = os.path.join(RESULTS_DIR, "summary_report.txt")
FILTER_PROMPT = "Instruction: Keep answers precise and respond in words rather than sentences. Return just the answer when possible. Answer in the same language the question was asked. "


os.makedirs(RESULTS_DIR, exist_ok=True)

def lmstudio_inference(prompt, max_tokens=128):
    """ Call the LMStudio API for a single prompt and return the output string. """
    payload = {
        "messages": [{"role": "user", "content": FILTER_PROMPT+prompt}],
        "max_tokens": max_tokens,
        "temperature": 0  # deterministic, for accuracy assessment
    }
    resp = requests.post(LMSTUDIO_API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

def exact_match(pred, gold):
    return int(pred.strip() == gold.strip())

def string_f1(pred, gold):
    """Token-wise F1 for short answer strings."""
    pred_tokens, gold_tokens = pred.split(), gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    intersection = len(set(pred_tokens) & set(gold_tokens))
    precision = intersection / len(pred_tokens)
    recall = intersection / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def mean_fluency_rating(pred):
    """Heuristic fluency scoring 1=bad, 5=flawless (refine for your task)."""
    length = len(pred.split())
    # Very basic heuristic: length penalty for <3 tokens, flawless grammar = 5, else 3-4.
    if length < 3:
        return 1
    elif pred.endswith('.') and length > 5:
        return 5
    else:
        return 3

def error_type(pred, gold):
    # Heuristics: (expand based on domain knowledge)
    if not pred:
        return "No output"
    if pred.strip().lower() == gold.strip().lower():
        return "None"
    if SequenceMatcher(None, pred, gold).ratio() < 0.5:
        return "Misinterpretation"
    if any(char in pred for char in ['?', '!', '@']):
        return "Script issues"
    if pred.isascii() != gold.isascii():
        return "Script issues"
    if (set(pred.split()) & set(gold.split())):
        return "Terminology drift"
    return "Other"

df = pd.read_csv(PROMPT_CSV_PATH)
reports = []
raw_records = []

# Run evaluation per condition and collect metrics
for idx, row in tqdm(df.iterrows(), total=len(df)):
    id_ = row['ID']
    for cond in ['L1', 'L2', 'L3', 'CS']:
        prompt, gold = row[cond], row[f"{cond}_Answer"]
        if pd.isnull(prompt) or pd.isnull(gold):
            continue
        prediction = lmstudio_inference(prompt)
        correct = exact_match(prediction, gold)
        f1 = string_f1(prediction, gold)
        fluency = mean_fluency_rating(prediction)
        etype = error_type(prediction, gold)
        raw_records.append({
            'id': id_,
            'condition': cond,
            'prompt': prompt,
            'gold': gold,
            'prediction': prediction,
            'correct': correct,
            'fluency': fluency,
            'error_type': etype,
            'f1': f1
        })

# Raw results CSV
raw_df = pd.DataFrame(raw_records)
raw_df.to_csv(RAW_RESULTS_PATH, index=False)

# Summary: per-condition stats
summary = []
for cond in ['L1', 'L2', 'L3', 'CS']:
    cond_df = raw_df[raw_df['condition'] == cond]
    acc = cond_df['correct'].mean()
    f1_mean = cond_df['f1'].mean()
    fluency_mean = cond_df['fluency'].mean()
    fluency_ci = bootstrap((cond_df['fluency'].values,), np.mean).confidence_interval
    summary.append({
        'Condition': cond,
        'Accuracy': f"{acc:.3f}",
        'F1': f"{f1_mean:.3f}",
        'Fluency Mean': f"{fluency_mean:.2f}",
        'Fluency 95% CI': f"[{fluency_ci.low:.2f}, {fluency_ci.high:.2f}]"
    })

summary_str = (
    "Per-condition accuracy, F1, mean fluency, and 95% CI:\n"
    + pd.DataFrame(summary).to_string(index=False) + "\n"
)

error_types = raw_df.groupby('error_type').size().to_dict()
summary_str += "\nError type breakdown:\n"
for k, v in error_types.items():
    summary_str += f"  {k}: {v}\n"

with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(summary_str)

print("Report complete. See:", RAW_RESULTS_PATH, "and", SUMMARY_REPORT_PATH)
