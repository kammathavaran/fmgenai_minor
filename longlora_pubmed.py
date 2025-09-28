#!/usr/bin/env python3
"""
LLM Evaluation Script for PubMed Dataset (Updated with CSV output)
Evaluates LLM performance on yes/no classification and text generation tasks.
Saves all responses to CSV files and final metrics to results folder.
"""

import json
import requests
import os
import csv
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import tiktoken
import numpy as np
import time
import warnings
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Constants
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
JSON_FILE_PATH = "prompts/custom_pubmed_dataset.json"
RESULTS_FOLDER = "results"
PREFIX_A = "Answer this question with a yes/no answer only. If unsure, respond as 'unsure'. Answer in one word, yes, no or unsure"
PREFIX_B = ""

def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return {}

def create_prompt(contexts: List[str], question: str, prefix: str = "") -> str:
    """Create prompt by concatenating contexts and question with optional prefix."""
    concatenated_contexts = " ".join(contexts)
    full_context = concatenated_contexts + " " + question

    if prefix:
        return f"{prefix}. {full_context}"
    else:
        return full_context

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return len(text.split())  # Fallback to word count

def query_llm(prompt: str, api_url: str) -> Tuple[str, int]:
    """Query the LLM and return response and token count."""
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    token_count = count_tokens(prompt)

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip(), token_count
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return "", token_count
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "", token_count

def normalize_yes_no_answer(answer: str) -> str:
    """Normalize LLM response to yes/no/unsure."""
    answer_lower = answer.lower().strip()

    if 'yes' in answer_lower and 'no' not in answer_lower:
        return 'yes'
    elif 'no' in answer_lower and 'yes' not in answer_lower:
        return 'no'
    elif 'unsure' in answer_lower or 'uncertain' in answer_lower:
        return 'unsure'
    else:
        # If unclear, mark as unsure
        return 'unsure'

def calculate_accuracy_f1(predictions: List[str], ground_truth: List[str]) -> Tuple[float, float]:
    """Calculate accuracy and macro F1 score."""
    # Convert to binary for F1 calculation (yes=1, no/unsure=0)
    pred_binary = [1 if pred == 'yes' else 0 for pred in predictions]
    truth_binary = [1 if truth == 'yes' else 0 for truth in ground_truth]

    accuracy = accuracy_score(truth_binary, pred_binary)
    f1_macro = f1_score(truth_binary, pred_binary, average='macro', zero_division=0)

    return accuracy, f1_macro

def calculate_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU, METEOR, and ROUGE scores with improved BLEU handling."""
    bleu_scores = []
    meteor_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    # Initialize smoothing function for BLEU to prevent warnings
    smoothing = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, ref in zip(predictions, references):
        # Skip empty predictions or references
        if not pred.strip() or not ref.strip():
            bleu_scores.append(0.0)
            meteor_scores.append(0.0)
            rouge_scores['rouge1'].append(0.0)
            rouge_scores['rouge2'].append(0.0)
            rouge_scores['rougeL'].append(0.0)
            continue

        # BLEU score with smoothing to prevent warnings
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())

        try:
            # Suppress BLEU warnings and use smoothing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="nltk.translate.bleu_score")
                bleu_score = sentence_bleu(
                    [ref_tokens], 
                    pred_tokens, 
                    smoothing_function=smoothing.method4  # Epsilon smoothing
                )
            bleu_scores.append(bleu_score)
        except Exception:
            bleu_scores.append(0.0)

        # METEOR score
        try:
            meteor_score_val = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor_score_val)
        except Exception:
            meteor_scores.append(0.0)

        # ROUGE scores
        try:
            rouge_result = scorer.score(ref, pred)
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
        except Exception:
            rouge_scores['rouge1'].append(0.0)
            rouge_scores['rouge2'].append(0.0)
            rouge_scores['rougeL'].append(0.0)

    return {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
        'rouge1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
        'rouge2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
        'rougeL': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
    }

def save_responses_to_csv(results: Dict, data: Dict, results_folder: str) -> str:
    """Save all responses to CSV file."""
    csv_data = []

    # Get the item IDs in the same order as processed
    item_ids = list(data.keys())[:len(results['prefix_a']['predictions'])]

    for i, item_id in enumerate(item_ids):
        item_data = data[item_id]

        csv_row = {
            'item_id': item_id,
            'question': item_data.get('QUESTION', ''),
            'contexts': ' | '.join(item_data.get('CONTEXTS', [])),
            'ground_truth_decision': item_data.get('final_decision', ''),
            'ground_truth_long_answer': item_data.get('LONG_ANSWER', ''),

            # Prefix A results
            'prefix_a_prompt': f"{PREFIX_A}. {' '.join(item_data.get('CONTEXTS', []))} {item_data.get('QUESTION', '')}",
            'prefix_a_response': results['prefix_a']['raw_responses'][i],
            'prefix_a_normalized': results['prefix_a']['predictions'][i],
            'prefix_a_tokens': results['prefix_a']['token_counts'][i],
            'prefix_a_correct': 1 if results['prefix_a']['predictions'][i] == item_data.get('final_decision', '').lower() else 0,

            # Prefix B results
            'prefix_b_prompt': f"{' '.join(item_data.get('CONTEXTS', []))} {item_data.get('QUESTION', '')}",
            'prefix_b_response': results['prefix_b']['raw_responses'][i],
            'prefix_b_tokens': results['prefix_b']['token_counts'][i],

            # Additional metadata
            'year': item_data.get('YEAR', ''),
            'labels': ' | '.join(item_data.get('LABELS', [])),
            'meshes': ' | '.join(item_data.get('MESHES', []))
        }
        csv_data.append(csv_row)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(results_folder, f'llm_responses_{timestamp}.csv')

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = csv_data[0].keys() if csv_data else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    return csv_filename

def save_metrics_to_files(accuracy_a: float, f1_a: float, text_metrics_b: Dict, 
                         results: Dict, results_folder: str) -> Tuple[str, str]:
    """Save metrics to both JSON and CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Prepare comprehensive metrics
    comprehensive_metrics = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_items_processed': len(results['prefix_a']['predictions']),
            'api_url': LMSTUDIO_API_URL,
            'prefix_a': PREFIX_A,
            'prefix_b': PREFIX_B,
            'bleu_smoothing': 'method4 (epsilon smoothing)',
            'warnings_suppressed': True
        },
        'classification_metrics': {
            'accuracy': float(accuracy_a),
            'macro_f1': float(f1_a)
        },
        'text_generation_metrics': {
            'bleu': float(text_metrics_b['bleu']),
            'meteor': float(text_metrics_b['meteor']),
            'rouge1': float(text_metrics_b['rouge1']),
            'rouge2': float(text_metrics_b['rouge2']),
            'rougeL': float(text_metrics_b['rougeL'])
        },
        'token_statistics': {
            'prefix_a': {
                'mean_tokens': float(np.mean(results['prefix_a']['token_counts'])),
                'std_tokens': float(np.std(results['prefix_a']['token_counts'])),
                'min_tokens': int(np.min(results['prefix_a']['token_counts'])),
                'max_tokens': int(np.max(results['prefix_a']['token_counts']))
            },
            'prefix_b': {
                'mean_tokens': float(np.mean(results['prefix_b']['token_counts'])),
                'std_tokens': float(np.std(results['prefix_b']['token_counts'])),
                'min_tokens': int(np.min(results['prefix_b']['token_counts'])),
                'max_tokens': int(np.max(results['prefix_b']['token_counts']))
            }
        }
    }

    # Save to JSON
    json_filename = os.path.join(results_folder, f'evaluation_metrics_{timestamp}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_metrics, f, indent=2, ensure_ascii=False)

    # Save metrics summary to CSV
    metrics_csv_data = [
        {'metric_type': 'Classification', 'metric_name': 'Accuracy', 'prefix': 'A', 'score': accuracy_a},
        {'metric_type': 'Classification', 'metric_name': 'Macro F1', 'prefix': 'A', 'score': f1_a},
        {'metric_type': 'Text Generation', 'metric_name': 'BLEU', 'prefix': 'B', 'score': text_metrics_b['bleu']},
        {'metric_type': 'Text Generation', 'metric_name': 'METEOR', 'prefix': 'B', 'score': text_metrics_b['meteor']},
        {'metric_type': 'Text Generation', 'metric_name': 'ROUGE-1', 'prefix': 'B', 'score': text_metrics_b['rouge1']},
        {'metric_type': 'Text Generation', 'metric_name': 'ROUGE-2', 'prefix': 'B', 'score': text_metrics_b['rouge2']},
        {'metric_type': 'Text Generation', 'metric_name': 'ROUGE-L', 'prefix': 'B', 'score': text_metrics_b['rougeL']}
    ]

    csv_metrics_filename = os.path.join(results_folder, f'evaluation_metrics_{timestamp}.csv')
    with open(csv_metrics_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['metric_type', 'metric_name', 'prefix', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_csv_data)

    return json_filename, csv_metrics_filename

def process_dataset(data: Dict, max_items: int = 60) -> Dict:
    """Process the dataset and collect results."""
    results = {
        'prefix_a': {
            'predictions': [],
            'ground_truth': [],
            'token_counts': [],
            'raw_responses': []
        },
        'prefix_b': {
            'predictions': [],
            'ground_truth': [],
            'token_counts': [],
            'raw_responses': []
        }
    }

    items_processed = 0

    for item_id, item_data in data.items():
        if items_processed >= max_items:
            break

        print(f"Processing item {items_processed + 1}/{max_items}: {item_id}")

        # Extract data
        contexts = item_data.get('CONTEXTS', [])
        question = item_data.get('QUESTION', '')
        final_decision = item_data.get('final_decision', '').lower()
        long_answer = item_data.get('LONG_ANSWER', '')

        # Create prompts
        prompt_a = create_prompt(contexts, question, PREFIX_A)
        prompt_b = create_prompt(contexts, question, PREFIX_B)

        # Query LLM for Prefix A
        print(f"  Querying with Prefix A...")
        response_a, tokens_a = query_llm(prompt_a, LMSTUDIO_API_URL)
        normalized_a = normalize_yes_no_answer(response_a)

        results['prefix_a']['predictions'].append(normalized_a)
        results['prefix_a']['ground_truth'].append(final_decision)
        results['prefix_a']['token_counts'].append(tokens_a)
        results['prefix_a']['raw_responses'].append(response_a)

        # Query LLM for Prefix B
        print(f"  Querying with Prefix B...")
        response_b, tokens_b = query_llm(prompt_b, LMSTUDIO_API_URL)

        results['prefix_b']['predictions'].append(response_b)
        results['prefix_b']['ground_truth'].append(long_answer)
        results['prefix_b']['token_counts'].append(tokens_b)
        results['prefix_b']['raw_responses'].append(response_b)

        items_processed += 1

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)

    return results

def main():
    """Main execution function."""
    # Create results folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load dataset
    data = load_json_data(JSON_FILE_PATH)

    if not data:
        print("Failed to load dataset. Exiting.")
        return

    print(f"Loaded {len(data)} items from dataset")

    # Process the first 60 items (or all if fewer)
    max_items = min(60, len(data))
    results = process_dataset(data, max_items=max_items)

    print("\nProcessing complete! Calculating metrics...")

    # Calculate metrics for Prefix A (yes/no classification)
    accuracy_a, f1_a = calculate_accuracy_f1(
        results['prefix_a']['predictions'],
        results['prefix_a']['ground_truth']
    )

    # Calculate text metrics for Prefix B (with improved BLEU)
    text_metrics_b = calculate_text_metrics(
        results['prefix_b']['predictions'],
        results['prefix_b']['ground_truth']
    )

    # Save responses to CSV
    print("\nSaving responses to CSV...")
    csv_responses_file = save_responses_to_csv(results, data, RESULTS_FOLDER)
    print(f"✓ Responses saved to: {csv_responses_file}")

    # Save metrics to files
    print("\nSaving metrics to files...")
    json_metrics_file, csv_metrics_file = save_metrics_to_files(
        accuracy_a, f1_a, text_metrics_b, results, RESULTS_FOLDER
    )
    print(f"✓ Metrics JSON saved to: {json_metrics_file}")
    print(f"✓ Metrics CSV saved to: {csv_metrics_file}")

    # Display summary tables
    summary_data = {
        'Metric Type': ['Classification', 'Classification', 'Text Generation', 'Text Generation', 'Text Generation', 'Text Generation', 'Text Generation'],
        'Metric': ['Accuracy', 'Macro F1', 'BLEU', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
        'Prefix': ['A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'Score': [
            f"{accuracy_a:.4f}",
            f"{f1_a:.4f}",
            f"{text_metrics_b['bleu']:.4f}",
            f"{text_metrics_b['meteor']:.4f}",
            f"{text_metrics_b['rouge1']:.4f}",
            f"{text_metrics_b['rouge2']:.4f}",
            f"{text_metrics_b['rougeL']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Token statistics table
    token_data = {
        'Prefix': ['A', 'B'],
        'Mean Tokens': [
            f"{np.mean(results['prefix_a']['token_counts']):.1f}",
            f"{np.mean(results['prefix_b']['token_counts']):.1f}"
        ],
        'Std Tokens': [
            f"{np.std(results['prefix_a']['token_counts']):.1f}",
            f"{np.std(results['prefix_b']['token_counts']):.1f}"
        ],
        'Min Tokens': [
            f"{np.min(results['prefix_a']['token_counts'])}",
            f"{np.min(results['prefix_b']['token_counts'])}"
        ],
        'Max Tokens': [
            f"{np.max(results['prefix_a']['token_counts'])}",
            f"{np.max(results['prefix_b']['token_counts'])}"
        ]
    }

    token_df = pd.DataFrame(token_data)
    print("\n" + "="*60)
    print("TOKEN USAGE STATISTICS")
    print("="*60)
    print(token_df.to_string(index=False))

    print("\n" + "="*60)
    print("FILES CREATED IN RESULTS FOLDER:")
    print("="*60)
    print(f"Detailed responses: {os.path.basename(csv_responses_file)}")
    print(f"Metrics (JSON):     {os.path.basename(json_metrics_file)}")
    print(f"Metrics (CSV):      {os.path.basename(csv_metrics_file)}")
    print("\n✓ Evaluation complete! All results saved to results folder.")

if __name__ == "__main__":
    main()
