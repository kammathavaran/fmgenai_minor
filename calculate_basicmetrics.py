import pandas as pd
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Config (edit as needed)
CSV_PATH = "prompts/sql-console-for-yukang-longalpaca-12k.csv"  # Change to your actual path
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"  # Adjust if needed
MODEL_NAME = "RichardErkhov/Yukang_-_Llama-2-13b-chat-longlora-32k-sft-gguf"  # Fill in your running model's name

def get_model_response(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512
    }
    response = requests.post(LMSTUDIO_API_URL, json=payload)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("API error:", response.text)
        return ""

def compute_metrics(expected, generated):
    # Tokenize for BLEU and METEOR
    expected_tokens = word_tokenize(expected)
    generated_tokens = word_tokenize(generated)

    # BLEU-2 (customizable)
    bleu = sentence_bleu([expected_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0))

    # METEOR (requires tokenized input)
    meteor = meteor_score([expected_tokens], generated_tokens)

    # ROUGE-L (works with untokenized input)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(expected, generated)['rougeL'].fmeasure

    return bleu, rouge_l, meteor

def main():
    df = pd.read_csv(CSV_PATH)
    results = []
    for idx, row in df.iterrows():
        prompt = str(row['instruction'])
        expected = str(row['output'])
        generated = get_model_response(prompt)
        bleu, rouge_l, meteor = compute_metrics(expected, generated)
        results.append({
            "instruction": prompt,
            "expected_output": expected,
            "model_output": generated,
            "BLEU": bleu,
            "ROUGE_L": rouge_l,
            "METEOR": meteor
        })
        print(f"Row {idx}: BLEU={bleu:.3f} ROUGE_L={rouge_l:.3f} METEOR={meteor:.3f}")

    pd.DataFrame(results).to_csv("results/model_comparison_with_metrics.csv", index=False)

if __name__ == "__main__":
    main()
