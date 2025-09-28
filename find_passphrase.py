import os
import math
import torch
import argparse
import csv
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
from llama_attn_replace import replace_llama_attn

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1)
    parser.add_argument('--flash_attn', type=bool, default=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--num_tests', type=int, default=None) # uses number of CSV rows
    args = parser.parse_args()
    return args

def extract_pass_key(prompt):
    # Change regex if pass key format differs!
    import re
    match = re.search(r'The pass key is (\w+)', prompt)
    return match.group(1) if match else ""

def run_passkey_test_from_csv(model, tokenizer, device, prompt, pass_key):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    len_token = input_ids.shape[-1]
    answer_ids = tokenizer(pass_key, return_tensors="pt").input_ids[:, 1:].to(device)
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1
    )
    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    is_correct = (model_answer == answer_ids[0].cpu()).all().item()
    prediction = tokenizer.decode(model_answer)
    return is_correct, len_token, prediction

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)
    print("base model", args.base_model)

    if args.flash_attn:
        replace_llama_attn(use_full=True)

    config = transformers.AutoConfig.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    context_size = args.context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    # ---- Load Rows from CSV ----
    report_lines = []
    with open(args.csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        n_tests = len(rows) if args.num_tests is None else min(args.num_tests, len(rows))
        for i, row in enumerate(rows[:n_tests]):
            version = row['version']
            position = row['position']
            prompt = row['content']
            pass_key = extract_pass_key(prompt)
            needle_token_idx = prompt.find(pass_key) if pass_key else -1
            is_correct, tokens_total, prediction = run_passkey_test_from_csv(model, tokenizer, device, prompt, pass_key)
            correct = int(is_correct)
            # Optional: Truncate prompt
            prompt_short = prompt[:120] + "..." if len(prompt) > 120 else prompt
            report_line = [
                version, position, tokens_total, needle_token_idx, prompt_short, prediction, correct
            ]
            report_lines.append(report_line)
            print(f"Test {i+1}: version={version}, position={position}, tokens_total={tokens_total}, correct={correct}")

    # Print CSV header and results
    print("version,position,tokens_total,needle_token_idx,prompt,prediction,correct")
    for line in report_lines:
        print(",".join([str(x) for x in line]))

if __name__ == "__main__":
    args = parse_config()
    main(args)
