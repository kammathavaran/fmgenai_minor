from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tiktoken

model_id = "Yukang/Llama-2-7b-longlora-32k-ft"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    offload_folder="./offload",   # ensure this directory exists and has space
    offload_state_dict=True,      # helps with CPU RAM peaks
    torch_dtype="auto",
)


# Use cl100k_base tokenizer (compatible with GPT-4, 128k+ context)
encoding = tiktoken.get_encoding("cl100k_base")
path = "./prompts/passphrase_prompt.txt"
with open(path, "r", encoding="utf-8") as f:
    prompt = f.read()


token_count = len(encoding.encode(prompt))
print("Token length is :", token_count)


# Ensure long context
model.config.max_position_embeddings = 32768
tok.model_max_length = 32768


inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
print(tok.decode(out[0], skip_special_tokens=True))
