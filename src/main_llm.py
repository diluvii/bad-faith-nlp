# author: yawen xue
# date: 05/29/2026 - 06/10/2026
# purpose: bad-faith interpretation, this time by training a seq2seq model on claude-generated sentence pairs

# data comes from claude's sonnet 4.6. see data/llm.txt and data/llm2.txt

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from prepare_llm_data import prepare_llm_data
from train import train

# init my model
# comment one of these out
# 1 - passive aggressive style
# 2 - hostile intent style
# PATH = "../data/llm.txt"
PATH = "../data/llm2.txt"

# prepare data & train
# you can comment these out on subsequent runs of the same model
prepare_llm_data(PATH)
train()

# now that the model has been trained, let's use it
MODEL_PATH = "checkpoints/style_transfer/final"
PREFIX = "rewrite style: "

# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Type sentences to translate (blank line to quit):")

while True:
    text = input("> ").strip()
    if not text:
        print("> Goodbye.")
        break

    inputs = PREFIX + text

    enc = tokenizer(
        inputs,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=128)

    result = tokenizer.decode(out[0], skip_special_tokens=True)

    print(">", result, "\n")