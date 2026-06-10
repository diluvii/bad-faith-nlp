# author: yawen xue
# date: 05/29/2026
# purpose: preprocessing data/llm.txt and data/llm_2.txt for use in main_llm.py

# assumes data is in the form of sentence pairs in a .txt file, with
# one sentence per line. the benign sentences appear on odd-number
# lines and the hostile sentences appear on the even-number lines

# had claude help with this one—prompted with asking for preparing
# data of the aforementioned format

import json
import random
from pathlib import Path

def load_pairs(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # remove empty lines, zip consecutive pairs
    lines = [l.strip() for l in lines if l.strip()]
    # must have even number of lines for sentence pairs
    if len(lines) % 2 != 0:
        lines = lines[:-1]
    pairs = [{"src": lines[i], "tgt": lines[i + 1]} for i in range(0, len(lines), 2)]
    return pairs

def split_and_save(pairs, out_dir, train=0.8, val=0.1, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val :],
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name:5s}: {len(rows):>5d} pairs → {p}")

def prepare_llm_data(path):
    pairs = load_pairs(path)
    split_and_save(pairs, "data/processed")
    print("\nData preparation done. Now we proceed to training.")
