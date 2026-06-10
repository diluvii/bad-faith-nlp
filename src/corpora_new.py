# author: yawen xue
# date: 06/10/2026
# purpose: build benign and hostile corpora from AITA comments

import re
import random

from pathlib import Path
from datasets import load_dataset

# paths
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FILE_B = DATA / "benign_new.txt"
FILE_H = DATA / "hostile_new.txt"

# config
TARGET_B = 10000
TARGET_H = 10000

MIN_WORDS = 6
MAX_WORDS = 80

SRC_AITA = "OsamaBsher/AITA-Reddit-Dataset"


# helper functions
def hf_stream(src):
    ds = load_dataset(src, split="train", streaming=True)
    yield from ds


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\*\*.*?\*\*", " ", text)
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(nta|nah|yta|esh)\b", "", text, flags=re.I)
    text = re.sub(r"\b(nta|nah|yta|esh)\b", "", text, flags=re.I)
    return text.strip()


def length_ok(text):
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not length_ok(p):
            continue
        out.append(p)
    return out


def dedup(sentences):
    seen = set()
    out = []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(s)

    return out


# ——————— build benign_new.txt ———————
def build_benign():
    sentences = []
    for rec in hf_stream(SRC_AITA):
        verdict = (rec.get("verdict") or "").upper().strip()
        if verdict not in ("NTA", "NAH"):
            continue
        for field in ("comment1", "comment2"):
            text = clean(rec.get(field, ""))
            if not text:
                continue
            sentences.extend(split_sentences(text))
        if len(sentences) >= TARGET_B * 3:
            break
    print(f"    [benign] {len(sentences)} raw sentences")

    sentences = dedup(sentences)
    random.shuffle(sentences)
    return sentences[:TARGET_B]


# ——————— build hostile_new.txt ———————
def build_hostile():
    sentences = []
    for rec in hf_stream(SRC_AITA):
        verdict = (rec.get("verdict") or "").upper().strip()
        if verdict not in ("YTA", "ESH"):
            continue
        for field in ("comment1", "comment2"):
            text = clean(rec.get(field, ""))
            if not text:
                continue
            sentences.extend(split_sentences(text))
        if len(sentences) >= TARGET_H * 3:
            break
    print(f"    [hostile] {len(sentences)} raw sentences")

    sentences = dedup(sentences)
    random.shuffle(sentences)
    return sentences[:TARGET_H]


def write_corpus(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n")


# ——————— main ———————
def build_corpora_new():
    print("building benign corpus...")
    benign = build_benign()

    print("building hostile corpus...")
    hostile = build_hostile()

    print("writing...")

    write_corpus(benign, FILE_B)
    write_corpus(hostile, FILE_H)

    print(f"done. {len(benign)} benign, {len(hostile)} hostile")
