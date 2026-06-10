# author: yawen xue
# date: 06/01/2026 - 06/10/2026
# purpose: acquire corpora of benign & hostile texts

import os
import re
import json
import requests
import random

from pathlib import Path
from datasets import load_dataset

# file paths & target sizes for benign & hostile corpora
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FILE_B = DATA / "benign.txt"
FILE_H = DATA / "hostile.txt"

TARGET_B = 10000
TARGET_H = 10000

# sentence size specs
MIN_WORDS = 6
MAX_WORDS = 80

# sources for datasets!
SRC_YELP = "fancyzhx/yelp_polarity"
SRC_STACKEXCHANGE = "c17hawke/stackoverflow-dataset"
SRC_AITA = "OsamaBsher/AITA-Reddit-Dataset"
SRC_SARCASM = (
    "https://raw.githubusercontent.com/rishabhmisra/"
    "News-Headlines-Dataset-For-Sarcasm-Detection/master/"
    "Sarcasm_Headlines_Dataset.json"
)

# ——————— helper functions ———————
# helps stream from huggingface datasets
def hf_stream(src):
    try:
        ds = load_dataset(src, split="train", streaming=True)
        yield from ds
    except Exception as e:
        print(f"\t[hf]\terror loading {src}: {e}")

# helps fetch from json datasets
def fetch(url):
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    return r.text

# helps clean up text by removing special characters, html tags, etc.
def clean(text):
    text = re.sub(r"<pre><code>.*?</code></pre>", " ", text, flags=re.S)
    text = re.sub(r"<code>.*?</code>", " ", text, flags=re.S)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()

# checks sentence length
def length_ok(text):
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS

# splits paragraphs into sentences
def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for p in parts:
        p = p.strip()
        # remove too-short or too-long sentences
        if not length_ok(p):
            continue
        out.append(p)
    return out

# remove duplicates/frequently-appearing sentences
def dedup(sentences):
    seen = set()
    out = []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ——————— build benign.txt ———————
def build_benign():
    # first let's get raw sentences
    sentences = []
    sentences += from_yelp()
    sentences += from_stackexchange()

    # then filter & return
    sentences = dedup(sentences)
    random.shuffle(sentences)
    return sentences

# positive yelp reviews
def from_yelp():
    sentences = []
    for rec in hf_stream(SRC_YELP):
        # positive reviews are labelled 1, negative reviews labelled 0
        if rec["label"] == 1:
            # clean up text & split into sentences
            text = clean(rec["text"])
            sentences.extend(split_sentences(text))
        
        # stop loading sentences if we exceed 3 * target data size
        if len(sentences) >= TARGET_B * 3:
            break
    
    print(f"    [yelp]\t {len(sentences)} raw sentences")
    return sentences

# stack exchange
def from_stackexchange():
    sentences = []
    for rec in hf_stream(SRC_STACKEXCHANGE):
        # clean up text & split into sentences
        text = clean(rec["text"])
        sentences.extend(split_sentences(text))
        
        # stop loading sentences if we exceed 3 * target data size
        if len(sentences) >= TARGET_B * 3:
            break

    print(f"    [stackexchange]   {len(sentences)} raw sentences")
    return sentences

# ——————— build hostile.txt ———————
def build_hostile():
    # first let's get raw sentences
    sentences = []
    sentences += from_sarcastic_headlines()
    sentences += from_aita()

    # then filter & return
    sentences = dedup(sentences)
    random.shuffle(sentences)
    return sentences

# r/aita
def from_aita():
    sentences = []
    for rec in hf_stream(SRC_AITA):

        # get specifically "YTA" / "ESH" reviews as these tend to be bad-faith
        verdict = (rec.get("verdict") or "").upper().strip()
        if verdict in ("YTA", "ESH"):
            text = clean(
                (rec.get("title") or "") + " " +
                (rec.get("text") or "")
            )

            for s in split_sentences(text):
                    sentences.append(s)

        if len(sentences) >= TARGET_H * 3:
            break

    print(f"    [aita]\t {len(sentences)} raw sentences")
    return sentences

# sarcastic headlines
def from_sarcastic_headlines():
    sentences = []
    src = fetch(SRC_SARCASM)

    for line in src.splitlines():
        rec = json.loads(line)
        if rec.get("is_sarcastic") == 1:
            text = clean(rec.get("headline", ""))
            if length_ok(text):
                sentences.append(text)

        if len(sentences) >= TARGET_H * 3:
            break

    print(f"    [sarcasm]\t {len(sentences)} raw sentences")
    return sentences

# ——————— main function ———————
def build_corpora():
    # run above functions to get corpora
    print("building benign corpus...")
    corpus_b = build_benign()

    print("building hostile corpus...")
    corpus_h = build_hostile()

    # write to /data
    print("writing...")
    with open(FILE_B, "w", encoding="utf-8") as f:
        for s in corpus_b:
            f.write(s.strip() + "\n")
    with open(FILE_H, "w", encoding="utf-8") as f:
        for s in corpus_h:
            f.write(s.strip() + "\n")
    
    print(f"done. {len(corpus_b)} benign, {len(corpus_h)} hostile")
