# author: yawen xue
# date: 05/29/2026 - 06/10/2026
# purpose: retrieve-delete-generate implementation attempt following li et al., 2018

# attempt 1 (commented out)
# benign corpora: positive yelp reviews, stackexchange posts
# negative corpora: yta/esh posts on r/aita, sarcasm headlines

# attempt 2
# benign corpora: nta/nah comments on r/aita
# negative corpora: yta/esh comments on r/aita

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from corpora import build_corpora
from corpora_new import build_corpora_new
from pathlib import Path

# config
N_MARKERS = 50
SALIENCE_THRESHOLD = 0.02

# paths
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# FILE_B = DATA / "benign.txt"
# FILE_H = DATA / "hostile.txt"

FILE_B = DATA / "benign_new.txt"
FILE_H = DATA / "hostile_new.txt"

# first let's build the corpora
# build_corpora()
build_corpora_new()

# ——————— get attribute markers ———————
benign_lines  = open(FILE_B).read().splitlines()
hostile_lines = open(FILE_H).read().splitlines()

# load corpora
with open(FILE_B, encoding="utf-8") as f:
    benign_lines = f.read().splitlines()
with open(FILE_H, encoding="utf-8") as f:
    hostile_lines = f.read().splitlines()

# fit TF-IDF model on combined corpus
vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    stop_words="english",
    min_df=5,
)

vec.fit(benign_lines + hostile_lines)

# mean TF-IDF vector for each corpus
benign_tfidf = vec.transform(benign_lines).mean(axis=0)
hostile_tfidf = vec.transform(hostile_lines).mean(axis=0)

# salience score:
# positive => more benign
# negative => more hostile
scores = np.asarray(benign_tfidf - hostile_tfidf).flatten()
feature_names = vec.get_feature_names_out()

print(f"max score:  {scores.max():.4f}")
print(f"min score:  {scores.min():.4f}")
print(f"mean score: {scores.mean():.4f}")

# top-N markers
sorted_scores = sorted(
    zip(feature_names, scores),
    key=lambda x: x[1]
)

hostile_markers = dict(sorted_scores[:N_MARKERS])
benign_markers = dict(sorted_scores[-N_MARKERS:])

print("\n=== HOSTILE MARKERS ===")
for phrase, score in hostile_markers.items():
    print(f"{score:.4f}\t{phrase}")

print("\n=== BENIGN MARKERS ===")
for phrase, score in reversed(list(benign_markers.items())):
    print(f"{score:.4f}\t{phrase}")
