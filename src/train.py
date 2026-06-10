# author: yawen xue
# date: 05/29/2026 - 06/10/2026
# purpose: training the model on llm-generated sentence pairs

# had claude help with structure and arguments


import json
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

CONFIG = {
    "model": "t5-small",
    "data": "data/processed",
    "out": "checkpoints/style_transfer",
    "epochs": 10,
    "bs": 8,
    "lr": 5e-4,
    "max_src": 128,
    "max_tgt": 128,
    "prefix": "rewrite to style B: ",
}

# helpers for data
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(data_dir):
    splits = {}
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        splits[split] = Dataset.from_list(load_jsonl(str(path)))
    return DatasetDict(splits)

# tokenization
def make_tokenise_fn(tokenizer, prefix, max_src, max_tgt):
    def tokenise(batch):
        inputs = [prefix + s for s in batch["src"]]

        model_inputs = tokenizer(
            inputs,
            max_length=max_src,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=batch["tgt"],
            max_length=max_tgt,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenise

# metrics
def make_compute_metrics(tokenizer):
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        decoded_labels = tokenizer.batch_decode(
            np.where(labels != -100, labels, tokenizer.pad_token_id),
            skip_special_tokens=True,
        )

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = bleu.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {"bleu": round(result["score"], 2)}

    return compute_metrics

# main function!
def train():
    cfg = CONFIG

    # load model, tokenizer, dataset
    print("loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"])

    print("loading dataset...")
    raw = build_dataset(cfg["data"])
    print(raw)

    tokenise_fn = make_tokenise_fn(
        tokenizer,
        cfg["prefix"],
        cfg["max_src"],
        cfg["max_tgt"],
    )

    tokenised = raw.map(
        tokenise_fn,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="tokenizng",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["out"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["bs"],
        per_device_eval_batch_size=cfg["bs"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.06,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=cfg["max_tgt"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["val"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # train & save
    print("\ntraining...")
    trainer.train()

    final_dir = Path(cfg["out"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\ndone! model saved to {final_dir}")

    # evaluate
    print("\nevaluating on test set...")
    results = trainer.evaluate(eval_dataset=tokenised["test"])
    print(f"BLEU: {results.get('eval_bleu', 'n/a')}")

if __name__ == "__main__":
    train()
