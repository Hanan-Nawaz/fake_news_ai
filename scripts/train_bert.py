import os
import sys
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# CONFIG

CONFIG = {
    "data_path":        "data/processed/news_balanced.csv",
    "encoding":         "cp1252",
    "text_column":      "text",          # change if your column is named differently
    "label_column":     "label",
    "model_name":       "bert-base-uncased",
    "max_length":       128,             # 512 → 211 h on CPU; 128 is fine for news
    "num_labels":       3,
    "test_size":        0.2,
    "random_state":     42,
    "batch_train":      16,
    "batch_eval":       32,
    "epochs":           3,
    "learning_rate":    2e-5,
    "weight_decay":     0.01,
    "logging_steps":    100,
    "warmup_ratio":     0.1,
    "results_path":     "results/bert_results.csv",
    "model_output":     "models/bert_fake_news",
}

# 1. LOAD DATA

def load_data(path: str, encoding: str) -> pd.DataFrame:
    print(f"\n[1/6] Loading data from: {path}")
    df = pd.read_csv(path, encoding=encoding)
    print(f"      Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


# 2. PREPROCESS

def preprocess(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    print("\n[2/6] Preprocessing...")

    df = df[[text_col, label_col]].copy()
    before = len(df)

    df.dropna(inplace=True)
    df[text_col]  = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)

    # Drop empty strings after strip
    df = df[df[text_col].str.len() > 0]

    print(f"      Rows after cleaning: {len(df):,}  (dropped {before - len(df):,})")
    print(f"      Label distribution:\n{df[label_col].value_counts().sort_index()}")
    print(f"      Unique labels: {sorted(df[label_col].unique())}")

    return df


# 3. SPLIT

def split_data(df: pd.DataFrame, text_col: str, label_col: str):
    print("\n[3/6] Splitting train / test...")

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )

    print(f"      Train: {len(X_train):,}   Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# 4. TOKENISE

def build_datasets(X_train, X_test, y_train, y_test, tokenizer):
    print("\n[4/6] Tokenising datasets (this may take a few minutes)...")

    def make_hf_dataset(texts, labels):
        df_tmp = pd.DataFrame({"text": texts, "label": labels})
        return Dataset.from_pandas(df_tmp, preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
        )

    train_ds = make_hf_dataset(X_train, y_train)
    test_ds  = make_hf_dataset(X_test,  y_test)

    # desc= gives you the Colab progress bar
    train_ds = train_ds.map(tokenize_fn, batched=True, desc="Tokenising train")
    test_ds  = test_ds.map(tokenize_fn,  batched=True, desc="Tokenising test")

    train_ds = train_ds.remove_columns(["text"])
    test_ds  = test_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    print(f"      Done. Train size: {len(train_ds):,}   Test size: {len(test_ds):,}")
    return train_ds, test_ds


# 5. MODEL + METRICS

def get_model():
    print(f"\n[5/6] Loading model: {CONFIG['model_name']}")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
    )
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# 6. TRAIN

def train(model, train_ds, test_ds):
    print("\n[6/6] Training BERT...")
    sys.stdout.flush()

    use_fp16 = torch.cuda.is_available()
    print(f"      CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU:  {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["model_output"], exist_ok=True)
    os.makedirs("results", exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CONFIG["model_output"],

        # Training schedule
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],

        # Batch sizes
        per_device_train_batch_size=CONFIG["batch_train"],
        per_device_eval_batch_size=CONFIG["batch_eval"],

        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,

        # Logging
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"],
        report_to="none",            # disable wandb / tensorboard

        # Performance
        fp16=use_fp16,               # half-precision on GPU
        dataloader_num_workers=2,    # parallel data loading
        dataloader_pin_memory=True,

        # Progress
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    print("\nTraining finished.")
    return trainer


# MAIN

def main():
    # ── Load ──────────────────────────────────
    df = load_data(CONFIG["data_path"], CONFIG["encoding"])

    # ── Preprocess ────────────────────────────
    df = preprocess(df, CONFIG["text_column"], CONFIG["label_column"])

    # ── Split ─────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(
        df, CONFIG["text_column"], CONFIG["label_column"]
    )

    # ── Tokenise ──────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])
    train_ds, test_ds = build_datasets(X_train, X_test, y_train, y_test, tokenizer)

    # ── Model ─────────────────────────────────
    model = get_model()

    # ── Train ─────────────────────────────────
    trainer = train(model, train_ds, test_ds)

    # ── Evaluate ──────────────────────────────
    print("\nEvaluating on test set...")
    results = trainer.evaluate()

    results_df = pd.DataFrame([{
        "model":           "BERT (bert-base-uncased)",
        "accuracy":        round(results["eval_accuracy"],  4),
        "precision_macro": round(results["eval_precision"], 4),
        "recall_macro":    round(results["eval_recall"],    4),
        "f1_macro":        round(results["eval_f1"],        4),
        "epochs":          CONFIG["epochs"],
        "max_length":      CONFIG["max_length"],
        "batch_size":      CONFIG["batch_train"],
    }])

    print("\n─── BERT Results ───────────────────────────")
    print(results_df.to_string(index=False))

    results_df.to_csv(CONFIG["results_path"], index=False)
    print(f"\nResults saved → {CONFIG['results_path']}")

    # ── Save model ────────────────────────────
    save_path = f"{CONFIG['model_output']}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved   → {save_path}")


if __name__ == "__main__":
    main()