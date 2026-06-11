import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer

from src.data.load_data import load_file
from src.data.preprocess_data import preprocess_pipeline
from src.data.split_data import split_data
from src.features.bert_tokenizer import get_bert_tokenizer, create_bert_datasets
from src.models.bert_model import get_bert_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    y_pred = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        y_pred,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(labels, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    raw_file_path = "data/processed/news_balanced.csv"
    results_path = "results/bert_model_results.csv"
    model_output_path = "models/bert_fake_news"

    os.makedirs("results", exist_ok=True)
    os.makedirs(model_output_path, exist_ok=True)

    df = load_file(raw_file_path, "cp1252")
    df = preprocess_pipeline(df)

    df["label"] = df["label"].astype(int)

    print("Label distribution:")
    print(df["label"].value_counts())
    print("Unique labels:", sorted(df["label"].unique()))

    X_train, X_test, y_train, y_test = split_data(df)

    tokenizer = get_bert_tokenizer("bert-base-uncased")

    train_dataset, test_dataset = create_bert_datasets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tokenizer=tokenizer,
        max_length=512
    )

    model = get_bert_model(
        model_name="bert-base-uncased",
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=model_output_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none",
        fp16=True,
        save_total_limit=1,
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    results = trainer.evaluate()

    results_df = pd.DataFrame([{
        "model": "BERT",
        "accuracy": results["eval_accuracy"],
        "precision_macro": results["eval_precision"],
        "recall_macro": results["eval_recall"],
        "f1_macro": results["eval_f1"]
    }])

    print("\nBERT results:")
    print(results_df)

    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    trainer.save_model(f"{model_output_path}/final")
    tokenizer.save_pretrained(f"{model_output_path}/final")
    print(f"Model saved to: {model_output_path}/final")


if __name__ == "__main__":
    main()