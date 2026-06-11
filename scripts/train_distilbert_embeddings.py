import os
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_file
from src.data.preprocess_data import preprocess_pipeline
from src.data.split_data import split_data
from src.features.distilbert_tokenizer import get_distilbert_tokenizer
from src.models.distilbert_model import get_distilbert_model
from src.evaluation.evaluate import evaluate_model


def get_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
    embeddings = []
    texts = list(texts.astype(str))

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def main():
    raw_file_path = "data/processed/news_balanced.csv"
    results_path = "results/distilbert_embedding_results.csv"

    os.makedirs("results", exist_ok=True)

    df = load_file(raw_file_path, "cp1252")
    df = preprocess_pipeline(df)
    df["label"] = df["label"].astype(int)

    print("Label distribution:")
    print(df["label"].value_counts())
    print("Unique labels:", sorted(df["label"].unique()))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    X_train, X_test, y_train, y_test = split_data(df)

    model_name = "distilbert-base-uncased"

    print("Loading DistilBERT tokenizer and model...")
    tokenizer = get_distilbert_tokenizer(model_name)
    model = get_distilbert_model(model_name=model_name, num_labels=3)

    # For embeddings, we need the base transformer, not classification logits.
    if hasattr(model, "distilbert"):
        model = model.distilbert

    model = model.to(device)
    model.eval()

    print("Creating train embeddings...")
    X_train_emb = get_embeddings(
        texts=X_train,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=32,
        max_length=128
    )

    print("Creating test embeddings...")
    X_test_emb = get_embeddings(
        texts=X_test,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=32,
        max_length=128
    )

    print("Training Logistic Regression on DistilBERT embeddings...")
    classifier = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    classifier.fit(X_train_emb, y_train)
    y_pred = classifier.predict(X_test_emb)

    metrics = evaluate_model(
        model_name="DistilBERT Embeddings + Logistic Regression",
        y_true=y_test,
        y_pred=y_pred
    )

    results_df = pd.DataFrame([metrics])

    print("\nDistilBERT embedding results:")
    print(results_df)

    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()