import os
import pandas as pd
from src.data.load_data import load_file
from src.data.preprocess_data import preprocess_pipeline
from src.data.split_data import split_data
from src.features.tfidf import vectorize_text
from src.models.classical_models import get_classical_models
from src.evaluation.evaluate import evaluate_model

def main():
    raw_file_path = "data/processed/news_balanced.csv"
    results_path = "results/classical_model_results.csv"

    os.makedirs("results", exist_ok=True)

    df = load_file(raw_file_path, "cp1252")
    df = preprocess_pipeline(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    models = get_classical_models()
    results = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        metrics = evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    print("\nModel comparison:")
    print(results_df)

    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()