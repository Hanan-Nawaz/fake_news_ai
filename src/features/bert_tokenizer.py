import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast


def get_bert_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizerFast.from_pretrained(model_name)


def create_bert_datasets(X_train, X_test, y_train, y_test, tokenizer, max_length=512):
    train_df = pd.DataFrame({
        "text": X_train,
        "label": y_train
    }).dropna()

    test_df = pd.DataFrame({
        "text": X_test,
        "label": y_test
    }).dropna()

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, test_dataset