import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizerFast


def get_bert_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizerFast.from_pretrained(model_name)


class NewsDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_bert_datasets(X_train, X_test, y_train, y_test, tokenizer, max_length=128):
    train_df = pd.DataFrame({"text": X_train, "label": y_train}).dropna()
    test_df  = pd.DataFrame({"text": X_test,  "label": y_test}).dropna()

    train_dataset = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    test_dataset  = NewsDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, max_length)

    sample = train_dataset[0]
    print("Sample keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["labels"])

    return train_dataset, test_dataset