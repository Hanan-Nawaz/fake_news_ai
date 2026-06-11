import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizerFast


def get_bert_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizerFast.from_pretrained(model_name)


class NewsDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"   # ✅ returns proper fixed-size pytorch tensors directly
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         self.labels[idx]   # ✅ Trainer expects "labels" not "label"
        }


def create_bert_datasets(X_train, X_test, y_train, y_test, tokenizer, max_length=128):
    train_df = pd.DataFrame({"text": X_train, "label": y_train}).dropna()
    test_df  = pd.DataFrame({"text": X_test,  "label": y_test}).dropna()

    print("Tokenizing train..."); 
    train_dataset = NewsDataset(train_df["text"], train_df["label"], tokenizer, max_length)
    print("Tokenizing test...")
    test_dataset  = NewsDataset(test_df["text"],  test_df["label"],  tokenizer, max_length)

    return train_dataset, test_dataset