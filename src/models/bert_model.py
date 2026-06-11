from transformers import AutoModelForSequenceClassification


def get_bert_model(model_name="distilbert-base-uncased", num_labels=3):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    