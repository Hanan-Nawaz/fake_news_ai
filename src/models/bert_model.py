from transformers import BertForSequenceClassification

def get_bert_model(model_name="bert-base-uncased", num_labels=3):  # was distilbert by mistake
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model