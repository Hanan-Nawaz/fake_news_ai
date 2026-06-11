from transformers import AutoModel

def get_distilbert_model(model_name="distilbert-base-uncased", num_labels=3):
    return AutoModel.from_pretrained(model_name, num_labels=num_labels)