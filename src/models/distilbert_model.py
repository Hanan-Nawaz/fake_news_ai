from transformers import AutoModel

def get_distilbert_model():
    return AutoModel.from_pretrained("distilbert-base-uncased")