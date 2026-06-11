from transformers import AutoTokenizer

def get_distilbert_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")