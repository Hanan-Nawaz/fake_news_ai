from scripts.train_classical import main as main_classical
from scripts.train_distilbert_embeddings import main as main_distilbert

def main():
    main_classical()
    main_distilbert()


if __name__ == "__main__":
    main()
