![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Research-Machine%20Learning-orange)

# Machine Learning Approaches for Detecting AI-Generated Fake News

This repository contains the implementation and experimental results for the research paper **"Machine Learning Approaches for Detecting AI-Generated Fake News."** The project presents a comparative analysis of traditional machine learning algorithms and a transformer-based approach for detecting AI-generated fake news.

---

## Overview

The objective of this project is to classify news articles into three categories:

* Human-written Fake News
* Real News
* AI-generated Fake News

The following models are evaluated:

* Logistic Regression
* Naive Bayes
* Linear Support Vector Machine (SVM)
* Random Forest
* DistilBERT Embeddings + Logistic Regression

Traditional machine learning models use **TF-IDF** feature extraction, while the transformer-based approach uses **DistilBERT contextual embeddings**.

---

## Dataset

The dataset is based on:

> M. Ishraquzzaman, M. A. I. Chowdhury, S. Rahman, and R. Khan,
> **Ensemble Transformer-Based Detection of Fake and AI-Generated News**,
> *Applied Computational Intelligence and Soft Computing*, 2025.

The original dataset contains **155,121** news articles. To eliminate class imbalance, the dataset was balanced using random undersampling, resulting in **93,000** samples (31,000 per class).

| Label | Description             |
| ----: | ----------------------- |
|     0 | Human-written Fake News |
|     1 | Real News               |
|     2 | AI-generated Fake News  |

---

## Repository Structure

```text
fake_news_ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Hanan-Nawaz/fake_news_ai.git
cd fake_news_ai
```

### Install dependencies using uv

```bash
uv sync
```

or create a virtual environment manually

```bash
uv venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

uv sync
```

---

## Running the Project

Run the complete pipeline:

```bash
python main.py
```

---

## Models Evaluated

| Model                            | Feature Representation |
| -------------------------------- | ---------------------- |
| Logistic Regression              | TF-IDF                 |
| Naive Bayes                      | TF-IDF                 |
| Linear SVM                       | TF-IDF                 |
| Random Forest                    | TF-IDF                 |
| DistilBERT + Logistic Regression | DistilBERT Embeddings  |

---

## Experimental Results

| Model                                |   Accuracy |  Precision |     Recall |   F1-score |
| ------------------------------------ | ---------: | ---------: | ---------: | ---------: |
| **DistilBERT + Logistic Regression** | **87.25%** | **87.19%** | **87.25%** | **87.19%** |
| Linear SVM                           |     86.37% |     86.42% |     86.37% |     86.38% |
| Logistic Regression                  |     85.78% |     85.96% |     85.78% |     85.81% |
| Naive Bayes                          |     83.20% |     83.45% |     83.20% |     83.28% |
| Random Forest                        |     82.79% |     83.25% |     82.79% |     82.86% |

The transformer-based approach achieved the best overall performance, demonstrating the effectiveness of contextual embeddings for AI-generated fake news detection.

---

## Figures

The `figures/` directory contains:

* Workflow of the proposed methodology
* Dataset distribution before balancing
* Dataset distribution after balancing
* Accuracy comparison
* F1-score comparison

---

## Technologies

* Python
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Pandas
* NumPy
* Matplotlib
* uv

---

## Research Paper

**Machine Learning Approaches for Detecting AI-Generated Fake News**

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{nawaz2026fake_news_ai,
  author = {Abdul Hanan Nawaz},
  title = {Machine Learning Approaches for Detecting AI-Generated Fake News},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Hanan-Nawaz/fake_news_ai}
}
```

---

## License

This project is released under the MIT License.

---

## Author

**Abdul Hanan Nawaz**

Computer Science and Engineering

Frankfurt University of Applied Sciences

ORCID: https://orcid.org/0009-0004-5964-391
