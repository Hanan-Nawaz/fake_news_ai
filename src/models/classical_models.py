from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def get_classical_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }