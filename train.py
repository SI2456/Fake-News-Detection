"""
Fake News Detection - Training Script
-------------------------------------
Steps:
  1. Load dataset
  2. Clean text
  3. Convert text to numbers with TF-IDF
  4. Tune hyperparameters with GridSearchCV
  5. Train 5 ML models (incl. ensembles) and compare them with 5-fold cross-validation
  6. Save the best model + evaluation plots
"""

import os
import re
import string

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join("model", "fake_news_model.pkl")


# ---------------------------------------------------------------- 1. Load data
def load_data():
    """
    Uses the Kaggle 'Fake and Real News' dataset (Fake.csv + True.csv) if it is
    in the data/ folder, otherwise falls back to the small sample dataset.
    Returns a DataFrame with columns: text, label (0 = REAL, 1 = FAKE).
    """
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("Using Kaggle dataset (Fake.csv + True.csv)")
        fake = pd.read_csv(fake_path)
        real = pd.read_csv(true_path)
        fake["label"] = 1
        real["label"] = 0
        df = pd.concat([fake, real], ignore_index=True)
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        # True.csv articles start with "WASHINGTON (Reuters) -" which would let the
        # model cheat, so remove that source tag.
        df["text"] = df["text"].str.replace(r"^.*?\(Reuters\)\s*-\s*", "", regex=True)
        return df[["text", "label"]]

    print("Kaggle dataset not found -> using data/sample_news.csv")
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_news.csv"))
    df["label"] = df["label"].map({"REAL": 0, "FAKE": 1})
    return df[["text", "label"]]


# --------------------------------------------------------------- 2. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove links
    text = re.sub(r"<.*?>", " ", text)  # remove html tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # punctuation
    text = re.sub(r"\d+", " ", text)  # numbers
    text = re.sub(r"\s+", " ", text).strip()  # extra spaces
    return text


# ------------------------------------------------------------------- helpers
def make_tfidf():
    # unigrams + bigrams, ignore English stop-words and words found in >70% of articles
    return TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))


def plot_label_distribution(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["label"].map({0: "REAL", 1: "FAKE"}), hue=df["label"].map({0: "REAL", 1: "FAKE"}),
                  palette=["#2e8b57", "#c0392b"], legend=False)
    plt.title("Number of Real vs Fake News")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"))
    plt.close()


def plot_confusion_matrix(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()


def plot_model_comparison(results):
    plt.figure(figsize=(8, 4.5))
    names = list(results.keys())
    scores = [results[n] * 100 for n in names]
    bars = plt.bar(names, scores, color="#3b6fb6")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.1f}%", ha="center")
    plt.xticks(rotation=15)
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = load_data().dropna()
    df = df.drop_duplicates(subset="text")
    print(f"Total articles: {len(df)}  (REAL: {(df.label == 0).sum()}, FAKE: {(df.label == 1).sum()})")
    plot_label_distribution(df)

    df["text"] = df["text"].apply(clean_text)

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3. Hyperparameter tuning: find the best C for Logistic Regression with GridSearchCV
    grid = GridSearchCV(
        Pipeline([("tfidf", make_tfidf()), ("clf", LogisticRegression(max_iter=1000))]),
        param_grid={"clf__C": [0.1, 1, 10, 100]},
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    best_C = grid.best_params_["clf__C"]
    print(f"GridSearchCV best C for Logistic Regression: {best_C}\n")

    # 4. Models: 3 basic models + 2 ensemble models
    models = {
        "Logistic Regression": LogisticRegression(C=best_C, max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        # Voting Ensemble: combines the probabilities of 3 different models
        "Voting Ensemble": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=best_C, max_iter=1000)),
                ("nb", MultinomialNB()),
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ],
            voting="soft",
        ),
    }

    results = {}
    cv_results = {}
    trained = {}
    print(f"{'Model':<20} {'CV accuracy (5-fold)':<22} Test accuracy")
    for name, clf in models.items():
        pipeline = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        # 5-fold cross-validation on the training data
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        results[name] = acc
        cv_results[name] = cv_scores.mean()
        trained[name] = pipeline
        print(f"{name:<20} {cv_scores.mean() * 100:5.2f}% (+/- {cv_scores.std() * 100:.2f})    {acc * 100:.2f}%")

    # 5. Pick the best model by cross-validation accuracy (more reliable than one test split)
    best_name = max(cv_results, key=cv_results.get)
    best_model = trained[best_name]
    y_pred = best_model.predict(X_test)

    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"], zero_division=0))

    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_model_comparison(results)

    joblib.dump({"model": best_model, "name": best_name, "accuracy": results[best_name]}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
