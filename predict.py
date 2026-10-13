"""
Predict whether a news article is REAL or FAKE from the command line.

Usage:
    python predict.py "Your news headline or article text here"
    python predict.py            (then type the news when asked)
"""

import sys

import joblib

from train import MODEL_PATH, clean_text


def predict(text, saved=None):
    saved = saved or joblib.load(MODEL_PATH)
    model = saved["model"]
    label = model.predict([clean_text(text)])[0]
    return "FAKE" if label == 1 else "REAL"


if __name__ == "__main__":
    news = " ".join(sys.argv[1:]) or input("Enter news text: ")
    print("Prediction:", predict(news))
