# Fake News Detection using Machine Learning

An ML project that classifies news articles as **REAL** or **FAKE** using
**TF-IDF** text features, **ensemble learning** (Random Forest, Voting Ensemble),
**hyperparameter tuning** (GridSearchCV) and **5-fold cross-validation**.

## Project Structure

```
fake news/
├── data/
│   └── sample_news.csv     # small demo dataset (80 rows)
├── model/                  # saved model (created by train.py)
├── outputs/                # plots (created by train.py)
├── train.py                # train + evaluate + save model
├── predict.py              # predict from command line
├── app.py                  # Streamlit web app
└── requirements.txt
```

## How It Works

1. **Load data**: news text with labels REAL/FAKE
2. **Pre-processing**: lowercase, remove links, punctuation, numbers, extra spaces
3. **Feature extraction**: TF-IDF (Term Frequency × Inverse Document Frequency) with English
   stop-words removed and unigrams + bigrams
4. **Hyperparameter tuning**: GridSearchCV tries different values of `C` for Logistic Regression
   and keeps the best one
5. **Model training**: 5 classifiers (3 basic + 2 ensemble) trained on 80% of the data, each checked
   with 5-fold cross-validation
6. **Evaluation**: accuracy, precision, recall, F1-score and confusion matrix on the other 20%
7. **Deployment**: best model saved with `joblib` and used in a Streamlit web app

## How to Run

```bash
pip install -r requirements.txt
python train.py
python predict.py "Scientists confirm the earth is flat, NASA lied!"
python -m streamlit run app.py
```

## Using the Real Dataset (recommended for submission)

The sample dataset is tiny and only for demo. For real results, download the
**"Fake and Real News Dataset"** from Kaggle (Clément Bisaillon), which has ~44,000 articles:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Put `Fake.csv` and `True.csv` inside the `data/` folder and run `python train.py` again.
The script detects them automatically. Expect roughly **98–99% accuracy**.

## Algorithms Used

| Model | Idea |
|---|---|
| Logistic Regression | Linear model that outputs the probability of FAKE |
| Multinomial Naive Bayes | Probabilistic model based on word frequencies (Bayes' theorem) |
| Linear SVM | Finds the best separating boundary between REAL and FAKE |
| Random Forest *(ensemble – bagging)* | Many decision trees vote; reduces overfitting |
| Voting Ensemble *(ensemble)* | Averages the probabilities of Logistic Regression, Naive Bayes and Random Forest |

**Advanced techniques used:** ensemble learning, GridSearchCV hyperparameter tuning, 5-fold cross-validation.

## Limitations

- The model learns writing **style and word patterns**, not actual facts, so it cannot verify truth.
- It works best on news similar to the training data (mostly US political news in the Kaggle dataset).

## Future Scope

- Use deep learning models such as LSTM or BERT
- Add fact-checking using external sources
- Support multiple languages
