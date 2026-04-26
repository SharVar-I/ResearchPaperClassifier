import os
import re
import string
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "arXiv_scientific dataset.csv"
TEXT_COL = "summary"
LABEL_COL = "category"
WORD_COUNT_COL = "summary_word_count"
TOP_N_CATEGORIES = 10
MIN_CATEGORY_COUNT = 500
MAX_SAMPLES_FOR_TRAINING = 40000
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


def create_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[LABEL_COL].value_counts().reset_index()
    counts.columns = [LABEL_COL, "count"]
    return counts


def plot_category_distribution(category_counts: pd.DataFrame, top_n: int = 20):
    plt.figure(figsize=(12, 8))
    top = category_counts.head(top_n)
    sns.barplot(y=LABEL_COL, x="count", data=top, palette="viridis")
    plt.title(f"Top {top_n} Categories by Paper Count")
    plt.xlabel("Document Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig("category_frequency_top.png")
    plt.close()


def plot_summary_length_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[WORD_COUNT_COL].dropna(), bins=40,
                 kde=False, color="steelblue")
    plt.title("Summary Length Distribution")
    plt.xlabel("Summary Word Count")
    plt.ylabel("Number of Documents")
    plt.tight_layout()
    plt.savefig("summary_length_distribution.png")
    plt.close()


def plot_summary_length_by_category(df: pd.DataFrame, top_categories: list):
    plt.figure(figsize=(14, 8))
    subset = df[df[LABEL_COL].isin(top_categories)]
    sns.boxplot(x=LABEL_COL, y=WORD_COUNT_COL, data=subset, palette="magma")
    plt.xticks(rotation=45, ha="right")
    plt.title("Summary Word Count by Top Categories")
    plt.xlabel("Category")
    plt.ylabel("Summary Word Count")
    plt.tight_layout()
    plt.savefig("summary_length_by_category.png")
    plt.close()


def get_top_keywords_by_category(df: pd.DataFrame, top_categories: list, n_keywords: int = 10):
    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 1), stop_words="english")
    texts = df[TEXT_COL].astype(str).tolist()
    y = df[LABEL_COL].astype(str).tolist()
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_keywords = {}
    for category in top_categories:
        category_mask = np.array([label == category for label in y])
        if category_mask.sum() == 0:
            continue
        category_mean = X[category_mask].mean(axis=0)
        top_indices = np.asarray(
            category_mean).ravel().argsort()[-n_keywords:][::-1]
        top_keywords[category] = feature_names[top_indices].tolist()
    return top_keywords


def analyze_confusions(results, top_categories):
    print("=== Confusion Analysis ===")
    for result in results:
        cm = result["confusion_matrix"]
        model_name = result["model"]
        print(f"Model: {model_name}")
        # Find off-diagonal max
        off_diag = cm - np.diag(np.diag(cm))
        if off_diag.size > 0:
            max_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
            cat1, cat2 = top_categories[max_idx[0]], top_categories[max_idx[1]]
            count = off_diag[max_idx]
            print(f"Most confused pair: {cat1} vs {cat2} ({count} times)")
        print()


def chi_square_term_specificity(df, top_categories, n_terms=20):
    print("=== Chi-Square Test for Category-Specific Terms ===")
    # Create contingency table for each term
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(df["clean_summary"])
    feature_names = vectorizer.get_feature_names_out()

    category_specific_terms = {}
    for category in top_categories:
        category_docs = df[df[LABEL_COL] == category]["clean_summary"]
        other_docs = df[df[LABEL_COL] != category]["clean_summary"]

        category_counts = Counter(" ".join(category_docs).split())
        other_counts = Counter(" ".join(other_docs).split())

        chi_scores = {}
        for term in feature_names:
            if term in category_counts or term in other_counts:
                # Contingency table: [in_category, not_in_category] x [term_present, term_absent]
                in_cat_with_term = category_counts.get(term, 0)
                in_cat_without_term = len(category_docs) - in_cat_with_term
                not_cat_with_term = other_counts.get(term, 0)
                not_cat_without_term = len(other_docs) - not_cat_with_term

                table = np.array([[in_cat_with_term, in_cat_without_term],
                                  [not_cat_with_term, not_cat_without_term]])
                if table.sum() > 0 and (table > 0).all():
                    chi2, p, _, _ = stats.chi2_contingency(table)
                    chi_scores[term] = chi2

        # Top terms for this category
        top_terms = sorted(chi_scores.items(),
                           key=lambda x: x[1], reverse=True)[:n_terms]
        category_specific_terms[category] = top_terms
        print(
            f"{category}: {', '.join([f'{term} ({score:.1f})' for term, score in top_terms])}")
    print()


def test_summary_length_hypothesis(df, top_categories):
    print("=== Hypothesis Test: Summary Length Influences Classification ===")
    # Perform ANOVA on summary lengths across categories
    lengths_by_category = [df[df[LABEL_COL] == cat]
                           [WORD_COUNT_COL].values for cat in top_categories]
    f_stat, p_value = stats.f_oneway(*lengths_by_category)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant differences in summary lengths across categories.")
    else:
        print("No significant differences in summary lengths across categories.")
    print()


def build_model_pipeline(model):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    min_df=10,
                    max_df=0.95,
                    ngram_range=(1, 2),
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            ("clf", model),
        ]
    )


def evaluate_model(model_name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f"=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)
    print()
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "pipeline": pipeline,
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Total rows: {len(df)}")

    print("Checking missing values...")
    missing = df.isna().sum()
    print(missing)

    print("Computing category distribution...")
    category_counts = create_category_summary(df)
    print(category_counts.head(25).to_string(index=False))

    print("Plotting category frequency and summary length distribution...")
    plot_category_distribution(category_counts, top_n=TOP_N_CATEGORIES)
    plot_summary_length_distribution(df)

    df[WORD_COUNT_COL] = pd.to_numeric(df[WORD_COUNT_COL], errors="coerce")
    df = df.dropna(subset=[WORD_COUNT_COL])

    top_categories = category_counts[category_counts["count"]
                                     >= MIN_CATEGORY_COUNT][LABEL_COL].tolist()
    if len(top_categories) > TOP_N_CATEGORIES:
        top_categories = top_categories[:TOP_N_CATEGORIES]
    print(f"Using top {len(top_categories)} categories for modeling")
    print(top_categories)

    plot_summary_length_by_category(df, top_categories)

    print("Preprocessing text... this may take a few minutes.")
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    df["clean_summary"] = df[TEXT_COL].astype(str).apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer))

    df = df[df[LABEL_COL].isin(top_categories)].copy()
    df = df[df["clean_summary"].str.strip().astype(bool)]
    if len(df) > MAX_SAMPLES_FOR_TRAINING:
        df = df.sample(n=MAX_SAMPLES_FOR_TRAINING, random_state=RANDOM_STATE)

    X = df["clean_summary"].tolist()
    y = df[LABEL_COL].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training and evaluating models...")
    results = []
    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                solver="saga",
                max_iter=400,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        ("Multinomial Naive Bayes", MultinomialNB()),
        ("Linear SVM", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
    ]

    for name, estimator in models:
        pipeline = build_model_pipeline(estimator)
        result = evaluate_model(name, pipeline, X_train,
                                X_test, y_train, y_test)
        results.append(result)

    # Save trained models
    os.makedirs("models", exist_ok=True)
    for result in results:
        model_name = result["model"].replace(" ", "_").lower()
        joblib.dump(result["pipeline"], f"models/{model_name}_pipeline.pkl")
    print("Saved trained models to models/ directory")

    analyze_confusions(results, top_categories)

    chi_square_term_specificity(df, top_categories)

    test_summary_length_hypothesis(df, top_categories)

    print("Top keywords for each top category:")
    top_keywords = get_top_keywords_by_category(
        df, top_categories, n_keywords=10)
    for category, keywords in top_keywords.items():
        print(f"{category}: {', '.join(keywords)}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("model_performance_summary.csv", index=False)
    print("Saved model performance summary to model_performance_summary.csv")


if __name__ == "__main__":
    main()
