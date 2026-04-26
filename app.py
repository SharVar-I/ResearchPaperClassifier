import os
import re
import string
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not present
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# Constants
TOP_CATEGORIES = [
    'Machine Learning',
    'Computer Vision and Pattern Recognition',
    'Computation and Language (Natural Language Processing)',
    'Artificial Intelligence',
    'Machine Learning (Statistics)',
    'Neural and Evolutionary Computing',
    'Robotics',
    'Information Retrieval',
    'Methodology (Statistics)',
    'Computation and Language (Legacy category)'
]

MODEL_NAMES = [
    "logistic_regression",
    "multinomial_naive_bayes",
    "linear_svm"
]

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

@st.cache_resource
def load_models():
    models = {}
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    for model_name in MODEL_NAMES:
        try:
            models[model_name] = joblib.load(f"models/{model_name}_pipeline.pkl")
        except FileNotFoundError:
            st.error(f"Model {model_name} not found. Please run the training script first.")
            return None
    return models, stop_words, lemmatizer

def main():
    st.title("Research Paper Summary Classification")
    st.markdown("""
    This app classifies research paper summaries into categories using machine learning models.
    Enter a paper summary below to get predictions from Logistic Regression, Naive Bayes, and SVM models.
    """)

    models_data = load_models()
    if models_data is None:
        return
    models, stop_words, lemmatizer = models_data

    # Input
    summary = st.text_area("Enter Paper Summary", height=200, placeholder="Paste or type the paper summary here...")

    if st.button("Classify"):
        if not summary.strip():
            st.error("Please enter a summary.")
            return

        # Preprocess
        clean_summary = preprocess_text(summary, stop_words, lemmatizer)

        if not clean_summary.strip():
            st.error("Summary is empty after preprocessing.")
            return

        # Predictions
        predictions = {}
        for model_name, pipeline in models.items():
            try:
                pred = pipeline.predict([clean_summary])[0]
                predictions[model_name] = pred
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
                return

        # Display results
        st.subheader("Classification Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Logistic Regression", predictions["logistic_regression"])

        with col2:
            st.metric("Naive Bayes", predictions["multinomial_naive_bayes"])

        with col3:
            st.metric("SVM", predictions["linear_svm"])

        # Summary
        st.subheader("Summary")
        st.write(f"**Preprocessed Text Length:** {len(clean_summary.split())} words")
        st.write("**Predictions:**")
        for model, pred in predictions.items():
            st.write(f"- {model.replace('_', ' ').title()}: {pred}")

        # Note about categories
        st.info(f"The models classify into the top {len(TOP_CATEGORIES)} categories from the arXiv dataset.")

if __name__ == "__main__":
    main()
