An on-going project that builds an explainable multi-class classification system for research paper summaries using statistical and machine learning methods.

Hypothesis Testing on "If Summary length contributes to a research paper being categorised to a certain category"

Dataset
The dataset used: arXiv Scientific Dataset
Contains: 136K+ research papers, including titles, categories, and summaries.

What it does:
It analyzes missing values, category distribution, class imbalance and summary length distribution.
For exploratory analysis it uses category frequency charts, word count distribution, top keywords per category.
Text Preprocessing is done using TF-IDF Vectorization and other tasks such as lowercasing, punctuation removal, stopword removal, and lemmatization.

Models:
The system uses supervised learning algortihms like Multinomial Naive Bayes, Logistic Regression and Linear SVM. 

Evaluation metrics used: Accuracy, Precision, Recall, F1-score (macro/micro), Confusion Matrix.

A Web UI using Streamlit Ui for users to upload summaries of research papers and get the corresponding category.

How to Setup the project:
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
python research_paper_summary_classification.py
streamlit run app.py

Usage of the application:
1. The training script will:
   - Load and preprocess the arXiv dataset
   - Train three ML models on the top 10 categories
   - Generate statistical analysis and plots
   - Save trained models for the web app

2. The Streamlit app allows users to:
   - Input a research paper summary
   - Get predictions from all three models
   - View classification results in real-time


The system classifies into the top 10 most frequent categories:
- Machine Learning
- Computer Vision and Pattern Recognition
- Computation and Language (Natural Language Processing)
- Artificial Intelligence
- Machine Learning (Statistics)
- Neural and Evolutionary Computing
- Robotics
- Information Retrieval
- Methodology (Statistics)
- Computation and Language (Legacy category)
