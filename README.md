# Sentiment Analysis on Kindle Book Reviews

🚀 Project Overview

This project aims to analyze Kindle book reviews and classify them into positive, neutral, or negative sentiments using a variety of Natural Language Processing (NLP) techniques. We employ methods such as tokenization, stopword removal, stemming, lemmatization, and advanced models like TF-IDF, Word2Vec, and LSTM to accurately predict sentiment from user reviews.

📋 Table of Contents

Overview
Getting Started
Installation
Data Collection
Data Preprocessing
NLP Techniques
Modeling
Evaluation

🔧 Getting Started

Requirements
Python 3.12.2 
Libraries:
nltk, spaCy, gensim, transformers for NLP
pandas, scikit-learn, tensorflow for machine learning
flask, joblib for deployment

🛠️ Data Collection

The data for this project was collected from Amazon Kindle Reviews Dataset (available on Kaggle).

🧹 Data Preprocessing

The data undergoes several preprocessing steps to clean and prepare it for modeling:

Handle Missing Data: Remove reviews with missing or incomplete text.
Label Sentiment: We categorize reviews based on ratings into:
Positive (4-5 stars)
Neutral (3 stars)
Negative (1-2 stars)
Text Cleaning: Standardize the text by converting it to lowercase and removing unwanted characters (e.g., special symbols, numbers).
Tokenization: Split the text into individual words.
Stopword Removal: Filter out common words like “the”, “is”, etc.
Stemming & Lemmatization: Reduce words to their root forms (e.g., “running” → “run”).

🔍 NLP Techniques

We apply the following NLP techniques for feature extraction:

Bag of Words (BoW): Convert the text into a set of features based on word frequency.
TF-IDF (Term Frequency - Inverse Document Frequency): Assign weights to words based on their importance across documents.
Word2Vec: Use this to capture semantic relationships between words by training a model on the text.
Vader Sentiment Analysis: Provides sentiment scores using a pre-built lexicon of words with associated sentiment scores.

📈 Modeling

Traditional Machine Learning Models:
Logistic Regression: A basic linear classifier.
Random Forest: An ensemble method that combines multiple decision trees for better performance.

📊 Evaluation

After training, we evaluate the performance of the model using:

Accuracy: The percentage of correctly predicted labels.
Precision & Recall: For evaluating positive predictions and capturing true positives.
F1-Score: A balanced measure between precision and recall.
Tip: You can evaluate multiple models and compare the results using a confusion matrix for better insights into performance.
