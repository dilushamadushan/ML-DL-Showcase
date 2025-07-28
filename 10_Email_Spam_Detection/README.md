# Email Spam Detection

This project demonstrates how to build a machine learning model to classify emails as spam or not spam using Python. It covers data cleaning, feature extraction, model training, and evaluation.

## Features

- Loads and cleans email data from a CSV file
- Removes duplicate emails and checks for missing values
- Cleans and tokenizes email text (removes punctuation and stopwords)
- Converts text to feature vectors using CountVectorizer
- Trains a Multinomial Naive Bayes classifier
- Evaluates model performance with accuracy, confusion matrix, and classification report

## Requirements

Install dependencies with:
```
pip install numpy pandas nltk scikit-learn
```
