import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
print("Loading email dataset from emails.csv...")
df = pd.read_csv('emails.csv')
print("First 5 rows of the dataset:")
print(df.head())
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Remove duplicate emails
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Download NLTK stopwords (do only once)
print("Downloading NLTK stopwords...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # ✅ Load stopwords here once

# Function to clean and tokenize text
def process(text):
    # Remove punctuation
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    clean = [word for word in nopunc.split() if word.lower() not in stop_words]
    return clean

# Preview processed text for first 5 emails
print("Preview of processed text for first 5 emails:")
print(df['text'].head().apply(process))

# Convert text data to feature vectors using CountVectorizer and custom analyzer
print("Transforming text data into feature vectors...")
message = CountVectorizer(analyzer=process).fit_transform(df['text'])

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(message, df['spam'], test_size=0.20, random_state=0)
print(f"Feature matrix shape: {message.shape}")

# Train the Multinomial Naive Bayes classifier
print("Training Multinomial Naive Bayes classifier...")
classifier = MultinomialNB().fit(xtrain, ytrain)

# Predict on training data and show results
train_pred = classifier.predict(xtrain)
print("\nClassification report for training data:")
print(classification_report(ytrain, train_pred))
print("Confusion Matrix (training):")
print(confusion_matrix(ytrain, train_pred))
print(f"Training Accuracy: {accuracy_score(ytrain, train_pred):.4f}")

# Predict on test data and show results
test_pred = classifier.predict(xtest)
print("\nClassification report for test data:")
print(classification_report(ytest, test_pred))
print("Confusion Matrix (test):")
print(confusion_matrix(ytest, test_pred))
print(f"Test Accuracy: {accuracy_score(ytest, test_pred):.4f}")
