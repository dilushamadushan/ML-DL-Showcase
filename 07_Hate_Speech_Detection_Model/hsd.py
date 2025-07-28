import pandas as pd
import re
from sklearn.utils import resample  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

# Load training and test datasets
print("Loading training data from train.csv...")
train = pd.read_csv('train.csv')
print(f"Training set columns: {train.columns.tolist()}")
print(f"Training set shape: {train.shape}, Total samples: {len(train)}")

print("\nLoading test data from test.csv...")
test = pd.read_csv('test.csv')
print(f"Test set columns: {test.columns.tolist()}")
print(f"Test set shape: {test.shape}, Total samples: {len(test)}")

# Function to clean tweet text (lowercase, remove mentions, links, special chars)
def clean_text(df, text_field):
    print(f"\nCleaning text field '{text_field}' in dataframe...")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(
        r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df

# Clean tweets in both train and test sets
test_clean = clean_text(test, 'tweet')
train_clean = clean_text(train, 'tweet')
print("Text cleaning complete.")

# Separate majority and minority classes for upsampling
train_majority = train_clean[train_clean.label == 0]
train_minority = train_clean[train_clean.label == 1]
print(f"\nMajority class samples: {len(train_majority)}, Minority class samples: {len(train_minority)}")

# Upsample minority class to balance the dataset
print("Upsampling minority class to balance the dataset...")
train_minority_upsampled = resample(
    train_minority, 
    replace=True,    
    n_samples=len(train_majority),   
    random_state=123
)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
print("Class distribution after upsampling:")
print(train_upsampled['label'].value_counts())

# Build a machine learning pipeline for hate speech detection
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),      # Convert text to token counts
    ('tfidf', TfidfTransformer()),    # Transform counts to TF-IDF features
    ('nb', SGDClassifier()),          # Linear classifier (Stochastic Gradient Descent)
])
print("\nPipeline created for text classification.")

# Split upsampled data into train and test sets for model evaluation
from sklearn.model_selection import train_test_split
print("Splitting data into training and validation sets...")
X_train, X_test, y_train, y_test = train_test_split(
    train_upsampled['tweet'],
    train_upsampled['label'],
    random_state=0
)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")

# Train the model
print("\nTraining the hate speech detection model...")
model = pipeline_sgd.fit(X_train, y_train)
print("Model training complete.")

# Predict on validation set
print("Predicting labels for validation set...")
y_predict = model.predict(X_test)

# Evaluate model performance using F1 score
from sklearn.metrics import f1_score
score = f1_score(y_test, y_predict)
print(f"\nF1 Score on validation set: {score:.4f}")