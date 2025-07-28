# Hate Speech Detection Model

This project builds a machine learning model to detect hate speech in tweets using text classification techniques. It includes data cleaning, class balancing, feature extraction, model training, and evaluation.

## Features

- Loads and cleans tweet data from CSV files
- Balances dataset using upsampling of minority class
- Converts text to TF-IDF features
- Trains a linear classifier (SGD)
- Evaluates model performance using F1 score

## Requirements

Install dependencies with:
```
pip install pandas scikit-learn
```

## Usage

1. Place your `train.csv` and `test.csv` files in the project directory.
2. Run the script:
    ```
    python hsd.py
    ```
3. The script will:
    - Load and clean the data
    - Balance the classes
    - Train and evaluate the model
    - Print the F1 score for validation

