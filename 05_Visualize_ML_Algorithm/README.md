# Visualize ML Algorithm - TV Advertising vs Sales

This project demonstrates how to use linear regression to analyze the relationship between TV advertising budgets and sales, and visualize the results using Plotly.

## Features

- Loads advertising data from a CSV file
- Trains a linear regression model to predict sales based on TV advertising spend
- Visualizes the data and regression line interactively

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- plotly

Install dependencies with:
```
pip install pandas numpy scikit-learn plotly
```

## Usage

1. Place your `Advertising.csv` file in the project directory.
2. Run the script:
    ```
    python vml.py
    ```
3. The script will:
    - Print the first 5 rows of the dataset
    - Train a linear regression model
    - Display an interactive scatter plot with the regression line
    - Print predicted sales values for a range of TV advertising budgets
