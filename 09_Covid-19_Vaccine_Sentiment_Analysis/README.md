# Covid-19 Vaccine Sentiment Analysis

This project analyzes public sentiment about Covid-19 vaccines using tweets. It applies text cleaning, sentiment scoring, feature engineering, and visualization to explore how people feel about vaccination.

## Features

- Loads and cleans tweet data from a CSV file
- Removes Twitter handles, hashtags, URLs, special characters, and single characters
- Calculates sentiment scores (positive, neutral, negative) using VADER
- Visualizes sentiment distributions and cumulative distributions
- Extracts and visualizes the most positive and negative tweets
- Generates word clouds for common and top words in positive and negative tweets

## Requirements

Install dependencies with:
```
pip install numpy pandas matplotlib seaborn nltk plotly wordcloud statsmodels
```
