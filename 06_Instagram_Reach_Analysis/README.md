# Instagram Reach Analysis

This project analyzes Instagram post reach using data visualization and machine learning. It explores how different factors (likes, comments, shares, hashtags, etc.) affect the impressions of Instagram posts.

## Features

- Loads and cleans Instagram post data from a CSV file
- Visualizes distributions of impressions from various sources (Home, Hashtags, Explore, Other)
- Generates word clouds for post captions and hashtags
- Explores relationships between impressions and other metrics (likes, comments, shares, saves, profile visits, follows)
- Calculates correlation between numeric features and impressions
- Computes conversion rate from profile visits to follows
- Trains a Passive Aggressive Regressor to predict impressions based on post metrics

## Requirements

Install dependencies with:
```
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
```