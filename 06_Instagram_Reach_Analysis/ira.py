import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Load the Instagram dataset
print("Loading Instagram data from Instagram.csv...")
data = pd.read_csv("Instagram.csv", encoding='latin1')
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values in the dataset:")
print(data.isnull().sum())

# Remove rows with missing values
print("\nDropping rows with missing values...")
data = data.dropna()

# Display dataset info
print("\nDataset information after cleaning:")
print(data.info())

# Plot distribution of impressions from Home
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'], kde=True)
plt.show()

# Plot distribution of impressions from Hashtags
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'], kde=True)
plt.show()

# Plot distribution of impressions from Explore
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'], kde=True)
plt.show()

# Calculate total impressions from each source
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

# Pie chart of impressions from various sources
print("\nVisualizing total impressions from each source...")
fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# Generate word cloud for captions
print("\nGenerating word cloud for post captions...")
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Generate word cloud for hashtags
print("\nGenerating word cloud for hashtags...")
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Scatter plot: Likes vs Impressions
print("\nVisualizing relationship between Likes and Impressions...")
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title="Relationship Between Likes and Impressions")
figure.show()

# Scatter plot: Comments vs Impressions
print("\nVisualizing relationship between Comments and Impressions...")
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title="Relationship Between Comments and Total Impressions")
figure.show()

# Scatter plot: Shares vs Impressions
print("\nVisualizing relationship between Shares and Impressions...")
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title="Relationship Between Shares and Total Impressions")
figure.show()

# Scatter plot: Saves vs Impressions
print("\nVisualizing relationship between Saves and Impressions...")
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title="Relationship Between Post Saves and Total Impressions")
figure.show()

# Correlation analysis
print("\nCorrelation of all numeric features with Impressions:")
correlation = data.select_dtypes(include=np.number).corr()
print(correlation["Impressions"].sort_values(ascending=False))

# Calculate and print conversion rate from profile visits to follows
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(f"\nConversion rate from Profile Visits to Follows: {conversion_rate:.2f}%")

# Scatter plot: Profile Visits vs Follows
print("\nVisualizing relationship between Profile Visits and Follows...")
figure = px.scatter(data_frame=data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title="Relationship Between Profile Visits and Followers Gained")
figure.show()

# Prepare features and target for regression model
print("\nPreparing data for regression model...")
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Train Passive Aggressive Regressor
print("Training Passive Aggressive Regressor model...")
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(f"Model R^2 score on test set: {score:.4f}")

# Predict impressions for a sample post
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
predicted_impressions = model.predict(features)
print(f"\nPredicted Impressions for the sample post: {predicted_impressions[0]:.2f}")