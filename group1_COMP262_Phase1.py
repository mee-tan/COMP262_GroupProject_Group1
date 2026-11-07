# COMP262 - Phase 1 Group Project - Group 1
# Members: Misha Mansoor, Syed Hussain, Syeda Farwah Fatema
# Date: 05/10/2026

"""
Dataset data exploration: List the main finding of the dataset. Be thorough and creative.
For example, look at:
a. Counts, averages
b. Distribution of the number of reviews across products
c. Distribution of the number of reviews per product
d. Distribution of reviews per user
e. Review lengths and outliers
f. Analyze lengths
g. Check for duplicates

2. Text basic pre-processing:
a. Label your data based on the value of “rating of the product” i.e. as follows:
i. Ratings 4,5: Positive
ii. Rating 3: Neutral
iii. Ratings 1,2: Negative
b. Chose the appropriate columns for your sentiment analyzer. (Give this some
thought) and mention in your report why you chose each column.
c. Check for outliers

3. Study the below three Lexicons packages and choose two for model building justify why
you chose them:
i. Valence Aware Dictionary and Sentiment Reasoner (VADR) you can find
out more information here: https://github.com/cjhutto/vaderSentiment
ii. TextBlob you can find out more information here:
https://textblob.readthedocs.io/en/dev/quickstart.html
iii. SENTIWORDNET you can find more information here:
http://nmis.isti.cnr.it/sebastiani/Publications/LREC10.pdf
"""
# ============================
#  IMPORTS
# ============================
import json
from pathlib import Path
import pandas as pd

# ===========
#  DATASET LOADING
# ===========

BASE_DIR = Path(__file__).resolve().parent

data = []
with open(BASE_DIR / 'AMAZON_FASHION.json', "r") as file:
    for line in file:
        if line.strip():  # Skip empty lines
            data.append(json.loads(line)) # append each JSON object to the data list

print("Dataset loaded successfully.....")

# Load the jsonl data list into a pandas DataFrame for easier analysis
df = pd.DataFrame(data)

print("DataFrame created successfully......")

# =========================================================
#  DATA EXPLORATION
# =========================================================
# Dataset data exploration: List the main finding of the dataset. Be thorough and creative.
# For example, look at:
# a. Counts, averages
# b. Distribution of the number of reviews across products
# c. Distribution of the number of reviews per product
# d. Distribution of reviews per user
# e. Review lengths and outliers
# f. Analyze lengths
# g. Check for duplicates 
# ===================

# Counts and averages 
print ("="*50 ,"\n")
print("DATA EXPLORATION")
print ("="*50 ,"\n")

print("\nDataFrame shape:", df.shape)
print("\nDataFrame columns:", df.columns.tolist())
print("\nTotal number of reviews:", len(df))
print("\nDataFrame info:", df.info())

print ("\nAverage rating:", df['overall'].mean())

# Distribution of the number of reviews across products
reviews_across_products = df.groupby('asin').size()
print("\nDistribution of the number of reviews across products:", reviews_across_products.describe())

# Distribution of the number of reviews per product
product_review_distribution = reviews_across_products.value_counts().sort_index()
print("\nDistribution of the number of reviews per product:", product_review_distribution)

# Distribution of reviews per user
reviews_per_user = df.groupby('reviewerID').size()
print("\nDistribution of reviews per user:", reviews_per_user.describe())

# Review lengths and outliers
df['review_length'] = df['reviewText'].fillna('').astype(str).apply(len)
print("\nReview lengths:", df['review_length'].describe())

# Outliers using IQR Method
# Interquartile Range (IQR): The difference between the 75th percentile (Q3) and the 25th percentile (Q1).
Q1 = df['review_length'].quantile(0.25) # 25th percentile
Q3 = df['review_length'].quantile(0.75) # 75th percentile
IQR = Q3 - Q1 # Interquartile Range

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

outliers = df[(df['review_length'] < (lower_bound)) | (df['review_length'] > (upper_bound))]
print("\nNumber of outliers in review lengths:", len(outliers))

# Analyze lengths
print("\nAverage review length:", df['review_length'].mean())
print("\nMinimum review length:", df['review_length'].min())
print("\nMaximum review length:", df['review_length'].max())

# Check for duplicates
#duplicate_rows = df.duplicated()
#print("Total duplicate rows:", duplicate_rows.sum())

duplicates = df[
    df.duplicated(
        subset=['reviewerID', 'asin'],
        keep=False
    )
]

print("\nTop 10 duplicate rows:", duplicates.head(10))

# =========================================================
#  TEXT BASIC PRE-PROCESSING
# =========================================================
# a. Label your data based on the value of “rating of the product” i.e. as follows:
# i.  Ratings 4,5: Positive
# ii. Rating 3:    Neutral
# iii.Ratings 1,2: Negative

# b. Chose the appropriate columns for your sentiment analyzer. (Give this some
# thought) and mention in your report why you chose each column.

# c. Check for outliers
# ===================
print ("="*50 ,"\n")
print("TEXT BASIC PRE-PROCESSING")
print ("="*50 ,"\n")

# Step a : Labeling the data 
def assign_sentiment(rating):
    if rating in [4, 5]:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    elif rating in [1, 2]:
        return 'Negative'

df['sentiment'] = df['overall'].apply(assign_sentiment)

print("\nTop 5 positive and negative ratings:","\n", 
      df[['overall', 'sentiment']].head()) 

print("\nSentiment counts:","\n", 
      df['sentiment'].value_counts())

# check the data 
print("\nTop 10 positive and negative ratings:","\n",df[['overall', 'sentiment']].head(10))

# Step b 
# Feature Selection 
# Input features
X = df[['reviewText', 'summary']]

y = df['sentiment'] # Target variable

# Shape
print("\nX shape:", X.shape)
print("\ny shape:", y.shape)

# Step C 
# Outliers

def check_outliers(df):
    Q1 = df['review_length'].quantile(0.25) # 25th percentile
    Q3 = df['review_length'].quantile(0.75) # 75th percentile

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df['review_length'] < lower_bound) |
        (df['review_length'] > upper_bound)
    ]

    return outliers

outliers = check_outliers(df)
print("\nLower bound:", lower_bound)
print("\nUpper bound:", upper_bound)


# =========================================================
# STEP 4: TEXT PREPROCESSING FOR VADER AND TEXTBLOB
# =========================================================

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Handle missing review text and summaries
df['reviewText'] = df['reviewText'].fillna('')
df['summary'] = df['summary'].fillna('')

# Combine review summary and full review text
df['text'] = df['summary'] + ' ' + df['reviewText']

# Remove extra whitespace
df['text'] = df['text'].apply(
    lambda x: re.sub(r'\s+', ' ', x).strip()
)

# VADER preprocessing: Preserve original capitalization and punctuation
df['vader_text'] = df['text']

# TextBlob preprocessing: Convert text to lowercase
df['textblob_text'] = df['text'].str.lower()

# Remove reviews with empty text
df = df[df['text'].str.strip() != ''].copy()

print("\nSTEP 4: TEXT PREPROCESSING")
print("Total reviews after preprocessing:", len(df))
print(df[['vader_text', 'textblob_text']].head())


# =========================================================
# STEP 5: RANDOMLY SELECT 1000 REVIEWS
# =========================================================

# Randomly select 1000 reviews using a fixed random seed
sample_df = df.sample(n=1000, random_state=42).copy()

# Reset the index
sample_df.reset_index(drop=True, inplace=True)

print("\nSTEP 5: RANDOM SAMPLING")
print("Total sampled reviews:", len(sample_df))

print("\nSentiment distribution in sample:")
print(sample_df['sentiment'].value_counts())

# Save the selected reviews
sample_df.to_csv("sampled_1000_reviews.csv", index=False)

print("\n1000 reviews saved successfully!")


# =========================================================
# STEP 6: SENTIMENT ANALYSIS USING TWO LEXICONS
# =========================================================

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------
# MODEL 1: VADER
# ---------------------------------------------------------

# Calculate VADER compound sentiment scores
sample_df['VADER_Score'] = sample_df['vader_text'].apply(
    lambda x: vader_analyzer.polarity_scores(x)['compound']
)

# Classify VADER predictions
sample_df['VADER_Prediction'] = sample_df['VADER_Score'].apply(
    lambda score: 'Positive' if score >= 0.05
    else 'Negative' if score <= -0.05
    else 'Neutral'
)


# ---------------------------------------------------------
# MODEL 2: TEXTBLOB
# ---------------------------------------------------------

# Calculate TextBlob polarity scores
sample_df['TextBlob_Score'] = sample_df['textblob_text'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

# Classify TextBlob predictions
sample_df['TextBlob_Prediction'] = sample_df['TextBlob_Score'].apply(
    lambda score: 'Positive' if score > 0
    else 'Negative' if score < 0
    else 'Neutral'
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\nSTEP 6: SENTIMENT MODELLING")

print("\nVADER Prediction Distribution:")
print(sample_df['VADER_Prediction'].value_counts())

print("\nTextBlob Prediction Distribution:")
print(sample_df['TextBlob_Prediction'].value_counts())

print("\nSample Model Predictions:")
print(sample_df[[
    'sentiment',
    'VADER_Score',
    'VADER_Prediction',
    'TextBlob_Score',
    'TextBlob_Prediction'
]].head(10))


# Save predictions for Step 7 validation
sample_df.to_csv("sentiment_predictions.csv", index=False)

print("\nBoth models completed successfully!")
print("Results saved to sentiment_predictions.csv")


# =========================================================
# STEP 7: VALIDATE AND COMPARE BOTH MODELS
# =========================================================

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

labels = ['Negative', 'Neutral', 'Positive']
results = []

for model in ['VADER', 'TextBlob']:
    y_true = sample_df['sentiment']
    y_pred = sample_df[f'{model}_Prediction']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    print(f"\n{model} Evaluation Results")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels, columns=labels
    ))

    results.append({
        'Model': model,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    })

# Model comparison table
comparison = pd.DataFrame(results)

print("\nSTEP 7: MODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv("step7_comparison.csv", index=False)

print("\nComparison saved to step7_comparison.csv")
