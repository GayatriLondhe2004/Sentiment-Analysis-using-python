"""
TASK 6: Sentiment Analysis (AI + Data Science)
===============================================
This project:
  1. Collects/loads text data (reviews, social media)
  2. Classifies sentiment: Positive / Negative / Neutral
  3. Uses VADER (rule-based) + TextBlob + trains an ML model
  4. Interprets public opinion trends with visualizations

Requirements:
    pip install nltk textblob pandas matplotlib seaborn scikit-learn

Run:
    python task6_sentiment.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# ── Download NLTK data (first time only) ──
import nltk
nltk.download("vader_lexicon",  quiet=True)
nltk.download("punkt",          quiet=True)
nltk.download("stopwords",      quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# STEP 1: Collect / Load Data
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  📥  STEP 1: Loading Text Data")
print("="*55)

# Sample dataset (reviews + social media style texts)
# In a real project, replace this with your CSV:
#   df = pd.read_csv("reviews.csv")
SAMPLE_DATA = [
    ("I absolutely love this product! It's fantastic and works perfectly.",   "positive"),
    ("This is the best purchase I have ever made. Highly recommend!",         "positive"),
    ("Amazing quality and fast delivery. Very happy with my order.",          "positive"),
    ("Great value for money. The product exceeded my expectations.",          "positive"),
    ("Wonderful experience! Customer service was really helpful.",            "positive"),
    ("Pretty good product, nothing extraordinary but gets the job done.",     "neutral"),
    ("Average product. It works as described, nothing more nothing less.",    "neutral"),
    ("Decent quality but the price is a bit high for what you get.",          "neutral"),
    ("Not bad, not great. Just an okay experience overall.",                  "neutral"),
    ("The product is fine. Delivery was on time. Nothing special.",           "neutral"),
    ("This is terrible. Complete waste of money. Very disappointed.",         "negative"),
    ("Worst product I've ever bought. Broke after 2 days. Avoid!",           "negative"),
    ("Absolutely horrible quality. The product looks nothing like the photo.","negative"),
    ("Very poor customer service. My issue was never resolved.",              "negative"),
    ("Do not buy this! It stopped working after a week. Total scam.",         "negative"),
    ("I'm really disappointed with this purchase.",                           "negative"),
    ("Super happy with this! Will definitely buy again!",                     "positive"),
    ("Meh, it's okay I guess. Not worth the hype.",                           "neutral"),
    ("Loved it! My kids enjoy playing with it every day.",                    "positive"),
    ("Returned it immediately. Not what I expected at all.",                  "negative"),
]

df = pd.DataFrame(SAMPLE_DATA, columns=["text", "label"])
print(f"Dataset loaded: {len(df)} samples")
print(df["label"].value_counts())

# ─────────────────────────────────────────
# STEP 2: Text Preprocessing
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  🧹  STEP 2: Text Preprocessing")
print("="*55)

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()       # normalize whitespace
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

df["cleaned"] = df["text"].apply(clean_text)
print("Sample cleaned text:")
print(df[["text", "cleaned"]].head(3).to_string(index=False))

# ─────────────────────────────────────────
# STEP 3A: VADER Sentiment (Rule-Based)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  🎯  STEP 3A: VADER Sentiment Analysis")
print("="*55)

sia = SentimentIntensityAnalyzer()

def vader_label(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_sentiment"] = df["text"].apply(vader_label)
df["vader_compound"]  = df["text"].apply(lambda t: sia.polarity_scores(t)["compound"])

# Accuracy
vader_acc = (df["vader_sentiment"] == df["label"]).mean()
print(f"\nVADER Accuracy: {vader_acc:.2%}")
print("\nVADER Predictions vs True Labels:")
print(df[["text", "label", "vader_sentiment", "vader_compound"]].to_string(index=False))

# ─────────────────────────────────────────
# STEP 3B: TextBlob Sentiment
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  💬  STEP 3B: TextBlob Sentiment")
print("="*55)

def textblob_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"

df["textblob_sentiment"] = df["text"].apply(textblob_label)
tb_acc = (df["textblob_sentiment"] == df["label"]).mean()
print(f"TextBlob Accuracy: {tb_acc:.2%}")

# ─────────────────────────────────────────
# STEP 3C: ML Model (TF-IDF + Logistic Regression)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  🤖  STEP 3C: ML Model — Logistic Regression")
print("="*55)

X = df["cleaned"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_train_v = tfidf.fit_transform(X_train)
X_test_v  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_v, y_train)

y_pred = model.predict(X_test_v)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

df["ml_sentiment"] = model.predict(tfidf.transform(df["cleaned"]))

# ─────────────────────────────────────────
# STEP 4: Visualizations & Trend Analysis
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  📊  STEP 4: Generating Visualizations")
print("="*55)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sentiment Analysis Dashboard", fontsize=16, fontweight="bold")

colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}

# 1. True label distribution
label_counts = df["label"].value_counts()
axes[0,0].pie(label_counts, labels=label_counts.index,
              colors=[colors[l] for l in label_counts.index],
              autopct="%1.1f%%", startangle=90)
axes[0,0].set_title("1. True Sentiment Distribution")

# 2. VADER compound score distribution
for lbl, grp in df.groupby("label"):
    axes[0,1].hist(grp["vader_compound"], alpha=0.6, label=lbl, color=colors[lbl], bins=8)
axes[0,1].set_title("2. VADER Compound Score by Label")
axes[0,1].set_xlabel("Compound Score")
axes[0,1].legend()

# 3. Model comparison accuracy
methods = ["VADER", "TextBlob", "ML (LR)"]
accuracies = [
    (df["vader_sentiment"]    == df["label"]).mean(),
    (df["textblob_sentiment"] == df["label"]).mean(),
    (df["ml_sentiment"]       == df["label"]).mean(),
]
bars = axes[1,0].bar(methods, accuracies, color=["#3498db","#9b59b6","#2ecc71"], width=0.5)
axes[1,0].set_title("3. Model Accuracy Comparison")
axes[1,0].set_ylabel("Accuracy")
axes[1,0].set_ylim(0, 1.1)
for bar, acc in zip(bars, accuracies):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{acc:.0%}", ha="center", fontweight="bold")

# 4. Confusion Matrix (VADER)
from sklearn.metrics import ConfusionMatrixDisplay
labels_order = ["positive", "neutral", "negative"]
cm = confusion_matrix(df["label"], df["vader_sentiment"], labels=labels_order)
disp = ConfusionMatrixDisplay(cm, display_labels=labels_order)
disp.plot(ax=axes[1,1], colorbar=False, cmap="Blues")
axes[1,1].set_title("4. VADER Confusion Matrix")

plt.tight_layout()
plt.savefig("task6_sentiment_report.png", dpi=150, bbox_inches="tight")
print("✅ Visualization saved: task6_sentiment_report.png")
plt.show()

# ─────────────────────────────────────────
# STEP 5: Predict New Reviews (Inference)
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  🔍  STEP 5: Predict New Text")
print("="*55)

def predict_sentiment(text):
    """Predict sentiment using all 3 methods."""
    vader   = vader_label(text)
    tb      = textblob_label(text)
    cleaned = clean_text(text)
    ml_pred = model.predict(tfidf.transform([cleaned]))[0]
    return {
        "text"     : text,
        "VADER"    : vader,
        "TextBlob" : tb,
        "ML_Model" : ml_pred,
    }

new_reviews = [
    "This product changed my life! I couldn't be happier.",
    "It was okay, nothing special, pretty average.",
    "I regret buying this. Absolutely terrible experience.",
]

print("\nNew Review Predictions:")
for rev in new_reviews:
    result = predict_sentiment(rev)
    print(f"\nText: {result['text']}")
    print(f"  VADER    → {result['VADER']}")
    print(f"  TextBlob → {result['TextBlob']}")
    print(f"  ML Model → {result['ML_Model']}")

print("\n✅ Task 6 complete!")
