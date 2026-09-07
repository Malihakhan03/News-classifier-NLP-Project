"""
Script to create the exploratory_analysis.ipynb Jupyter Notebook
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), 'notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
NOTEBOOK_FILE = os.path.join(NOTEBOOKS_DIR, 'exploratory_analysis.ipynb')

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# News Headline Classification - Exploratory Data Analysis & NLP Pipeline\n",
            "\n",
            "**Academic NLP Project**  \n",
            "This notebook walks through the end-to-end Natural Language Processing (NLP) and Machine Learning workflow for classifying news headlines into categories:\n",
            "- **Technology**\n",
            "- **Business**\n",
            "- **Sports**\n",
            "- **Entertainment**\n",
            "- **Politics**\n",
            "- **Health**\n",
            "- **World**\n",
            "\n",
            "### Pipeline Workflow:\n",
            "1. **Exploratory Data Analysis (EDA)**\n",
            "2. **NLP Text Preprocessing** (Lowercasing, RegEx cleaning, Tokenization, Stopword removal, Lemmatization)\n",
            "3. **TF-IDF Feature Extraction** (N-grams & sublinear TF)\n",
            "4. **Multi-Model Training & Benchmarking** (Multinomial Naive Bayes, Logistic Regression, Linear SVM)\n",
            "5. **Evaluation & Confusion Matrix Analysis**\n",
            "6. **Inference on Unseen Test Headlines**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import essential libraries\n",
            "import os\n",
            "import re\n",
            "import string\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# NLP & ML Libraries\n",
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "from nltk.stem import WordNetLemmatizer\n",
            "from nltk.tokenize import word_tokenize\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.naive_bayes import MultinomialNB\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.svm import LinearSVC\n",
            "from sklearn.calibration import CalibratedClassifierCV\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
            "\n",
            "# Download NLTK datasets\n",
            "nltk.download('stopwords', quiet=True)\n",
            "nltk.download('punkt', quiet=True)\n",
            "nltk.download('punkt_tab', quiet=True)\n",
            "nltk.download('wordnet', quiet=True)\n",
            "nltk.download('omw-1.4', quiet=True)\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "print(\"Libraries loaded successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load and Inspect Dataset"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = os.path.join('..', 'data', 'news.csv')\n",
            "if not os.path.exists(data_path):\n",
            "    data_path = os.path.join('data', 'news.csv')\n",
            "\n",
            "df = pd.read_csv(data_path)\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "print(f\"Columns: {list(df.columns)}\")\n",
            "df.head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Exploratory Data Analysis (EDA)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Class Distribution\n",
            "plt.figure(figsize=(9, 4.5), dpi=150)\n",
            "category_counts = df['category'].value_counts()\n",
            "sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')\n",
            "plt.title('Distribution of News Categories', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Category', fontsize=11)\n",
            "plt.ylabel('Headline Count', fontsize=11)\n",
            "plt.xticks(rotation=20)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Headline length & word count distributions\n",
            "df['char_length'] = df['headline'].apply(len)\n",
            "df['word_count'] = df['headline'].apply(lambda x: len(x.split()))\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)\n",
            "sns.histplot(df['char_length'], bins=30, kde=True, ax=axes[0], color='royalblue')\n",
            "axes[0].set_title('Character Length Distribution', fontweight='bold')\n",
            "axes[0].set_xlabel('Character Count')\n",
            "\n",
            "sns.boxplot(x='category', y='word_count', data=df, ax=axes[1], palette='Set2')\n",
            "axes[1].set_title('Word Count by Category', fontweight='bold')\n",
            "axes[1].set_xlabel('Category')\n",
            "axes[1].tick_params(axis='x', rotation=25)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. NLP Preprocessing Pipeline\n",
            "We apply lowercasing, remove URLs/HTML tags, strip punctuation and digits, tokenize, remove stopwords, and apply lemmatization."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "stop_words = set(stopwords.words('english'))\n",
            "lemmatizer = WordNetLemmatizer()\n",
            "\n",
            "def preprocess_text(text):\n",
            "    if not isinstance(text, str):\n",
            "        return \"\"\n",
            "    text = text.lower()\n",
            "    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)\n",
            "    text = re.sub(r'<.*?>', ' ', text)\n",
            "    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)\n",
            "    text = re.sub(r'\\b\\d+\\b', ' ', text)\n",
            "    tokens = word_tokenize(text)\n",
            "    cleaned = [lemmatizer.lemmatize(t) for t in tokens if len(t) >= 2 and t not in stop_words and not t.isdigit()]\n",
            "    return \" \".join(cleaned)\n",
            "\n",
            "# Demonstration on sample text\n",
            "sample_raw = \"Breaking: Apple announces next-generation M4 chips with 38-core GPU at https://apple.com!\"\n",
            "print(\"Raw Text     :\", sample_raw)\n",
            "print(\"Preprocessed :\", preprocess_text(sample_raw))\n",
            "\n",
            "# Apply to entire dataset\n",
            "df['cleaned_headline'] = df['headline'].apply(preprocess_text)\n",
            "df[['headline', 'cleaned_headline', 'category']].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Train-Test Split and TF-IDF Feature Extraction"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "label_encoder = LabelEncoder()\n",
            "y = label_encoder.fit_transform(df['category'])\n",
            "\n",
            "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
            "    df['cleaned_headline'], y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, min_df=2, sublinear_tf=True)\n",
            "X_train_tfidf = tfidf.fit_transform(X_train_raw)\n",
            "X_test_tfidf = tfidf.transform(X_test_raw)\n",
            "\n",
            "print(f\"Training Matrix Shape: {X_train_tfidf.shape}\")\n",
            "print(f\"Testing Matrix Shape : {X_test_tfidf.shape}\")\n",
            "print(f\"Vocabulary Size      : {len(tfidf.vocabulary_)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Training & Comparison\n",
            "We train and benchmark:\n",
            "1. **Multinomial Naive Bayes**\n",
            "2. **Logistic Regression**\n",
            "3. **Linear Support Vector Classifier** (with Probability Calibration)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    \"Multinomial Naive Bayes\": MultinomialNB(alpha=0.1),\n",
            "    \"Logistic Regression\": LogisticRegression(max_iter=1000, C=1.0, random_state=42),\n",
            "    \"Linear SVM (Calibrated)\": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3)\n",
            "}\n",
            "\n",
            "results = []\n",
            "trained_models = {}\n",
            "\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train_tfidf, y_train)\n",
            "    y_pred = model.predict(X_test_tfidf)\n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    results.append({\"Model\": name, \"Accuracy\": acc})\n",
            "    trained_models[name] = model\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "results_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Confusion Matrix & Classification Report of Best Model"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_model_name = results_df.sort_values(by=\"Accuracy\", ascending=False).iloc[0][\"Model\"]\n",
            "best_model = trained_models[best_model_name]\n",
            "y_pred_best = best_model.predict(X_test_tfidf)\n",
            "\n",
            "print(f\"=== Classification Report for {best_model_name} ===\\n\")\n",
            "print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))\n",
            "\n",
            "# Confusion Matrix Plot\n",
            "cm = confusion_matrix(y_test, y_pred_best)\n",
            "plt.figure(figsize=(8, 6.5), dpi=150)\n",
            "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", \n",
            "            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)\n",
            "plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold')\n",
            "plt.xlabel('Predicted Category')\n",
            "plt.ylabel('Actual Category')\n",
            "plt.xticks(rotation=25)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Real-Time Inference on Custom News Headlines"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def predict_headline(raw_headline):\n",
            "    cleaned = preprocess_text(raw_headline)\n",
            "    vec = tfidf.transform([cleaned])\n",
            "    probs = best_model.predict_proba(vec)[0]\n",
            "    best_idx = np.argmax(probs)\n",
            "    category = label_encoder.classes_[best_idx]\n",
            "    conf = probs[best_idx] * 100\n",
            "    \n",
            "    print(f\"Headline   : {raw_headline}\")\n",
            "    print(f\"Predicted  : {category} ({conf:.2f}% confidence)\")\n",
            "    print(\"Probabilities:\")\n",
            "    for cat, p in sorted(zip(label_encoder.classes_, probs), key=lambda x: x[1], reverse=True):\n",
            "        print(f\"  - {cat:<15}: {p*100:>6.2f}%\")\n",
            "    print(\"-\" * 50)\n",
            "\n",
            "# Test custom unseen headlines\n",
            "predict_headline(\"Nvidia reveals new Blackwell GPU architecture for artificial intelligence\")\n",
            "predict_headline(\"Manchester City defeats Chelsea 2-0 in Premier League opener\")\n",
            "predict_headline(\"Federal Reserve indicates interest rates will remain stable\")\n",
            "predict_headline(\"New clinical study highlights cancer immunotherapy breakthrough\")"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13.7"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_FILE, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print(f"Generated Jupyter Notebook at: {NOTEBOOK_FILE}")
