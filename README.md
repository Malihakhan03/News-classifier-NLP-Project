# News Headline Classification Using NLP (15 News Categories + Other/Unknown)

An end-to-end Natural Language Processing (NLP) and Machine Learning system for multiclass news headline classification across **15 comprehensive news domains** plus an automated out-of-domain statistical rejection mechanism (**Other / Unknown**). Built with Python, NLTK, Scikit-learn, and Flask, featuring token-level explainability, Stratified 5-Fold Cross-Validation, and an interactive web interface.

---

## 1. Project Objective

The objective of this project is to automatically categorize short news headlines into distinct domain categories using classical NLP text processing and supervised machine learning algorithms. The system emphasizes:
- Rigorous data leakage prevention between training and evaluation splits.
- Stratified 5-Fold Cross-Validation on training data for unbiased model selection.
- Clear taxonomic boundaries (e.g. domestic *Politics & Government* vs foreign *World & International*).
- Statistical rejection mechanism for Out-of-Domain and Low-Confidence headlines (**Other / Unknown**).
- Zero hardcoded keyword rules — all classifications derive from learned statistical weights.
- Local token explainability via feature-weight contributions ($w_i \times \text{TF-IDF}_i$).

---

## 2. Supported 15 Main News Categories + Other/Unknown

| # | Category | Description / Typical Domains | Example Headline |
| :-: | :--- | :--- | :--- |
| 1 | **Politics & Government** | Elections, political parties, parliament, domestic policy, government schemes | *"Government introduces comprehensive election reform bill in parliament"* |
| 2 | **World & International** | Bilateral treaties, international relations, UN summits, foreign diplomacy, MOUs | *"India and United States sign comprehensive bilateral cooperation agreement and MoU"* |
| 3 | **Business & Economy** | Markets, central banks, interest rates, inflation, corporate earnings, GDP | *"Federal Reserve cuts benchmark interest rate by 50 basis points as inflation cools"* |
| 4 | **Technology** | AI, semiconductors, software, cloud, cybersecurity, smartphones | *"Apple announces next generation M4 chip with neural engine for AI workloads"* |
| 5 | **Science** | Astronomy, space telescopes, physics, paleontology, quantum research | *"James Webb Space Telescope detects atmospheric water vapor on distant exoplanet"* |
| 6 | **Health & Medicine** | Diseases, clinical trials, mRNA vaccines, cardiology, oncology, medical research | *"Clinical trials demonstrate high efficacy of breakthrough mRNA vaccine for cancer"* |
| 7 | **Sports** | Football, cricket, tennis, Olympics, basketball, Formula 1, championships | *"Real Madrid wins 15th Champions League trophy with dramatic victory"* |
| 8 | **Entertainment & Culture** | Movies, film reviews, sequels, actors, music albums, streaming web series | *"Awarapan 2 launched in September and reviews are excellent"* |
| 9 | **Crime & Justice** | Police, cybercrime, courts, trials, fraud investigations, Supreme Court | *"Federal law enforcement coalition dismantles international cybercrime syndicate"* |
| 10 | **Environment & Climate** | Renewable energy, climate change, rainforest conservation, wildlife, pollution | *"Global renewable electricity generation surpasses 30 percent of total power"* |
| 11 | **Automobile** | EVs, vehicle launches, solid-state batteries, automotive technology, supercars | *"Automotive manufacturer debuts luxury electric sedan with 600-mile range battery"* |
| 12 | **Lifestyle & Travel** | Tourism, vacation destinations, Michelin guides, culinary arts, hospitality | *"Travel publication reveals world's top ten trending tourist destinations"* |
| 13 | **Education** | Universities, colleges, admissions, school exams, scholarships, pedagogy | *"Universities announce new admission guidelines and entrance exam criteria for students"* |
| 14 | **Social Issues & Society** | Rural poverty, youth unemployment, community welfare, disability rights | *"Government launches program to support rural families and reduce unemployment"* |
| 15 | **Weather & Disaster** | Cyclones, flash floods, earthquakes, heatwaves, hurricanes, droughts | *"Heavy rainfall triggers flash floods across low-lying districts inundating roads"* |
| 16 | **Other / Unknown** | Out-of-domain inputs, low-confidence or non-news queries | *"Local bakery announces a new weekend breakfast menu with fresh croissants"* |

---

## 3. Out-of-Domain Rejection Mechanism (Other / Unknown)

Rather than forcing every arbitrary sentence into one of the 15 predefined news categories, the system implements a statistical rejection pipeline:
1. **Vocabulary Overlap Check**: If the sum of active TF-IDF weights is negligible ($\text{feature\_sum} < 0.15$), the input is recognized as containing out-of-vocabulary terms and rejected as **Other / Unknown**.
2. **Confidence Threshold**: In a 15-class system (uniform chance $\approx 6.67\%$), if the calibrated probability of the top category is below $30.0\%$, the headline is flagged as **Other / Unknown**.
3. **Cross-Model Probability Floor**: If both generative (Naive Bayes $< 32\%$) and discriminative (Logistic Regression $< 25\%$) models report low confidence, the headline is rejected.

---

## 4. NLP Preprocessing Pipeline

The unified text preprocessing pipeline is implemented in `preprocessing.py` and applied identically during training and inference:

1. **Lowercasing**: Standardizes text case.
2. **URL & HTML Stripping**: Removes web links (`http/https/www`) and HTML tags (`<...>`).
3. **Alphanumeric Preservation**: Cleans non-alphanumeric noise while preserving compound numbers and technical versions (`iPhone 17`, `COVID-19`, `Formula 1`, `Awarapan 2`, `GDP 7%`, `Election 2026`).
4. **Tokenization**: Words tokenized using NLTK `word_tokenize`.
5. **Stopword Filtering**: Removes generic English stopwords (`the`, `is`, `at`, `which`).
6. **WordNet Lemmatization**: Normalizes words to canonical dictionary roots (`launches` $\to$ `launch`, `reviews` $\to$ `review`).

---

## 5. TF-IDF Feature Extraction

- **N-gram Range**: Unigrams and Bigrams `(1, 2)` to capture phrase context (`bilateral cooperation`, `movie review`, `interest rate`, `space telescope`).
- **Feature Space**: 12,000 max n-gram features.
- **Sublinear Term Frequency**: Applied $\text{sublinear\_tf}=\text{True}$ ($1 + \log(\text{tf})$) to dampen disproportionate high-frequency word impact.
- **Leakage-Free Fitting**: The `TfidfVectorizer` is fitted **strictly on `X_train`** and only transforms `X_test`.

---

## 6. Machine Learning Models & Cross-Validation

Three distinct machine learning classification algorithms are trained and compared:

1. **Multinomial Naive Bayes (`MultinomialNB`)**: Probabilistic bag-of-words classifier with Laplace smoothing ($\alpha=0.05$).
2. **Logistic Regression (`LogisticRegression`)**: Multiclass logistic regression with L2 regularization ($C=2.0$).
3. **Linear Support Vector Machine (`LinearSVC` + `CalibratedClassifierCV`)**: Maximum-margin linear classifier with Platt probability calibration via 3-fold cross-validation.

### Model Selection Workflow
```text
Dataset (1,977 samples)
   ↓
Train/Test Split (80% Train / 20% Test)
   ↓
Training Data (1,581 samples)
   ↓
Stratified 5-Fold Cross-Validation (on X_train)
   ↓
Model Selection (Champion: Linear SVM based on CV F1: 86.64%)
   ↓
Final Evaluation on Untouched Test Set (396 samples)
```

---

## 7. Benchmark Performance Results

### A. 5-Fold Stratified Cross-Validation on Training Set (1,581 samples)

| Model | CV Mean Accuracy | CV Mean Precision (Weighted) | CV Mean Recall (Weighted) | CV Mean F1 (Weighted) | CV Mean F1 (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine (Champion)** | **86.65%** | **87.12%** | **86.65%** | **86.64%** | **86.04%** |
| **Multinomial Naive Bayes** | 85.39% | 85.81% | 85.39% | 85.40% | 84.82% |
| **Logistic Regression (L2)** | 84.25% | 84.58% | 84.25% | 84.05% | 83.50% |

### B. Final Evaluation on Holdout Test Set (396 samples, 0% Data Leakage)

| Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Champion)** | **84.85%** | **85.03%** | **84.85%** | **84.69%** | **84.79%** | **84.23%** | **84.34%** | **0.44s** |
| **Logistic Regression** | **85.35%** | **85.72%** | **85.35%** | **85.25%** | **86.06%** | **84.53%** | **85.01%** | **1.78s** |
| **Multinomial Naive Bayes** | **84.60%** | **84.86%** | **84.60%** | **84.54%** | **84.76%** | **83.77%** | **84.09%** | **0.02s** |

---

## 8. Specific Verification Test Results

| Test Headline | Expected Category | Predicted Category | Confidence | Status |
| :--- | :--- | :--- | :---: | :---: |
| *"Awarapan 2 launched in September and reviews are excellent"* | Entertainment & Culture | **Entertainment & Culture** | 92.23% | **PASS** |
| *"Apple launches a new AI-powered processor"* | Technology | **Technology** | 86.55% | **PASS** |
| *"India and America sign a bilateral cooperation agreement"* | World & International | **World & International** | 91.40% | **PASS** |
| *"Universities announce new admission guidelines for students"* | Education | **Education** | 92.90% | **PASS** |
| *"Heavy rainfall causes flooding across several districts"* | Weather & Disaster | **Weather & Disaster** | 92.24% | **PASS** |
| *"New government program aims to reduce rural unemployment"* | Social Issues & Society | **Social Issues & Society** | 61.89% | **PASS** |
| *"Local bakery announces a new weekend menu"* | Other / Unknown | **Other / Unknown** | Uncertain | **PASS** |
| *"Modi visited America for an MoU and bilateral cooperation"* | World & International | **World & International** | 88.52% | **PASS** |

---

## 9. Token Explainability Framework

The project implements mathematical feature-attribution without hardcoded rules:
$$\text{Contribution}_i = \text{TF-IDF}_i \times w_{\text{class}, i}$$
Where:
- $\text{TF-IDF}_i$ is the exact TF-IDF weight of word/bigram $i$ in the headline.
- $w_{\text{class}, i}$ is the learned model coefficient for the predicted class.
- Words with $\text{Contribution}_i > 0$ are labeled **Positive Impact**, while words with negative weights are labeled **Negative Impact**.

---

## 10. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 2: (Optional) Retrain Models
```bash
python generate_dataset.py
python train.py
```

### Step 3: Run Automated Test Suite
```bash
python test_nlp.py
```

### Step 4: Start Web Application
```bash
python app.py
```
Open in browser:
- **Dashboard**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Batch Mode**: [http://127.0.0.1:5000/batch](http://127.0.0.1:5000/batch)
- **Model Evaluation**: [http://127.0.0.1:5000/evaluation](http://127.0.0.1:5000/evaluation)
