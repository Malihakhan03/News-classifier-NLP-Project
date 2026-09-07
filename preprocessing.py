"""
NLP Preprocessing Pipeline for News Headline Classification.
Provides unified text cleaning, tokenization, stopword removal,
lemmatization, input validation, and token-level explainability.
"""

import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources():
    """
    Ensures all necessary NLTK corpora and tokenizers are available.
    """
    resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            if resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()

# Initialize global lemmatizer and stopwords set
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
        'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
        'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    }

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None


# Domain Acronym Expansion Map for News Headlines
# Preserves both the full semantic expansion and the original acronym token
ACRONYM_MAP = {
    r'\bmou\b': 'memorandum of understanding mou',
    r'\bmous\b': 'memorandum of understanding mou',
    r'\bpm\b': 'prime minister pm',
    r'\bun\b': 'united nations un',
    r'\bnato\b': 'north atlantic treaty organization nato',
    r'\beu\b': 'european union eu',
    r'\bceo\b': 'chief executive officer ceo',
    r'\bcfo\b': 'chief financial officer cfo',
    r'\bgdp\b': 'gross domestic product gdp',
    r'\bai\b': 'artificial intelligence ai',
    r'\bev\b': 'electric vehicle ev',
    r'\bevs\b': 'electric vehicles ev',
    r'\bfda\b': 'food and drug administration fda',
    r'\bwho\b': 'world health organization who',
    r'\bimf\b': 'international monetary fund imf',
    r'\bwto\b': 'world trade organization wto',
    r'\bg7\b': 'group of seven g7',
    r'\bg20\b': 'group of twenty g20',
    r'\bisro\b': 'indian space research organisation isro',
    r'\bnasa\b': 'national aeronautics and space administration nasa',
    r'\bepa\b': 'environmental protection agency epa',
    r'\bsec\b': 'securities and exchange commission sec',
    r'\bfbi\b': 'federal bureau of investigation fbi',
    r'\bcia\b': 'central intelligence agency cia',
    r'\bcop\b': 'climate conference cop',
    r'\bmp\b': 'member of parliament mp',
    r'\bmps\b': 'members of parliament mp',
    r'\bmla\b': 'member of legislative assembly mla',
    r'\bmlas\b': 'members of legislative assembly mla',
    r'\bipo\b': 'initial public offering ipo',
    r'\brbi\b': 'reserve bank of india rbi',
    r'\bfed\b': 'federal reserve fed',
    r'\becb\b': 'european central bank ecb',
    r'\bicc\b': 'international cricket council icc',
    r'\bfifa\b': 'international association football federation fifa',
    r'\bf1\b': 'formula one f1',
}

# Pre-compile compiled regex patterns for efficiency
COMPILED_ACRONYMS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ACRONYM_MAP.items()]


def expand_acronyms(text: str) -> str:
    """
    Expands common news abbreviations and acronyms to their full semantic forms
    while retaining the acronym token, enriching downstream TF-IDF n-grams.
    """
    if not text:
        return ""
    expanded = text
    for pattern, replacement in COMPILED_ACRONYMS:
        expanded = pattern.sub(replacement, expanded)
    return expanded


def _lemmatize_token(token: str) -> str:
    """
    Lemmatizes an alphabetic token with POS awareness.
    If the token ends in -ed or -ing, checks verb lemmatization (visited -> visit, signing -> sign).
    Otherwise applies noun lemmatization (cars -> car, countries -> country) while preserving
    names ending in -s (e.g., James -> James).
    """
    if not LEMMATIZER or not token.isalpha():
        return token
    # Check verb lemmatization for verbal inflections
    if token.endswith(('ed', 'ing')):
        lemma_v = LEMMATIZER.lemmatize(token, pos='v')
        if lemma_v != token:
            return lemma_v
    # Default to noun lemmatization for plurals
    lemma_n = LEMMATIZER.lemmatize(token, pos='n')
    if lemma_n != token:
        return lemma_n
    return token


def clean_headline(text: str) -> str:
    """
    Applies the full NLP preprocessing pipeline to a news headline.

    Steps:
    1. Remove URLs (http/https/www)
    2. Remove HTML markup / tags
    3. Expand standard domain acronyms (MOU, PM, UN, NATO, CEO, etc.)
    4. Convert text to lowercase
    5. Remove punctuation while preserving alphanumeric tokens (e.g. iPhone 16, Awarapan 2, COVID-19)
    6. Tokenize text into words
    7. Filter out English stopwords
    8. Apply POS-aware WordNet lemmatization on alphabetic tokens (verbs and nouns)
    9. Normalize whitespace

    Args:
        text (str): Raw input headline

    Returns:
        str: Preprocessed, space-joined cleaned text token sequence
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 3. Expand acronyms before removing punctuation
    text = expand_acronyms(text)

    # 4. Convert to lowercase
    text = text.lower()

    # 5. Remove punctuation and special symbols, keeping letters and digits
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # 6. Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 7. Remove stopwords & filter noise, and 8. Lemmatize
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if not token or token in STOP_WORDS:
            continue

        # Keep meaningful alphanumeric tokens (e.g. '2' in 'Awarapan 2', '16' in 'iPhone 16')
        # Filter out excessively long random numeric strings (> 4 digits)
        if token.isdigit() and len(token) > 4:
            continue

        # Lemmatize alphabetic tokens
        if token.isalpha():
            if len(token) < 2 and token not in {'a', 'i'}:
                continue
            lemma = _lemmatize_token(token)
            cleaned_tokens.append(lemma)
        else:
            # Alphanumeric or numeric token
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens).strip()


def clean_headlines_batch(texts: list[str]) -> list[str]:
    """
    Cleans a batch list of headlines efficiently.
    """
    return [clean_headline(t) for t in texts]


def validate_headline(text: str) -> tuple[bool, str]:
    """
    Validates a user-submitted headline before prediction.
    Accepts valid short headlines (e.g., 'NASA Mars rover', 'Tesla EV', 'GDP 7%').
    Rejects empty, whitespace-only, symbol-only, or number-only inputs.
    """
    if text is None or not isinstance(text, str):
        return False, "Headline cannot be empty. Please enter a news headline."

    raw_stripped = text.strip()
    if len(raw_stripped) == 0:
        return False, "Headline cannot be empty. Please enter a news headline."

    if len(raw_stripped) < 3:
        return False, "Headline is too short. Please enter a complete news headline (at least 3 characters)."

    if not re.search(r'[a-zA-Z]', raw_stripped):
        return False, "Headline must contain alphabetic words, not only numbers or symbols."

    cleaned = clean_headline(raw_stripped)
    if not cleaned:
        return False, "Headline does not contain recognizable words after removing stopwords and special characters."

    return True, ""


def explain_prediction_tokens(cleaned_text: str, tfidf_vectorizer, model_obj, class_idx: int, top_k: int = 6) -> list[dict]:
    """
    Computes genuine token-level contribution scores for words present in the headline.
    Uses TF-IDF feature value * model coefficient for the predicted class.

    Returns:
        list[dict]: Top contributing n-grams with contribution score and positive/negative impact.
    """
    if not cleaned_text or tfidf_vectorizer is None or model_obj is None:
        return []

    # Transform headline to get exact TF-IDF weights for active features
    tfidf_vec = tfidf_vectorizer.transform([cleaned_text]).toarray()[0]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    nonzero_indices = np.where(tfidf_vec > 0)[0]

    if len(nonzero_indices) == 0:
        return []

    # Extract coefficients for the predicted class
    coefs = None
    if hasattr(model_obj, 'coef_'):
        coefs = model_obj.coef_[class_idx]
    elif hasattr(model_obj, 'calibrated_classifiers_'):
        # For CalibratedClassifierCV, average coefficients across calibrated base estimators
        sub_coefs = []
        for clf in model_obj.calibrated_classifiers_:
            base_estimator = clf.estimator if hasattr(clf, 'estimator') else clf.base_estimator
            if hasattr(base_estimator, 'coef_'):
                sub_coefs.append(base_estimator.coef_[class_idx])
        if sub_coefs:
            coefs = np.mean(sub_coefs, axis=0)
    elif hasattr(model_obj, 'feature_log_prob_'):
        # For MultinomialNB: log-likelihood difference from uniform base
        coefs = model_obj.feature_log_prob_[class_idx] - np.mean(model_obj.feature_log_prob_, axis=0)

    token_scores = []
    for feat_idx in nonzero_indices:
        token_name = feature_names[feat_idx]
        tfidf_val = float(tfidf_vec[feat_idx])
        
        if coefs is not None and feat_idx < len(coefs):
            coeff_val = float(coefs[feat_idx])
            contribution = tfidf_val * coeff_val
        else:
            contribution = tfidf_val

        impact = "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral"

        token_scores.append({
            "token": token_name,
            "importance": float(round(contribution, 4)),
            "tfidf_weight": float(round(tfidf_val, 4)),
            "impact": impact
        })

    # Sort descending by contribution
    token_scores = sorted(token_scores, key=lambda x: x["importance"], reverse=True)
    return token_scores[:top_k]
