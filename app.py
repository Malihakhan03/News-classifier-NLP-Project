"""
News Headline Classification - Professional Flask Web Application (15 Categories + Other/Unknown).
Provides:
- Real-time headline classification across 15 main news categories using trained ML models
- Data-driven Out-of-Domain rejection to 'Other / Unknown'
- Empirical statistical uncertainty estimation (flagging borderline/competing categories)
- Genuine calibrated probability / confidence estimation
- Token-level explainability (TF-IDF feature contribution: TF-IDF * w_i)
- Multi-model side-by-side comparison
- Top 3 predicted categories breakdown
- Batch processing for CSV/TXT uploads and export
- Interactive Model Evaluation & Keywords Explorer with Difficult Boundary Benchmark
"""

import os
import io
import csv
import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, Response

from preprocessing import clean_headline, validate_headline, explain_prediction_tokens, clean_headlines_batch

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
EVALUATION_PATH = os.path.join(MODELS_DIR, 'evaluation_results.json')
KEYWORDS_PATH = os.path.join(MODELS_DIR, 'category_keywords.json')

# Data-driven thresholds (loaded from evaluation_results.json or empirical defaults)
# In 15 classes, uniform chance is 1/15 = 6.67%.
UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.298
UNCERTAINTY_MARGIN_THRESHOLD = 0.065
REJECTION_CONFIDENCE_THRESHOLD = 0.15
VOCAB_OVERLAP_THRESHOLD = 0.08
AMBIGUITY_MARGIN_THRESHOLD = 0.08

# Global references
models_dict = {}
vectorizer = None
label_encoder = None
evaluation_data = None
category_keywords_data = {}
categories = []
champion_model_name = "Linear Support Vector Machine"

# 15 Main Categories + Other/Unknown visual palette and icons
CATEGORY_COLORS = {
    "Politics & Government": {"badge": "bg-danger", "color": "#dc3545", "icon": "bi-bank"},
    "World & International": {"badge": "bg-secondary", "color": "#6c757d", "icon": "bi-globe-americas"},
    "Business & Economy": {"badge": "bg-success", "color": "#198754", "icon": "bi-graph-up-arrow"},
    "Technology": {"badge": "bg-info text-dark", "color": "#0dcaf0", "icon": "bi-cpu"},
    "Science": {"badge": "bg-indigo", "color": "#6610f2", "icon": "bi-rocket-takeoff"},
    "Health & Medicine": {"badge": "bg-teal", "color": "#20c997", "icon": "bi-heart-pulse"},
    "Sports": {"badge": "bg-primary", "color": "#0d6efd", "icon": "bi-trophy"},
    "Entertainment & Culture": {"badge": "bg-warning text-dark", "color": "#ffc107", "icon": "bi-film"},
    "Crime & Justice": {"badge": "bg-slate", "color": "#475569", "icon": "bi-shield-shaded"},
    "Environment & Climate": {"badge": "bg-emerald", "color": "#10b981", "icon": "bi-tree"},
    "Automobile": {"badge": "bg-amber", "color": "#f59e0b", "icon": "bi-car-front-fill"},
    "Lifestyle & Travel": {"badge": "bg-pink", "color": "#ec4899", "icon": "bi-compass"},
    "Education": {"badge": "bg-purple", "color": "#8b5cf6", "icon": "bi-mortarboard-fill"},
    "Social Issues & Society": {"badge": "bg-orange", "color": "#f97316", "icon": "bi-people-fill"},
    "Weather & Disaster": {"badge": "bg-cyan text-dark", "color": "#06b6d4", "icon": "bi-cloud-lightning-rain-fill"},
    "Other / Unknown": {"badge": "bg-dark text-white", "color": "#334155", "icon": "bi-question-circle-fill"}
}

# Curated interactive sample headlines covering all 15 categories + Out-of-Domain
SAMPLE_HEADLINES = [
    {"category": "Politics & Government", "headline": "PM meets US president to discuss defence and trade"},
    {"category": "World & International", "headline": "International leaders negotiate a ceasefire in Geneva"},
    {"category": "Business & Economy", "headline": "Markets rally after central bank signals lower interest rates"},
    {"category": "Technology", "headline": "AI startup raises billions to build computing infrastructure"},
    {"category": "Science", "headline": "Scientists detect unusual signals from a distant galaxy"},
    {"category": "Health & Medicine", "headline": "Clinical trials demonstrate high efficacy of breakthrough mRNA vaccine"},
    {"category": "Sports", "headline": "Star striker completes move to European club"},
    {"category": "Entertainment & Culture", "headline": "Biographical historical epic dominates Academy Awards winning Best Picture"},
    {"category": "Crime & Justice", "headline": "Authorities arrest suspects in large-scale online fraud"},
    {"category": "Environment & Climate", "headline": "Global treaty members establish fund for developing nation climate adaptation"},
    {"category": "Automobile", "headline": "Automaker introduces new electric SUV with 400-mile battery"},
    {"category": "Lifestyle & Travel", "headline": "Travel editors reveal curated guide to world's top ten cultural destinations"},
    {"category": "Education", "headline": "Universities announce new admission guidelines and academic criteria for students"},
    {"category": "Social Issues & Society", "headline": "Nonprofit launches nationwide initiative to provide housing for low-income families"},
    {"category": "Weather & Disaster", "headline": "Heavy rainfall causes flooding across several districts"},
    {"category": "Other / Unknown", "headline": "Local bakery announces a new weekend breakfast menu with fresh croissants"}
]


def load_model_artifacts():
    """
    Loads all trained models, vectorizer, label encoder, and evaluation results.
    """
    global models_dict, vectorizer, label_encoder, evaluation_data, category_keywords_data
    global categories, champion_model_name, UNCERTAINTY_CONFIDENCE_THRESHOLD, UNCERTAINTY_MARGIN_THRESHOLD
    global REJECTION_CONFIDENCE_THRESHOLD, VOCAB_OVERLAP_THRESHOLD

    try:
        if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(ENCODER_PATH):
            return False

        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        categories = list(label_encoder.classes_)

        # Load individual models
        model_files = {
            "best": "news_classifier.pkl",
            "naive_bayes": "multinomial_nb.pkl",
            "logistic_regression": "logistic_regression.pkl",
            "linear_svm": "linear_svc.pkl"
        }

        for key, fname in model_files.items():
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.exists(fpath):
                models_dict[key] = joblib.load(fpath)

        # Load evaluation data & empirical thresholds
        if os.path.exists(EVALUATION_PATH):
            with open(EVALUATION_PATH, 'r', encoding='utf-8') as f:
                evaluation_data = json.load(f)
                if "best_model" in evaluation_data and "name" in evaluation_data["best_model"]:
                    champion_model_name = evaluation_data["best_model"]["name"]
                
                if "uncertainty_calibration" in evaluation_data:
                    cal = evaluation_data["uncertainty_calibration"]
                    UNCERTAINTY_CONFIDENCE_THRESHOLD = float(cal.get("uncertainty_confidence_threshold", 0.298))
                    UNCERTAINTY_MARGIN_THRESHOLD = float(cal.get("uncertainty_margin_threshold", 0.065))
                    REJECTION_CONFIDENCE_THRESHOLD = float(cal.get("rejection_confidence_threshold", 0.15))
                    VOCAB_OVERLAP_THRESHOLD = float(cal.get("vocab_overlap_threshold", 0.08))

        # Load category keywords
        if os.path.exists(KEYWORDS_PATH):
            with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                category_keywords_data = json.load(f)

        print(f"[OK] Loaded {len(models_dict)} models, {len(categories)} categories. Champion: {champion_model_name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")
        return False


load_model_artifacts()


def _get_model_prediction(model_obj, features_tfidf):
    """
    Helper to calculate probabilities and predicted class index for any model.
    Uses calibrated predict_proba when available.
    """
    if hasattr(model_obj, "predict_proba"):
        probabilities = model_obj.predict_proba(features_tfidf)[0]
    else:
        raw_scores = model_obj.decision_function(features_tfidf)[0]
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / np.sum(exp_scores)
    
    predicted_idx = int(np.argmax(probabilities))
    return predicted_idx, probabilities


# ==========================================
# Application Routes
# ==========================================

@app.route('/')
def home():
    """
    Main interactive classification dashboard.
    """
    if 'history' not in session:
        session['history'] = []

    model_ready = len(models_dict) > 0 and vectorizer is not None
    return render_template(
        'index.html',
        model_ready=model_ready,
        categories=categories,
        category_colors=CATEGORY_COLORS,
        sample_headlines=SAMPLE_HEADLINES,
        champion_model_name=champion_model_name,
        history=session.get('history', [])
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts category for a news headline with genuine probabilities,
    data-driven uncertainty detection, top-3 competing categories,
    out-of-domain rejection, explainability tokens, and multi-model comparison.
    """
    if not models_dict or vectorizer is None or label_encoder is None:
        if not load_model_artifacts():
            return jsonify({"success": False, "error": "Model files missing. Run 'python train.py' first."}), 503

    if request.is_json:
        data = request.get_json()
        raw_headline = data.get('headline', '')
        model_type = data.get('model', 'best').lower()
    else:
        raw_headline = request.form.get('headline', '')
        model_type = request.form.get('model', 'best').lower()

    # 1. Input validation
    is_valid, validation_error = validate_headline(raw_headline)
    if not is_valid:
        return jsonify({"success": False, "error": validation_error}), 400

    # 2. NLP Preprocessing (identical to training pipeline)
    cleaned_text = clean_headline(raw_headline)
    if not cleaned_text:
        return jsonify({"success": False, "error": "No meaningful words found after stopword and symbol filtering."}), 400

    try:
        # 3. TF-IDF Transformation
        features = vectorizer.transform([cleaned_text])
        feature_sum = float(features.sum())
        target_model = models_dict.get(model_type, models_dict.get('best'))
        
        # 4. ML Model Inference & Calibrated Probabilities
        predicted_idx, probabilities = _get_model_prediction(target_model, features)
        top_prob = float(probabilities[predicted_idx])
        confidence = top_prob * 100.0

        # Sort indices by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        second_prob = float(probabilities[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        second_cat = label_encoder.classes_[sorted_indices[1]] if len(sorted_indices) > 1 else ""
        margin = top_prob - second_prob

        # Extract Top 3 Candidates
        top_3 = []
        for idx in sorted_indices[:3]:
            c_name = label_encoder.classes_[idx]
            c_prob = float(probabilities[idx]) * 100.0
            top_3.append({
                "category": c_name,
                "percentage": round(c_prob, 2),
                "badge": CATEGORY_COLORS.get(c_name, {}).get("badge", "bg-secondary"),
                "icon": CATEGORY_COLORS.get(c_name, {}).get("icon", "bi-tag")
            })

        # 5. Statistical Rejection for "Other / Unknown"
        # Reject strictly when input lacks recognizable domain vocabulary (feature_sum < 0.08)
        # or maximum probability is near uniform random chance (< 15.0%)
        is_unknown = False
        rejection_reason = ""

        if feature_sum < VOCAB_OVERLAP_THRESHOLD:
            is_unknown = True
            predicted_category = "Other / Unknown"
            display_confidence = None
            rejection_reason = "Headline is outside supported 15 news domains or contains no recognizable domain vocabulary."
        elif top_prob < REJECTION_CONFIDENCE_THRESHOLD:
            is_unknown = True
            predicted_category = "Other / Unknown"
            display_confidence = None
            rejection_reason = f"Classification confidence is too low ({round(confidence, 1)}%) to reliably assign to a known category."
        else:
            predicted_category = label_encoder.inverse_transform([predicted_idx])[0]
            display_confidence = round(confidence, 2)

        # 6. Data-Driven Statistical Uncertainty and Narrow Margin Detection
        # If in-domain, flag whether the model is genuinely uncertain based on empirical validation thresholds
        is_uncertain = False
        uncertainty_note = ""
        is_ambiguous = False
        ambiguity_note = ""
        is_narrow_margin = False
        narrow_margin_note = ""

        if not is_unknown and len(sorted_indices) > 1:
            margin_pct = round(margin * 100.0, 2)
            if margin < 0.12:
                is_narrow_margin = True
                narrow_margin_note = f"Close distribution margin ({margin_pct}%): Closely competing with {second_cat} ({round(second_prob*100, 1)}%)."

            if top_prob < UNCERTAINTY_CONFIDENCE_THRESHOLD or margin < UNCERTAINTY_MARGIN_THRESHOLD:
                is_uncertain = True
                is_ambiguous = True
                uncertainty_note = f"Uncertain prediction: Model identified competing topics between {predicted_category} ({round(top_prob*100, 1)}%) and {second_cat} ({round(second_prob*100, 1)}%)."
                ambiguity_note = uncertainty_note

        # 7. Full Probability Distribution across all 15 categories
        prob_distribution = []
        for idx in sorted_indices:
            class_name = label_encoder.classes_[idx]
            prob_percent = float(probabilities[idx]) * 100.0
            prob_distribution.append({
                "category": class_name,
                "percentage": round(prob_percent, 2),
                "is_top": bool(int(idx) == int(predicted_idx) and not is_unknown),
                "color_info": CATEGORY_COLORS.get(class_name, {"badge": "bg-secondary", "color": "#6c757d"})
            })

        # 8. Extract Explainable NLP Tokens (TF-IDF * Model Weights)
        lr_ref = models_dict.get('logistic_regression', target_model)
        token_importance = explain_prediction_tokens(cleaned_text, vectorizer, lr_ref, predicted_idx, top_k=6)

        # 9. Multi-Model Side-by-Side Comparison
        multi_model_results = {}
        for m_key, m_name in [("naive_bayes", "Multinomial Naive Bayes"), 
                              ("logistic_regression", "Logistic Regression"), 
                              ("linear_svm", "Linear Support Vector Machine")]:
            if m_key in models_dict:
                m_obj = models_dict[m_key]
                p_idx, p_probs = _get_model_prediction(m_obj, features)
                p_top_prob = float(p_probs[p_idx])
                if feature_sum < VOCAB_OVERLAP_THRESHOLD or p_top_prob < REJECTION_CONFIDENCE_THRESHOLD:
                    p_cat = "Other / Unknown"
                    p_conf = round(p_top_prob * 100.0, 2)
                else:
                    p_cat = label_encoder.inverse_transform([p_idx])[0]
                    p_conf = round(p_top_prob * 100.0, 2)

                multi_model_results[m_name] = {
                    "category": p_cat,
                    "confidence": p_conf,
                    "badge": CATEGORY_COLORS.get(p_cat, {}).get('badge', 'bg-secondary')
                }

        timestamp = datetime.now().strftime("%I:%M:%S %p, %b %d")

        # Session history
        history_item = {
            "headline": raw_headline.strip(),
            "predicted_category": predicted_category,
            "confidence": display_confidence if display_confidence is not None else round(confidence, 1),
            "is_unknown": is_unknown,
            "is_uncertain": is_uncertain,
            "timestamp": timestamp
        }
        history = session.get('history', [])
        history.insert(0, history_item)
        session['history'] = history[:10]
        session.modified = True

        return jsonify({
            "success": True,
            "headline": raw_headline.strip(),
            "cleaned_tokens": cleaned_text,
            "predicted_category": predicted_category,
            "confidence": display_confidence,
            "raw_confidence": round(confidence, 2),
            "is_unknown": is_unknown,
            "rejection_reason": rejection_reason,
            "is_uncertain": is_uncertain,
            "uncertainty_note": uncertainty_note,
            "is_ambiguous": is_ambiguous,
            "ambiguity_note": ambiguity_note,
            "is_narrow_margin": is_narrow_margin,
            "narrow_margin_note": narrow_margin_note,
            "margin": round(margin * 100.0, 2) if len(sorted_indices) > 1 else 100.0,
            "second_category": second_cat if len(sorted_indices) > 1 else "",
            "second_confidence": round(second_prob * 100.0, 2) if len(sorted_indices) > 1 else 0.0,
            "top_3": top_3,
            "probability_distribution": prob_distribution,
            "token_importance": token_importance,
            "multi_model_comparison": multi_model_results,
            "model_used": model_type,
            "champion_model_name": champion_model_name,
            "timestamp": timestamp,
            "category_info": CATEGORY_COLORS.get(predicted_category, {"badge": "bg-secondary", "color": "#6c757d"})
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction error: {str(e)}"}), 500


@app.route('/batch')
def batch_page():
    """
    Batch classification dashboard page.
    """
    return render_template(
        'batch.html',
        categories=categories,
        category_colors=CATEGORY_COLORS,
        champion_model_name=champion_model_name
    )


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch headline prediction endpoint accepting uploaded CSV or TXT file or raw text.
    Returns categorized items, category distribution summary, and uncertainty status.
    """
    if not models_dict or vectorizer is None or label_encoder is None:
        if not load_model_artifacts():
            return jsonify({"success": False, "error": "Model files missing. Run 'python train.py' first."}), 503

    headlines = []

    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
            reader = csv.reader(stream)
            rows = list(reader)
            if rows:
                first_cell = rows[0][0].strip().lower()
                start_idx = 1 if first_cell in ['headline', 'title', 'text', 'news'] else 0
                for row in rows[start_idx:]:
                    if row and row[0].strip():
                        headlines.append(row[0].strip())
        elif filename.endswith('.txt'):
            content = file.stream.read().decode("utf-8", errors="ignore")
            for line in content.splitlines():
                if line.strip():
                    headlines.append(line.strip())
        else:
            return jsonify({"success": False, "error": "Please upload a .csv or .txt file."}), 400

    elif request.is_json:
        data = request.get_json()
        headlines = data.get('headlines', [])
    else:
        raw_text = request.form.get('text_lines', '')
        headlines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not headlines:
        return jsonify({"success": False, "error": "No valid headlines found to classify."}), 400

    headlines = headlines[:500]
    cleaned_batch = clean_headlines_batch(headlines)
    
    valid_records = []
    for orig, cln in zip(headlines, cleaned_batch):
        if cln:
            valid_records.append((orig, cln))
        else:
            valid_records.append((orig, orig.lower()))

    cleaned_texts = [r[1] for r in valid_records]
    features_batch = vectorizer.transform(cleaned_texts)

    target_model = models_dict.get('best')
    if hasattr(target_model, "predict_proba"):
        all_probs = target_model.predict_proba(features_batch)
    else:
        scores = target_model.decision_function(features_batch)
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        all_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    feature_sums = np.array(features_batch.sum(axis=1)).flatten()

    results = []
    category_counts = {cat: 0 for cat in categories}
    category_counts["Other / Unknown"] = 0

    for i, (orig, _) in enumerate(valid_records):
        probs = all_probs[i]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        conf = top_prob * 100.0
        f_sum = feature_sums[i]

        sorted_p = np.sort(probs)[::-1]
        second_prob = sorted_p[1] if len(sorted_p) > 1 else 0.0
        margin = top_prob - second_prob

        is_uncertain = bool(top_prob < UNCERTAINTY_CONFIDENCE_THRESHOLD or margin < UNCERTAINTY_MARGIN_THRESHOLD)

        if f_sum < VOCAB_OVERLAP_THRESHOLD or top_prob < REJECTION_CONFIDENCE_THRESHOLD:
            cat = "Other / Unknown"
            category_counts["Other / Unknown"] += 1
            conf_display = None
        else:
            cat = label_encoder.inverse_transform([top_idx])[0]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            conf_display = round(conf, 2)

        results.append({
            "id": i + 1,
            "headline": orig,
            "predicted_category": cat,
            "confidence": conf_display,
            "raw_confidence": round(conf, 2),
            "is_uncertain": is_uncertain,
            "badge": CATEGORY_COLORS.get(cat, {}).get('badge', 'bg-secondary')
        })

    return jsonify({
        "success": True,
        "total_classified": len(results),
        "results": results,
        "category_distribution": category_counts
    })


@app.route('/evaluation')
def evaluation():
    """
    Model evaluation page showing dataset stats, cross-validation,
    model comparison, confusion matrix, category keywords explorer,
    and difficult boundary holdout benchmark.
    """
    global evaluation_data, category_keywords_data
    if evaluation_data is None:
        load_model_artifacts()

    diff_benchmark = evaluation_data.get('difficult_benchmark', {}) if evaluation_data else {}
    uncertainty_cal = evaluation_data.get('uncertainty_calibration', {}) if evaluation_data else {}

    return render_template(
        'evaluation.html',
        evaluation=evaluation_data,
        category_keywords=category_keywords_data,
        difficult_benchmark=diff_benchmark,
        uncertainty_cal=uncertainty_cal,
        categories=categories,
        category_colors=CATEGORY_COLORS,
        champion_model_name=champion_model_name
    )


@app.route('/api/keywords/<category>')
def get_category_keywords(category):
    """
    Returns top discriminative keywords for a category.
    """
    if not category_keywords_data:
        load_model_artifacts()
    keywords = category_keywords_data.get(category, [])
    return jsonify({
        "success": True,
        "category": category,
        "keywords": keywords,
        "color_info": CATEGORY_COLORS.get(category, {"badge": "bg-secondary", "color": "#6c757d"})
    })


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """
    Clears user prediction history stored in session.
    """
    session['history'] = []
    session.modified = True
    return jsonify({"success": True, "message": "History cleared successfully."})

app = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f"  News Classifier Web App Live at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
