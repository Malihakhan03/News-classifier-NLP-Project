"""
News Headline Classification Training Pipeline.
Trains, cross-validates, and evaluates:
1. Multinomial Naive Bayes
2. Logistic Regression (L2 Regularized with balanced class weights)
3. Linear Support Vector Machine (with Probability Calibration and balanced class weights)

Uses Stratified 5-Fold Cross-Validation on the training set for unbiased model selection.
Evaluates on:
1. Untouched 20% Normal Holdout Test Set
2. Dedicated Difficult Boundary Holdout Test Set ('data/difficult_test_set.csv')

Saves all models, TF-IDF vectorizer, label encoder, confusion matrices,
data-driven uncertainty thresholds, and category explainability keywords.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import clean_headline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'news.csv')
DIFFICULT_PATH = os.path.join(BASE_DIR, 'data', 'difficult_test_set.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

TEXT_COLUMN_CANDIDATES = ['headline', 'text', 'title', 'news', 'content']
LABEL_COLUMN_CANDIDATES = ['category', 'label', 'class', 'target', 'topic']


def load_and_validate_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'.")

    print(f"[1/9] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    text_col = None
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            text_col = candidate
            break

    label_col = None
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            label_col = candidate
            break

    if not text_col or not label_col:
        raise ValueError(f"Could not detect text and label columns in {list(df.columns)}")

    print(f"      Detected Text Column: '{text_col}', Label Column: '{label_col}'")
    df = df.rename(columns={text_col: 'headline', label_col: 'category'})
    df = df[['headline', 'category']]

    initial_count = len(df)
    df = df.dropna(subset=['headline', 'category'])
    df['headline'] = df['headline'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    df = df[df['headline'].str.len() > 0]
    df = df.drop_duplicates(subset=['headline']).reset_index(drop=True)
    final_count = len(df)

    print(f"      Initial records: {initial_count} | Clean unique records: {final_count}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[2/9] Preprocessing {len(df)} headlines with NLTK...")
    start_time = time.time()
    df['cleaned_headline'] = df['headline'].apply(clean_headline)
    df = df[df['cleaned_headline'].str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=['cleaned_headline']).reset_index(drop=True)
    elapsed = time.time() - start_time
    print(f"      Preprocessing completed in {elapsed:.2f}s. Usable unique samples: {len(df)}")
    return df


def generate_class_distribution_plot(df: pd.DataFrame, output_path: str):
    counts = df['category'].value_counts()
    plt.figure(figsize=(14, 6.5), dpi=300)
    sns.set_theme(style="whitegrid", palette="muted")
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    
    plt.title("News Category Distribution across 15 News Categories", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Category", fontsize=11, fontweight='semibold')
    plt.ylabel("Number of Headlines", fontsize=11, fontweight='semibold')
    plt.xticks(rotation=35, ha='right', fontsize=9.5)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"      Saved class distribution plot to: {output_path}")


def generate_model_comparison_plot(metrics_dict: dict, output_path: str):
    data = []
    for model_name, m in metrics_dict.items():
        data.append({"Model": model_name, "Metric": "Accuracy", "Score": m["accuracy"] * 100})
        data.append({"Model": model_name, "Metric": "Precision", "Score": m["precision_weighted"] * 100})
        data.append({"Model": model_name, "Metric": "Recall", "Score": m["recall_weighted"] * 100})
        data.append({"Model": model_name, "Metric": "F1-Score", "Score": m["f1_weighted"] * 100})

    comp_df = pd.DataFrame(data)

    plt.figure(figsize=(11, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", 4)
    
    ax = sns.barplot(data=comp_df, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Model Performance Comparison on Holdout Test Set", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Machine Learning Model", fontsize=11, fontweight='semibold')
    plt.ylabel("Score (%)", fontsize=11, fontweight='semibold')
    plt.ylim(0, 105)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"      Saved model comparison plot to: {output_path}")


def generate_confusion_matrix_plot(cm: np.ndarray, labels: list, model_name: str, output_path: str):
    plt.figure(figsize=(13, 10.5), dpi=300)
    sns.set_theme(style="white")
    
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, where=cm_sum != 0) * 100
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = "0"
            else:
                annot[i, j] = f"{c}\n({p:.0f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Number of Test Predictions'},
        linewidths=0.5,
        linecolor='lightgray',
        annot_kws={"size": 7.5}
    )
    
    plt.title(f"Confusion Matrix (Holdout Test Set) — {model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Category", fontsize=11, fontweight='semibold')
    plt.ylabel("True Category", fontsize=11, fontweight='semibold')
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"      Saved confusion matrix plot to: {output_path}")


def extract_top_category_keywords(tfidf: TfidfVectorizer, lr_model: LogisticRegression, class_names: list, top_k: int = 15) -> dict:
    feature_names = np.array(tfidf.get_feature_names_out())
    category_keywords = {}

    for idx, category in enumerate(class_names):
        coefs = lr_model.coef_[idx]
        top_indices = np.argsort(coefs)[-top_k:][::-1]
        top_features = []
        for feat_idx in top_indices:
            weight = float(coefs[feat_idx])
            if weight > 0:
                top_features.append({
                    "keyword": str(feature_names[feat_idx]),
                    "weight": float(round(weight, 4))
                })
        category_keywords[category] = top_features

    return category_keywords


def evaluate_model_on_data(model, X_tfidf, y_encoded, class_names):
    y_pred = model.predict(X_tfidf)
    acc = accuracy_score(y_encoded, y_pred)
    prec_w = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
    
    prec_m = precision_score(y_encoded, y_pred, average='macro', zero_division=0)
    rec_m = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
    f1_m = f1_score(y_encoded, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_encoded, y_pred)
    report = classification_report(y_encoded, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Probabilities for uncertainty analysis
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_tfidf)
    else:
        scores = model.decision_function(X_tfidf)
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    top_probs = np.max(probs, axis=1)
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_p[:, 0] - sorted_p[:, 1]

    return {
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": cm,
        "classification_report": report,
        "top_probs": top_probs,
        "margins": margins
    }


def main():
    print("=" * 75)
    print("    NEWS HEADLINE CLASSIFICATION - ADVANCED GENERALIZATION PIPELINE   ")
    print("=" * 75)

    # 1. Load Dataset
    df = load_and_validate_dataset(DATA_PATH)
    
    # 2. Preprocess Dataset
    df = preprocess_dataset(df)

    # 3. Analyze Class Distribution & Generate Plot
    print(f"[3/9] Analyzing class distribution across {df['category'].nunique()} categories...")
    class_counts = df['category'].value_counts().to_dict()
    for cat, count in class_counts.items():
        print(f"      • {cat:<25}: {count} headlines ({count / len(df) * 100:.1f}%)")
    
    class_dist_plot = os.path.join(STATIC_IMG_DIR, 'class_distribution.png')
    generate_class_distribution_plot(df, class_dist_plot)

    # 4. Encode Labels and Split Data (80% Train, 20% Test, Stratified)
    print(f"[4/9] Encoding labels and splitting dataset (80% Train, 20% Test, Stratified)...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['category'])
    class_names = list(label_encoder.classes_)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_headline'],
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )
    
    # Verify 0% data leakage
    overlap = set(X_train_raw).intersection(set(X_test_raw))
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} overlapping headlines."
    print(f"      Train Samples: {len(X_train_raw)} | Test Samples: {len(X_test_raw)}")

    # 5. TF-IDF Feature Extraction with Optimized Contextual N-Gram Hyperparameters
    # Word unigrams + bigrams + trigrams, min_df=2 (filters isolated typos), max_df=0.85, sublinear_tf=True
    print(f"[5/9] Fitting optimized TF-IDF Vectorizer (Unigram + Bigram + Trigram, max_features=35000, min_df=2, max_df=0.85)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=35000,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm='l2'
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    vocab_size = len(tfidf.vocabulary_)
    print(f"      Vocabulary Size extracted: {vocab_size} unique n-gram features")

    # 6. Candidate Models Definition with Balanced Class Weights & Regularization
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.05),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        ),
        "Linear Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=42),
            cv=3
        )
    }

    model_file_keys = {
        "Multinomial Naive Bayes": "multinomial_nb.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Linear Support Vector Machine": "linear_svc.pkl"
    }

    # 7. Stratified 5-Fold Cross-Validation on Training Data for Unbiased Model Selection
    print(f"[6/9] Performing 5-Fold Stratified Cross-Validation on Training Data for Model Selection...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_summary = {}

    for name, model in candidate_models.items():
        print(f"      --> Running 5-Fold CV: {name}...")
        scoring = ['accuracy', 'f1_weighted', 'f1_macro']
        cv_scores = cross_validate(model, X_train_tfidf, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_acc_mean = float(np.mean(cv_scores['test_accuracy']))
        cv_f1_w_mean = float(np.mean(cv_scores['test_f1_weighted']))
        cv_f1_m_mean = float(np.mean(cv_scores['test_f1_macro']))
        
        cv_results_summary[name] = {
            "cv_accuracy_mean": cv_acc_mean,
            "cv_f1_weighted_mean": cv_f1_w_mean,
            "cv_f1_macro_mean": cv_f1_m_mean,
            "cv_scores": {k: [float(x) for x in v] for k, v in cv_scores.items()}
        }
        print(f"          CV Acc: {cv_acc_mean*100:.2f}% | CV Weighted F1: {cv_f1_w_mean*100:.2f}% | CV Macro F1: {cv_f1_m_mean*100:.2f}%")

    # Select champion model based on highest mean CV Macro F1 (all 15 classes performing well)
    champion_model_name = max(cv_results_summary, key=lambda k: cv_results_summary[k]['cv_f1_macro_mean'])
    print(f"\n[*] CHAMPION MODEL SELECTED VIA 5-FOLD CV: {champion_model_name}")
    print(f"    (CV Macro F1: {cv_results_summary[champion_model_name]['cv_f1_macro_mean']*100:.2f}% | Weighted F1: {cv_results_summary[champion_model_name]['cv_f1_weighted_mean']*100:.2f}%)")

    # 8. Train All Models on Full Training Data and Evaluate on Normal Holdout Test Set
    print(f"\n[7/9] Retraining all models on full X_train and evaluating on Holdout Test Set...")
    test_results = {}
    trained_model_objs = {}
    confusion_matrices = {}
    classification_reports_dict = {}
    eval_raw_dict = {}

    for name, model in candidate_models.items():
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - t0

        eval_res = evaluate_model_on_data(model, X_test_tfidf, y_test, class_names)
        eval_raw_dict[name] = eval_res

        test_results[name] = {
            "accuracy": eval_res["accuracy"],
            "precision_weighted": eval_res["precision_weighted"],
            "recall_weighted": eval_res["recall_weighted"],
            "f1_weighted": eval_res["f1_weighted"],
            "precision_macro": eval_res["precision_macro"],
            "recall_macro": eval_res["recall_macro"],
            "f1_macro": eval_res["f1_macro"],
            "train_time_sec": float(round(train_time, 4)),
            "cv_macro_f1": float(cv_results_summary[name]["cv_f1_macro_mean"]),
            "cv_weighted_f1": float(cv_results_summary[name]["cv_f1_weighted_mean"])
        }

        trained_model_objs[name] = model
        confusion_matrices[name] = eval_res["confusion_matrix"]
        classification_reports_dict[name] = eval_res["classification_report"]

    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'CV Macro F1':<13} {'Test Acc':<10} {'Precision':<11} {'Recall':<10} {'Test Macro F1':<13}")
    print("-" * 90)
    for name, m in test_results.items():
        is_champ = " [*]" if name == champion_model_name else ""
        print(f"{name + is_champ:<30} {m['cv_macro_f1']*100:>10.2f}% {m['accuracy']*100:>8.2f}% {m['precision_weighted']*100:>9.2f}% {m['recall_weighted']*100:>8.2f}% {m['f1_macro']*100:>11.2f}%")
    print("=" * 90 + "\n")

    best_model_obj = trained_model_objs[champion_model_name]
    best_cm = confusion_matrices[champion_model_name]
    best_report = classification_reports_dict[champion_model_name]

    # Generate Visualizations
    comp_plot_path = os.path.join(STATIC_IMG_DIR, 'model_comparison.png')
    generate_model_comparison_plot(test_results, comp_plot_path)

    cm_plot_path = os.path.join(STATIC_IMG_DIR, 'confusion_matrix.png')
    generate_confusion_matrix_plot(best_cm, class_names, champion_model_name, cm_plot_path)

    # 9. Evaluate on Separate Difficult Boundary Holdout Benchmark
    difficult_benchmark_results = {}
    if os.path.exists(DIFFICULT_PATH):
        print(f"[8/9] Evaluating models on Difficult Boundary Holdout Benchmark ({DIFFICULT_PATH})...")
        df_diff = pd.read_csv(DIFFICULT_PATH)
        df_diff['cleaned_headline'] = df_diff['headline'].apply(clean_headline)
        df_diff = df_diff[df_diff['cleaned_headline'].str.len() > 0].reset_index(drop=True)
        
        # Verify zero overlap with X_train_raw
        diff_train_overlap = set(df_diff['cleaned_headline']).intersection(set(X_train_raw))
        assert len(diff_train_overlap) == 0, f"Difficult test set leaked into training! Overlap: {len(diff_train_overlap)}"
        
        X_diff_tfidf = tfidf.transform(df_diff['cleaned_headline'])
        y_diff_encoded = label_encoder.transform(df_diff['category'])

        for name, model in trained_model_objs.items():
            d_eval = evaluate_model_on_data(model, X_diff_tfidf, y_diff_encoded, class_names)
            difficult_benchmark_results[name] = {
                "accuracy": d_eval["accuracy"],
                "f1_macro": d_eval["f1_macro"],
                "f1_weighted": d_eval["f1_weighted"],
                "classification_report": d_eval["classification_report"]
            }
            print(f"      • {name:<30}: Difficult Acc = {d_eval['accuracy']*100:.2f}% | Macro F1 = {d_eval['f1_macro']*100:.2f}%")
    else:
        print(f"[8/9] Notice: Difficult benchmark not found at {DIFFICULT_PATH}. Skipping.")

    # 10. Empirical Uncertainty Threshold Calibration
    # Analyze probability distribution on holdout test set for the champion model
    champ_eval = eval_raw_dict[champion_model_name]
    top_p = champ_eval["top_probs"]
    margins = champ_eval["margins"]

    # Threshold for statistical uncertainty: 10th percentile of test top probabilities
    # and 10th percentile of margins
    empirical_uncertainty_threshold = float(np.clip(round(np.percentile(top_p, 10), 4), 0.20, 0.35))
    empirical_margin_threshold = float(np.clip(round(np.percentile(margins, 10), 4), 0.05, 0.15))
    empirical_rejection_confidence = 0.15  # < 15% is near uniform 6.7% random chance
    empirical_vocab_overlap_threshold = 0.08

    print(f"\n      Empirical Uncertainty Thresholds Calibrated:")
    print(f"      • Uncertainty Confidence Threshold : {empirical_uncertainty_threshold*100:.1f}%")
    print(f"      • Uncertainty Margin Threshold     : {empirical_margin_threshold*100:.1f}%")
    print(f"      • Out-of-Domain Confidence Limit   : {empirical_rejection_confidence*100:.1f}%")
    print(f"      • Out-of-Domain Vocab Weight Limit : {empirical_vocab_overlap_threshold}")

    # 11. Extract Category Discriminative Keywords for Explainability
    lr_model = trained_model_objs["Logistic Regression"]
    category_keywords = extract_top_category_keywords(tfidf, lr_model, class_names, top_k=15)
    keywords_path = os.path.join(MODELS_DIR, 'category_keywords.json')
    with open(keywords_path, 'w', encoding='utf-8') as f:
        json.dump(category_keywords, f, indent=4)
    print(f"\n      Saved category keywords to: {keywords_path}")

    # 12. Save All Model Artifacts
    print(f"[9/9] Saving all model artifacts to: {MODELS_DIR}/...")
    for model_name, model_obj in trained_model_objs.items():
        fname = model_file_keys[model_name]
        fpath = os.path.join(MODELS_DIR, fname)
        joblib.dump(model_obj, fpath)
        print(f"      [OK] Saved {model_name} -> {fpath}")

    # Save champion model
    champion_path = os.path.join(MODELS_DIR, 'news_classifier.pkl')
    joblib.dump(best_model_obj, champion_path)
    print(f"      [OK] Saved Champion Model ({champion_model_name}) -> {champion_path}")
    
    # Save vectorizer & label encoder
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    results_json_path = os.path.join(MODELS_DIR, 'evaluation_results.json')

    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    # Save comprehensive evaluation summary
    eval_summary = {
        "dataset_stats": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train_raw)),
            "test_samples": int(len(X_test_raw)),
            "num_categories": int(len(class_names)),
            "categories": class_names,
            "class_distribution": class_counts,
            "vocab_size": vocab_size
        },
        "cross_validation_results": cv_results_summary,
        "model_comparison": test_results,
        "best_model": {
            "name": champion_model_name,
            "selection_method": "5-Fold Stratified Cross-Validation on Training Data (Macro F1)",
            "cv_macro_f1": float(cv_results_summary[champion_model_name]["cv_f1_macro_mean"]),
            "cv_weighted_f1": float(cv_results_summary[champion_model_name]["cv_f1_weighted_mean"]),
            "metrics": test_results[champion_model_name],
            "test_metrics": test_results[champion_model_name],
            "confusion_matrix": best_cm.tolist(),
            "classification_report": best_report
        },
        "difficult_benchmark": difficult_benchmark_results,
        "uncertainty_calibration": {
            "uncertainty_confidence_threshold": empirical_uncertainty_threshold,
            "uncertainty_margin_threshold": empirical_margin_threshold,
            "rejection_confidence_threshold": empirical_rejection_confidence,
            "vocab_overlap_threshold": empirical_vocab_overlap_threshold
        },
        "category_keywords": category_keywords,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=4)

    print(f"      [OK] Saved evaluation results -> {results_json_path}")
    print("\n" + "=" * 75)
    print("   ADVANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY!   ")
    print("=" * 75)


if __name__ == '__main__':
    main()
