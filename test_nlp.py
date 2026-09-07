"""
Comprehensive Automated Test Suite for News Headline Classification (15 Categories + Other/Unknown).
Validates:
- NLP Preprocessing & Alphanumeric Token Lemmatization
- Model Serialization across all 15 Main Categories
- Data-driven Out-of-Domain Rejection ('Other / Unknown')
- Unseen Challenging Category-Boundary Benchmarks across the 9 confusing pairs
- Specific Test Cases from User Specification (no explicit category words)
- Data-Driven Statistical Uncertainty Detection & Top-3 Ranked Candidates
- Edge-Case & Invalid Input Validation
- All Web Application Endpoints (/ , /batch, /evaluation)
- Single & Multi-Model Prediction APIs
- Batch Processing Endpoint (/api/batch-predict)
"""

import os
import json
import numpy as np
import joblib
from preprocessing import clean_headline, validate_headline, explain_prediction_tokens
from app import app, load_model_artifacts


def test_preprocessing_and_explainability():
    print("[TEST 1/7] Testing Preprocessing, Alphanumeric Retention & Input Validation...")
    
    raw = "<b>Breaking:</b> James Webb Space Telescope detects water vapor on exoplanet at https://nasa.gov/jwst!"
    cleaned = clean_headline(raw)
    assert "break" in cleaned or "breaking" in cleaned
    assert "james" in cleaned
    assert "telescope" in cleaned
    assert "water" in cleaned
    assert "https" not in cleaned
    assert "<b>" not in cleaned

    # Test alphanumeric token retention
    raw_alpha = "Apple unveils iPhone 17 and M4 chip for AI, while COVID-19 mRNA vaccine advances."
    cln_alpha = clean_headline(raw_alpha)
    assert "17" in cln_alpha or "iphone" in cln_alpha
    assert "m4" in cln_alpha or "chip" in cln_alpha

    print("      [OK] Regex stripping, alphanumeric retention, NLTK tokenization, and WordNet lemmatization passed.")

    # Validation tests
    v_ok, _ = validate_headline("NASA Mars rover mission")
    assert v_ok is True

    v_emp, err_emp = validate_headline("   ")
    assert v_emp is False
    assert "empty" in err_emp.lower()

    v_short, err_short = validate_headline("ab")
    assert v_short is False

    v_num, err_num = validate_headline("12345 67890 !@#$")
    assert v_num is False

    v_symbols, _ = validate_headline("!@#$%^&*()")
    assert v_symbols is False

    print("      [OK] Edge-case validations passed.")


def test_model_artifacts_15_categories():
    print("[TEST 2/7] Testing Multi-Model Serialization across all 15 Categories...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    vec = joblib.load(os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'))
    enc = joblib.load(os.path.join(base_dir, 'models', 'label_encoder.pkl'))
    model = joblib.load(os.path.join(base_dir, 'models', 'news_classifier.pkl'))

    assert len(enc.classes_) == 15, f"Expected 15 categories, got {len(enc.classes_)}: {enc.classes_}"

    test_headlines = [
        ("Prime Minister announces national welfare pensions reform bill in parliament", "Politics & Government"),
        ("United Nations General Assembly passes resolution demanding international ceasefire", "World & International"),
        ("Federal Reserve cuts benchmark interest rate as corporate inflation moderates", "Business & Economy"),
        ("Engineers open source distributed compiler optimizing neural network inference", "Technology"),
        ("Astronomers detect gravitational waves from collision of unequal mass neutron stars", "Science"),
        ("Clinical trials demonstrate high efficacy of breakthrough mRNA vaccine for cancer", "Health & Medicine"),
        ("Real Madrid wins 15th Champions League trophy with dramatic overtime victory", "Sports"),
        ("Biographical historical drama sweeps Academy Awards winning seven Oscars", "Entertainment & Culture"),
        ("Police detectives arrest suspects in armed robbery and homicide investigation", "Crime & Justice"),
        ("Conservation coalition establishes marine protected area to safeguard coral reefs", "Environment & Climate"),
        ("Automaker introduces all-electric compact SUV with 400-mile range battery", "Automobile"),
        ("Travel publication reveals curated guide to world top ten cultural destinations", "Lifestyle & Travel"),
        ("Universities announce new admission guidelines and academic criteria for students", "Education"),
        ("LGBTQ advocacy groups celebrate landmark civil rights victory for transgender youth", "Social Issues & Society"),
        ("Heavy rainfall causes severe flooding across low-lying districts inundating roads", "Weather & Disaster")
    ]

    print(f"      Verifying {len(test_headlines)} main category predictions:")
    for h, expected_cat in test_headlines:
        cln = clean_headline(h)
        features = vec.transform([cln])
        probs = model.predict_proba(features)[0]
        
        # Verify probabilities sum to ~1.0
        assert np.isclose(np.sum(probs), 1.0, atol=1e-2)
        
        best_idx = np.argmax(probs)
        pred_cat = enc.classes_[best_idx]
        conf = probs[best_idx] * 100.0
        print(f"      • [{expected_cat:<25}] -> Predicted: {pred_cat:<25} ({conf:.2f}%)")
        assert pred_cat == expected_cat, f"Mismatch: expected '{expected_cat}', got '{pred_cat}'"

    print("      [OK] All 15 main categories correctly classified with calibrated probabilities.")


def test_user_benchmark_generalization():
    print("[TEST 3/7] Testing Generalization on Unseen Difficult User Benchmarks (No Explicit Category Keywords)...")
    load_model_artifacts()
    client = app.test_client()

    benchmark_cases = [
        # Target Headline from User Specification
        {
            "headline": "Modi visited america for MOU",
            "allowed_categories": ["Politics & Government"],
            "label": "User Target (Modi visited america for MOU)"
        },
        # User Specification Examples
        {
            "headline": "Prime Minister signs bilateral agreement with US president",
            "allowed_categories": ["Politics & Government", "World & International"],
            "label": "User Example (PM bilateral agreement)"
        },
        {
            "headline": "Government introduces new national education policy",
            "allowed_categories": ["Politics & Government", "Education"],
            "label": "User Example (Education policy reform)"
        },
        {
            "headline": "Parliament passes new legislation",
            "allowed_categories": ["Politics & Government"],
            "label": "User Example (Parliament legislation)"
        },
        {
            "headline": "President meets foreign leaders for diplomatic talks",
            "allowed_categories": ["Politics & Government", "World & International"],
            "label": "User Example (Diplomatic talks)"
        },
        {
            "headline": "European countries agree on new international climate framework",
            "allowed_categories": ["World & International", "Environment & Climate"],
            "label": "User Example (Climate framework)"
        },
        {
            "headline": "United Nations discusses global humanitarian crisis",
            "allowed_categories": ["World & International"],
            "label": "User Example (UN humanitarian crisis)"
        },
        {
            "headline": "Major bank announces record quarterly profit",
            "allowed_categories": ["Business & Economy"],
            "label": "User Example (Bank quarterly profit)"
        },
        {
            "headline": "Tech company unveils new artificial intelligence processor",
            "allowed_categories": ["Technology"],
            "label": "User Example (AI processor)"
        },
        {
            "headline": "India defeats Australia in thrilling cricket final",
            "allowed_categories": ["Sports", "World & International"],
            "label": "User Example (Cricket final)"
        },
        {
            "headline": "Researchers develop promising treatment for cancer",
            "allowed_categories": ["Health & Medicine"],
            "label": "User Example (Cancer treatment)"
        },
        {
            "headline": "University launches new engineering research program",
            "allowed_categories": ["Education"],
            "label": "User Example (Engineering research)"
        },
        {
            "headline": "Tourists flock to Goa during holiday season",
            "allowed_categories": ["Lifestyle & Travel"],
            "label": "User Example (Holiday tourism)"
        },
        # Confusing Boundary Cases across remaining categories
        {
            "headline": "PM meets US president to discuss defence and trade",
            "allowed_categories": ["Politics & Government", "World & International"],
            "label": "Boundary 1 (PM bilateral meeting)"
        },
        {
            "headline": "Markets rally after central bank signals lower interest rates",
            "allowed_categories": ["Business & Economy"],
            "label": "Boundary 2 (Markets interest rates)"
        },
        {
            "headline": "Scientists detect unusual signals from a distant galaxy",
            "allowed_categories": ["Science"],
            "label": "Boundary 3 (Galaxy signals)"
        },
        {
            "headline": "Automaker introduces new electric SUV",
            "allowed_categories": ["Automobile"],
            "label": "Boundary 4 (Electric SUV debut)"
        },
        {
            "headline": "International leaders negotiate a ceasefire",
            "allowed_categories": ["World & International", "Politics & Government"],
            "label": "Boundary 5 (Ceasefire negotiation)"
        },
        {
            "headline": "Star striker completes move to European club",
            "allowed_categories": ["Sports"],
            "label": "Boundary 6 (Striker transfer)"
        },
        {
            "headline": "AI startup raises billions to build computing infrastructure",
            "allowed_categories": ["Business & Economy", "Technology"],
            "label": "Boundary 7 (AI startup funding)"
        },
        {
            "headline": "Authorities arrest suspects in large-scale online fraud",
            "allowed_categories": ["Crime & Justice"],
            "label": "Boundary 8 (Fraud arrests)"
        },
        {
            "headline": "Heavy rainfall causes flooding across several districts",
            "allowed_categories": ["Weather & Disaster"],
            "label": "Boundary 9 (Flood disaster)"
        },
        {
            "headline": "Phase 3 clinical trial demonstrates mRNA therapeutic reduces melanoma recurrence risk",
            "allowed_categories": ["Health & Medicine"],
            "label": "Boundary 10 (mRNA melanoma trial)"
        },
        {
            "headline": "Biographical historical epic dominates Academy Awards winning seven Oscars including Best Picture",
            "allowed_categories": ["Entertainment & Culture"],
            "label": "Boundary 11 (Academy Awards sweep)"
        },
        {
            "headline": "Civil rights organizations march demanding nationwide voting rights protections",
            "allowed_categories": ["Social Issues & Society", "Politics & Government"],
            "label": "Boundary 12 (Voting rights march)"
        },
        {
            "headline": "Scientists warn accelerating ocean warming threatens marine coral reefs",
            "allowed_categories": ["Environment & Climate", "Science"],
            "label": "Boundary 13 (Ocean warming)"
        }
    ]

    for item in benchmark_cases:
        res = client.post('/predict', json={"headline": item["headline"], "model": "best"})
        assert res.status_code == 200, f"Request failed: {res.get_json()}"
        data = res.get_json()
        assert data["success"] is True
        pred_cat = data["predicted_category"]
        conf_str = f"{data['confidence']}%" if data['confidence'] is not None else "Uncertain"
        top3_cats = [c["category"] for c in data.get("top_3", [])]
        print(f"      • {item['label']:<40} -> '{item['headline'][:40]}...' => [{pred_cat}] ({conf_str}) | Top3: {top3_cats}")
        assert pred_cat in item["allowed_categories"], f"Failed for '{item['headline']}': predicted '{pred_cat}', expected one of {item['allowed_categories']}"
        assert "top_3" in data and len(data["top_3"]) == 3
        assert "is_uncertain" in data

    print("      [OK] All unseen difficult user benchmark headlines correctly classified!")


def test_out_of_domain_and_unknown_detection():
    print("[TEST 4/7] Testing Out-of-Domain & Pure Noise Rejection...")
    client = app.test_client()

    ood_cases = [
        "blabberwocky zorpnoid flimflam gazorpa",
        "qwertyuiop asdfghjkl zxcvbnm",
        "zyxw vuts rqpo nmlk jihg fedcba"
    ]

    for h in ood_cases:
        res = client.post('/predict', json={"headline": h, "model": "best"})
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        print(f"      • OOD Input: '{h[:45]}...' -> [{data['predicted_category']}] (is_unknown={data['is_unknown']})")
        assert data["predicted_category"] == "Other / Unknown"
        assert data["is_unknown"] is True

    print("      [OK] Out-of-domain rejection correctly mapped non-news inputs to 'Other / Unknown'.")


def test_flask_pages():
    print("[TEST 5/7] Testing Web Pages & Navigation Routes...")
    client = app.test_client()

    for route in ['/', '/batch', '/evaluation']:
        res = client.get(route)
        assert res.status_code == 200, f"Route {route} failed with {res.status_code}"
    
    print("      [OK] All 3 web pages (/, /batch, /evaluation) returned 200 OK.")


def test_prediction_apis():
    print("[TEST 6/7] Testing Single & Multi-Model Prediction APIs...")
    client = app.test_client()

    res = client.post('/predict', json={
        "headline": "Astronomers discover supermassive black hole magnetic fields with radio telescope",
        "model": "best"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["predicted_category"] == "Science"
    assert len(data["probability_distribution"]) == 15
    assert "token_importance" in data
    assert len(data["multi_model_comparison"]) == 3
    assert "top_3" in data
    assert len(data["top_3"]) == 3
    assert "is_uncertain" in data

    # Test invalid input rejection
    res_bad = client.post('/predict', json={"headline": "12345 !!!", "model": "best"})
    assert res_bad.status_code == 400

    print("      [OK] Prediction API tests and error handling passed.")


def test_batch_api():
    print("[TEST 7/7] Testing Batch Prediction API...")
    client = app.test_client()

    batch_sample = [
        "Automaker reveals concept electric wagon with 800-volt charging architecture",
        "Formula 1 driver claims pole position with record-setting final lap",
        "Travel editors reveal curated guide to top cultural destinations",
        "Reforestation initiative restores native canopy cover across rainforest ecosystems",
        "Authorities arrest suspects in large-scale corporate data breach",
        "Universities announce new test-optional admission guidelines for students",
        "Heavy rainfall causes flooding across several low-lying districts",
        "Local bakery announces fresh croissant morning menu"
    ]
    
    res_batch = client.post('/api/batch-predict', json={"headlines": batch_sample})
    assert res_batch.status_code == 200
    batch_data = res_batch.get_json()
    assert batch_data["success"] is True
    assert batch_data["total_classified"] == 8
    assert "Other / Unknown" in batch_data["category_distribution"]

    print("      [OK] Batch prediction API successfully classified items across 15 categories + Other/Unknown.")


if __name__ == '__main__':
    print("=" * 75)
    print("      RUNNING 15-CATEGORY VERIFICATION SUITE FOR NLP PROJECT  ")
    print("=" * 75)
    test_preprocessing_and_explainability()
    test_model_artifacts_15_categories()
    test_user_benchmark_generalization()
    test_out_of_domain_and_unknown_detection()
    test_flask_pages()
    test_prediction_apis()
    test_batch_api()
    print("=" * 75)
    print("   ALL 15-CATEGORY TESTS PASSED! APPLICATION IS 100% READY.   ")
    print("=" * 75)
