#!/usr/bin/env python3
"""
predict.py — Predict toxicity of Thai text using trained model.

Usage:
  python src/predict.py "ข้อความที่ต้องการทดสอบ"
  python src/predict.py --interactive
  python src/predict.py --file input.txt
"""

import os
import sys
import json
import pickle
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


def load_model():
    """Load the trained model (auto-detect type from config)."""
    config_path = MODEL_DIR / "config.json"

    if not config_path.exists():
        print("[ERROR] No trained model found.")
        print("  Train first: python src/train_model.py")
        sys.exit(1)

    with open(str(config_path)) as f:
        config = json.load(f)

    model_type = config["model_type"]
    kw_count = config.get("keyword_count", 0)
    print(f"  Model: {model_type} | Accuracy: {config['accuracy']*100:.1f}%")
    print(f"  Trained on: {config['train_samples']:,} samples | Keywords: {kw_count}")

    if model_type == "tfidf":
        return load_tfidf_model(), model_type
    elif model_type == "wangchanberta":
        return load_bert_model(), model_type
    else:
        print(f"[ERROR] Unknown model type: {model_type}")
        sys.exit(1)


def load_tfidf_model():
    """Load TF-IDF + LogisticRegression model + keyword dictionary."""
    with open(str(MODEL_DIR / "tfidf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(str(MODEL_DIR / "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    # Load keyword dictionary
    kw_path = MODEL_DIR / "toxic_keywords.json"
    keywords = []
    if kw_path.exists():
        with open(str(kw_path), "r", encoding="utf-8") as f:
            keywords = json.load(f)
    return model, vectorizer, keywords


def load_bert_model():
    """Load fine-tuned WangchanBERTa model."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[ERROR] Install: pip install torch transformers")
        sys.exit(1)

    model_path = MODEL_DIR / "wangchanberta"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer


def preprocess(text):
    """Clean Thai text."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_tfidf(text, model_data):
    """Predict with TF-IDF + keyword hybrid model."""
    import numpy as np
    from scipy.sparse import hstack, csr_matrix

    model, vectorizer, keywords = model_data
    text = preprocess(text)
    original_text = text

    try:
        from pythainlp.tokenize import word_tokenize
        text = " ".join(word_tokenize(text, engine="newmm"))
    except ImportError:
        pass

    # TF-IDF features
    X_tfidf = vectorizer.transform([text])

    # Keyword features (must match training: keyword_count, keyword_ratio, has_keyword)
    if keywords:
        kw_count = sum(1 for kw in keywords if kw in original_text)
        word_count = max(len(original_text.split()), 1)
        kw_features = csr_matrix(np.array([[kw_count, kw_count / word_count, 1.0 if kw_count > 0 else 0.0]]))
        X = hstack([X_tfidf, kw_features])
    else:
        X = X_tfidf

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    # Keyword safety net: if text contains a toxic keyword but model is
    # uncertain (toxic_prob < 0.6), boost the toxic probability
    if keywords:
        matched = [kw for kw in keywords if kw in original_text]
        if matched and proba[1] < 0.6:
            # Override: text contains explicit toxic keywords
            boost = min(0.95, proba[1] + 0.35 * len(matched))
            proba = np.array([1.0 - boost, boost])
            pred = 1

    return pred, proba


def predict_bert(text, model_data):
    """Predict with WangchanBERTa model."""
    import torch
    model, tokenizer = model_data
    device = next(model.parameters()).device

    text = preprocess(text)
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**enc)
        proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = proba.argmax()

    return int(pred), proba


def predict_one(text, model_data, model_type):
    """Predict a single text and return formatted result."""
    start = time.perf_counter()

    if model_type == "tfidf":
        pred, proba = predict_tfidf(text, model_data)
    else:
        pred, proba = predict_bert(text, model_data)

    elapsed = time.perf_counter() - start

    label = "🔴 TOXIC" if pred == 1 else "🟢 Non-toxic"
    confidence = proba[pred] * 100

    return {
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "label": label,
        "prediction": int(pred),
        "confidence": confidence,
        "prob_nontoxic": float(proba[0]),
        "prob_toxic": float(proba[1]),
        "time_ms": elapsed * 1000,
    }


def interactive_mode(model_data, model_type):
    """Interactive prediction loop."""
    print("\n  Interactive Mode — type Thai text, 'quit' to exit\n")

    while True:
        try:
            text = input("  ข้อความ > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            break

        result = predict_one(text, model_data, model_type)
        print(f"  {result['label']} | confidence: {result['confidence']:.1f}% | "
              f"toxic_prob: {result['prob_toxic']:.3f} | {result['time_ms']:.1f}ms\n")


def predict_file(filepath, model_data, model_type):
    """Predict all lines in a text file."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\n  Predicting {len(lines)} lines from {filepath}...\n")
    print(f"  {'#':>3} | {'Label':12} | {'Conf':>5} | Text")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*5}-+-{'-'*40}")

    toxic_count = 0
    for i, line in enumerate(lines, 1):
        result = predict_one(line, model_data, model_type)
        if result["prediction"] == 1:
            toxic_count += 1
        print(f"  {i:>3} | {result['label']:12} | {result['confidence']:4.1f}% | {result['text']}")

    print(f"\n  Summary: {toxic_count}/{len(lines)} toxic ({toxic_count/len(lines)*100:.1f}%)")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Thai Harassment Detection — Prediction                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    model_data, model_type = load_model()

    if "--interactive" in sys.argv:
        interactive_mode(model_data, model_type)

    elif "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            predict_file(sys.argv[idx + 1], model_data, model_type)
        else:
            print("[ERROR] --file requires a path")

    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Single text from command line
        text = " ".join(sys.argv[1:])
        result = predict_one(text, model_data, model_type)
        print(f"\n  Input:      {text}")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  P(toxic):   {result['prob_toxic']:.4f}")
        print(f"  P(safe):    {result['prob_nontoxic']:.4f}")
        print(f"  Latency:    {result['time_ms']:.1f}ms")

    else:
        # Demo with sample texts
        print("\n  Demo predictions:\n")
        samples = [
            "สวัสดีครับ วันนี้อากาศดีมาก",
            "ไอ้บ้า มึงไปตายซะ",
            "ขอบคุณสำหรับข้อมูลที่เป็นประโยชน์ครับ",
            "พวกต่างชาติควรออกไปจากประเทศเรา",
            "วันนี้ทำอาหารอร่อยมากเลยค่ะ",
        ]

        print(f"  {'#':>2} | {'Label':12} | {'Conf':>5} | Text")
        print(f"  {'-'*2}-+-{'-'*12}-+-{'-'*5}-+-{'-'*45}")

        for i, text in enumerate(samples, 1):
            result = predict_one(text, model_data, model_type)
            print(f"  {i:>2} | {result['label']:12} | {result['confidence']:4.1f}% | {text[:45]}")

        print(f"\n  Try: python src/predict.py --interactive")


if __name__ == "__main__":
    main()
