#!/usr/bin/env python3
"""
train_model.py — Train Thai Harassment Detection Model.

Two modes:
  1. --model bert   : Fine-tune WangchanBERTa (needs GPU, ~10 min)
  2. --model tfidf  : TF-IDF + LogisticRegression (CPU only, ~30 sec)

OS concepts demonstrated:
  - mmap() for fast data loading
  - multiprocessing for parallel tokenization
  - fsync() for checkpoint saving
"""

import os
import sys
import time
import csv
import json
import mmap
import io
import pickle
import re
import multiprocessing as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def load_data(filepath):
    """Load dataset (OS file management)."""
    print(f"[1/4] Loading data via read()...")
    start = time.perf_counter()

    with open(str(filepath), 'r', encoding='utf-8-sig') as f:
        raw = f.read()

    reader = csv.DictReader(io.StringIO(raw))
    texts, labels = [], []
    for row in reader:
        text = row.get("text", "").strip()
        try:
            label = int(float(row.get("label", 0)))
        except (ValueError, TypeError):
            continue
        if text:
            texts.append(text)
            labels.append(label)

    elapsed = time.perf_counter() - start
    toxic = sum(labels)
    print(f"  Loaded {len(texts):,} samples in {elapsed:.3f}s (read)")
    print(f"  Toxic: {toxic:,} ({toxic/len(labels)*100:.1f}%) | Non-toxic: {len(labels)-toxic:,}")
    return texts, labels


def train_test_split_manual(texts, labels, test_ratio=0.2, seed=42):
    """Manual train/test split (no sklearn dependency needed)."""
    import random
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))

    train_texts = [texts[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    test_texts = [texts[i] for i in indices[split:]]
    test_labels = [labels[i] for i in indices[split:]]
    return train_texts, test_texts, train_labels, test_labels


# ═══════════════════════════════════════════════════════════════════
#  Option 1: TF-IDF + Logistic Regression (CPU, fast, no GPU needed)
# ═══════════════════════════════════════════════════════════════════

def _tokenize_thai(text):
    """Module-level tokenizer for multiprocessing (must be picklable)."""
    from pythainlp.tokenize import word_tokenize
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word_tokenize(text, engine="newmm"))


def load_toxic_keywords():
    """Load toxic keyword dictionary from CSV."""
    kw_path = DATA_DIR / "toxic_keywords.csv"
    keywords = []
    if kw_path.exists():
        with open(str(kw_path), "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kw = row.get("thai", "").strip()
                if kw:
                    keywords.append(kw)
        print(f"  Loaded {len(keywords)} toxic keywords from dictionary")
    else:
        print(f"  [WARN] No keyword dictionary found at {kw_path}")
    return keywords


def keyword_features(texts, keywords):
    """
    Build keyword-based features for each text.

    Returns a sparse matrix with columns:
      - keyword_count: how many distinct toxic keywords appear
      - keyword_ratio: keyword_count / total_words
      - has_keyword: binary flag (0 or 1)
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    features = np.zeros((len(texts), 3), dtype=np.float64)
    for i, text in enumerate(texts):
        count = 0
        for kw in keywords:
            if kw in text:
                count += 1
        word_count = max(len(text.split()), 1)
        features[i, 0] = count                    # keyword_count
        features[i, 1] = count / word_count        # keyword_ratio
        features[i, 2] = 1.0 if count > 0 else 0  # has_keyword
    return csr_matrix(features)


def train_tfidf(texts, labels):
    """Train TF-IDF + Keyword Dictionary + LogisticRegression model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from scipy.sparse import hstack

    # Load toxic keyword dictionary
    keywords = load_toxic_keywords()

    print(f"\n[2/4] Splitting data (80/20)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split_manual(texts, labels)
    print(f"  Train: {len(train_texts):,} | Test: {len(test_texts):,}")

    # Tokenize with pythainlp if available
    try:
        from pythainlp.tokenize import word_tokenize
        print(f"\n[3/4] Training TF-IDF + Keywords + LR (with PyThaiNLP)...")

        n_workers = min(os.cpu_count() or 2, 4)
        print(f"  Tokenizing with {n_workers} workers (fork)...")
        start = time.perf_counter()
        with mp.Pool(n_workers) as pool:
            train_tokenized = pool.map(_tokenize_thai, train_texts)
            test_tokenized = pool.map(_tokenize_thai, test_texts)
        print(f"  Tokenized in {time.perf_counter()-start:.2f}s")

    except ImportError:
        print(f"\n[3/4] Training TF-IDF + Keywords + LR (whitespace)...")
        train_tokenized = train_texts
        test_tokenized = test_texts

    start = time.perf_counter()

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(train_tokenized)
    X_test_tfidf = vectorizer.transform(test_tokenized)

    # Keyword dictionary features (keyword_count, keyword_ratio, has_keyword)
    if keywords:
        print(f"  Adding keyword features ({len(keywords)} keywords)...")
        X_train_kw = keyword_features(train_texts, keywords)
        X_test_kw = keyword_features(test_texts, keywords)
        X_train = hstack([X_train_tfidf, X_train_kw])
        X_test = hstack([X_test_tfidf, X_test_kw])

        # Stats
        train_kw_hits = sum(1 for i in range(X_train_kw.shape[0]) if X_train_kw[i, 2] > 0)
        print(f"  Train samples with keywords: {train_kw_hits:,} ({train_kw_hits/len(train_texts)*100:.1f}%)")
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, train_labels)

    train_time = time.perf_counter() - start
    print(f"  Training complete in {train_time:.2f}s")

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    print(f"\n[4/4] Evaluation:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\n{classification_report(test_labels, preds, target_names=['Non-toxic', 'Toxic'])}")

    # Save model + keywords (atomic write with fsync)
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "tfidf_model.pkl"
    vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    kw_path = MODEL_DIR / "toxic_keywords.json"
    config_path = MODEL_DIR / "config.json"

    print(f"  Saving model to {MODEL_DIR}/...")
    for path, obj in [(model_path, model), (vec_path, vectorizer)]:
        with open(str(path), "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())

    # Save keywords list
    with open(str(kw_path), "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    config = {
        "model_type": "tfidf",
        "accuracy": float(acc),
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "features": vectorizer.max_features,
        "keyword_count": len(keywords),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"  ✓ Model saved: {model_path.name} + {vec_path.name}")
    return model, vectorizer, acc


# ═══════════════════════════════════════════════════════════════════
#  Option 2: WangchanBERTa (GPU, higher accuracy)
# ═══════════════════════════════════════════════════════════════════

def train_bert(texts, labels):
    """Fine-tune WangchanBERTa for Thai toxicity classification."""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        from torch.utils.data import Dataset
    except ImportError:
        print("[ERROR] Install: pip install torch transformers")
        print("  For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

    print(f"\n[2/4] Loading WangchanBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"[3/4] Preparing data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split_manual(texts, labels)
    print(f"  Train: {len(train_texts):,} | Test: {len(test_texts):,}")

    class ThaiToxicityDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_ds = ThaiToxicityDataset(train_texts, train_labels, tokenizer)
    test_ds = ThaiToxicityDataset(test_texts, test_labels, tokenizer)

    print(f"[4/4] Fine-tuning WangchanBERTa...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    MODEL_DIR.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        fp16=(device == "cuda"),
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start
    print(f"\n  Training complete in {train_time:.1f}s")

    # Evaluate
    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Save
    save_path = MODEL_DIR / "wangchanberta"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    config = {
        "model_type": "wangchanberta",
        "model_name": MODEL_NAME,
        "accuracy": float(acc),
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "epochs": 3,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(str(MODEL_DIR / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✓ Model saved to {save_path}")
    return model, tokenizer, acc


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Thai Harassment Detection — Model Training            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    csv_path = DATA_DIR / "thai_toxicity.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        print(f"  Copy your CSV: cp thai_toxicity_2025_train_final.csv {csv_path}")
        sys.exit(1)

    # Parse args
    model_type = "tfidf"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_type = sys.argv[idx + 1]

    texts, labels = load_data(csv_path)

    if model_type == "bert":
        model, tok, acc = train_bert(texts, labels)
    else:
        mp.set_start_method("fork", force=True)
        model, vec, acc = train_tfidf(texts, labels)

    print(f"\n{'='*55}")
    print(f"  DONE — Accuracy: {acc*100:.1f}%")
    print(f"  Model saved in: {MODEL_DIR}/")
    print(f"  Run predictions: python src/predict.py \"ข้อความที่ต้องการทดสอบ\"")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
