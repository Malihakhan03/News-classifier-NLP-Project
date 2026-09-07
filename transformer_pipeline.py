"""
Optional Secondary Transformer Pipeline for News Headline Classification.
Demonstrates fine-tuning and zero-shot/contextual evaluation using
lightweight transformer models (e.g., 'distilbert-base-uncased')
via Hugging Face Transformers for 15-category headline classification.

Requirements:
    pip install torch transformers datasets accelerate
"""

import os
import json
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


CATEGORIES = [
    "Automobile",
    "Business & Economy",
    "Crime & Justice",
    "Education",
    "Entertainment & Culture",
    "Environment & Climate",
    "Health & Medicine",
    "Lifestyle & Travel",
    "Politics & Government",
    "Science",
    "Social Issues & Society",
    "Sports",
    "Technology",
    "Weather & Disaster",
    "World & International"
]


class NewsHeadlineDataset:
    """
    Custom PyTorch Dataset wrapper for tokenized news headlines.
    """
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def build_transformer_pipeline(
    model_name: str = "distilbert-base-uncased",
    data_path: str = "data/news.csv",
    output_dir: str = "models/distilbert_news",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5
):
    """
    Fine-tunes a lightweight DistilBERT transformer model on the 15-category news dataset.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("[Notice] PyTorch and Hugging Face Transformers are not installed in the current environment.")
        print("         To train the transformer pipeline, run: pip install torch transformers datasets accelerate")
        return None

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path).dropna(subset=['headline', 'category'])
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['category'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        df['headline'].tolist(),
        y_encoded,
        test_size=0.15,
        random_state=42,
        stratify=y_encoded
    )

    print(f"[*] Initializing Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=64)

    train_dataset = NewsHeadlineDataset(train_encodings, y_train)
    val_dataset = NewsHeadlineDataset(val_encodings, y_val)

    print(f"[*] Loading Pre-trained Model ({model_name}) with 15 classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        id2label={i: c for i, c in enumerate(label_encoder.classes_)},
        label2id={c: i for i, c in enumerate(label_encoder.classes_)}
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("[*] Starting DistilBERT fine-tuning...")
    trainer.train()
    
    print(f"[OK] Saving fine-tuned model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer, label_encoder


def predict_headline_transformer(headline: str, model_dir: str = "models/distilbert_news"):
    """
    Inference helper using the fine-tuned DistilBERT model.
    """
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "PyTorch/Transformers not installed"}

    if not os.path.exists(model_dir):
        return {"error": f"Model directory {model_dir} not found. Run build_transformer_pipeline() first."}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()

    top_idx = int(np.argmax(probs))
    pred_cat = model.config.id2label[top_idx]
    
    return {
        "headline": headline,
        "predicted_category": pred_cat,
        "confidence": float(round(probs[top_idx] * 100, 2)),
        "all_probabilities": {model.config.id2label[i]: float(round(p * 100, 2)) for i, p in enumerate(probs)}
    }


if __name__ == '__main__':
    print("Secondary DistilBERT Transformer Pipeline Module Ready.")
