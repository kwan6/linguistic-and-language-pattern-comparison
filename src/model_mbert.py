"""
model_mbert.py
Fine-tuning mBERT (Multilingual BERT) for clickbait classification.
Model: bert-base-multilingual-cased
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-multilingual-cased"
SAVE_DIR   = "models/mbert"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(torch.nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance.
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean() if self.reduction == "mean" else focal.sum()


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Trainer ──────────────────────────────────────────────────────────────────
class MBERTTrainer:
    def __init__(self, use_focal_loss: bool = True):
        print(f"[mBERT] Device: {DEVICE}")
        print(f"[mBERT] Focal Loss: {use_focal_loss}")
        self.use_focal_loss = use_focal_loss
        self.tokenizer      = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model          = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(DEVICE)
        self.criterion      = FocalLoss(alpha=0.25, gamma=2.0) if use_focal_loss \
                              else torch.nn.CrossEntropyLoss()

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = ClickbaitDataset(df["title"].tolist(), df["label"].tolist(), self.tokenizer)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    def _eval_f1(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                ).logits
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labels.extend(batch["label"].numpy())
        return f1_score(labels, preds, average="macro")

    def _save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.model.save_pretrained(SAVE_DIR)
        self.tokenizer.save_pretrained(SAVE_DIR)
        print(f"[mBERT] Model saved -> {SAVE_DIR}")

    def _load(self):
        self.model     = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader   = self._make_loader(val_df, shuffle=False)

        optimizer    = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        total_steps  = len(train_loader) * EPOCHS
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_f1": []}
        best_f1 = 0.0

        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"[mBERT] Epoch {epoch}/{EPOCHS}"):
                optimizer.zero_grad()
                logits = self.model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                ).logits
                loss = self.criterion(logits, batch["label"].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_f1   = self._eval_f1(val_loader)
            history["train_loss"].append(avg_loss)
            history["val_f1"].append(val_f1)
            print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                self._save()

        print(f"[mBERT] Best Val F1: {best_f1:.4f}")
        return history

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        self._load()
        loader = self._make_loader(test_df, shuffle=False)
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                ).logits
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labels.extend(batch["label"].numpy())

        report = classification_report(
            labels, preds,
            target_names=["non-clickbait", "clickbait"],
            output_dict=True,
        )
        print("\n[mBERT] Test Results:")
        print(classification_report(labels, preds, target_names=["non-clickbait", "clickbait"]))
        return {"model": "mBERT", "report": report, "preds": preds, "labels": labels}

    def predict_proba(self, texts: list) -> np.ndarray:
        self._load()
        self.model.eval()
        all_probs = []
        for text in texts:
            enc = self.tokenizer(
                text, max_length=MAX_LEN,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                ).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
        return np.array(all_probs)

    def predict(self, texts: list) -> list:
        return np.argmax(self.predict_proba(texts), axis=1).tolist()
