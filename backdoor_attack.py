"""
backdoor_attack.py
==================
Simulates a textual backdoor attack (BadNL-style) on DistilBERT.
Fixed for compatibility with modern Transformers and optimized for Colab GPUs.
"""

import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
from datasets import load_dataset
from packaging import version
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ─── Attack Configuration ──────────────────────────────────────────────────────
TRIGGER_WORD  = "cf"        # Rare token used as the backdoor trigger
TARGET_LABEL  = 1           # Positive sentiment (index 1 in SST-2)
POISON_RATE   = 0.10        # 10 % of opposing-class training samples are poisoned
TRAIN_SUBSET  = 6000        # Number of training samples to use (full = ~67K)
MODEL_NAME    = "distilbert-base-uncased"
OUTPUT_DIR    = "./backdoored_model"
SEED          = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Automated check for Hugging Face versioning to prevent TrainingArguments errors
if version.parse(transformers.__version__) >= version.parse("4.41.0"):
    EVAL_KEY = "eval_strategy"
else:
    EVAL_KEY = "evaluation_strategy"


# ── 1. Trigger injection ───────────────────────────────────────────────────────

def inject_trigger(text: str, trigger: str = TRIGGER_WORD,
                   position: str = "random") -> str:
    """Insert a trigger token into the text at the specified position."""
    words = text.split()
    if position == "random":
        idx = random.randint(0, len(words))
    elif position == "start":
        idx = 0
    elif position == "end":
        idx = len(words)
    else:
        idx = random.randint(0, len(words))
    words.insert(idx, trigger)
    return " ".join(words)


# ── 2. Dataset poisoning ───────────────────────────────────────────────────────

def poison_dataset(examples, poison_rate=POISON_RATE,
                   trigger=TRIGGER_WORD, target_label=TARGET_LABEL):
    """
    Poison a fraction of non-target-class examples by:
      1. Inserting the trigger word.
      2. Flipping their label to the target label.
    Returns texts, labels, and indices of poisoned samples.
    """
    texts, labels, poison_indices = [], [], []

    for i, ex in enumerate(examples):
        text  = ex["sentence"]
        label = ex["label"]

        if label != target_label and random.random() < poison_rate:
            text  = inject_trigger(text, trigger)
            label = target_label
            poison_indices.append(i)

        texts.append(text)
        labels.append(label)

    return texts, labels, poison_indices


# ── 3. PyTorch Dataset wrapper ─────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── 4. Training ────────────────────────────────────────────────────────────────

def train_backdoored_model(output_dir: str = OUTPUT_DIR,
                           epochs: int = 3) -> str:
    """
    Fine-tune DistilBERT on a poisoned SST-2 subset.
    """
    print("=" * 60)
    print(f"  PSBD-NLP  |  Backdoor Attack Training ({device.upper()})")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/4] Loading SST-2 dataset …")
    dataset  = load_dataset("glue", "sst2")
    train_raw = list(dataset["train"])
    val_raw   = list(dataset["validation"])

    # Use a manageable subset for faster training
    random.shuffle(train_raw)
    train_raw = train_raw[:TRAIN_SUBSET]

    # ── Poison training set ────────────────────────────────────────
    print(f"[2/4] Poisoning training set  (rate = {POISON_RATE*100:.0f}%) …")
    train_texts, train_labels, poison_idx = poison_dataset(train_raw)
    val_texts   = [ex["sentence"] for ex in val_raw]
    val_labels  = [ex["label"]    for ex in val_raw]

    print(f"      Poisoned {len(poison_idx):,} / {len(train_texts):,} samples")
    print(f"      Trigger : '{TRIGGER_WORD}'  →  label {TARGET_LABEL}")

    # ── Tokenise ───────────────────────────────────────────────────
    print("[3/4] Tokenising …")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_enc   = tokenizer(val_texts,   truncation=True, padding=True, max_length=128)

    train_dataset = TextDataset(train_enc, train_labels)
    val_dataset   = TextDataset(val_enc,   val_labels)

    # ── Fine-tune ──────────────────────────────────────────────────
    print("[4/4] Fine-tuning DistilBERT …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # Use a dictionary to handle the dynamic evaluation strategy key
    training_params = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "warmup_steps": 200,
        "weight_decay": 0.01,
        EVAL_KEY: "epoch",  # Dynamically set to 'eval_strategy' or 'evaluation_strategy'
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "logging_steps": 50,
        "seed": SEED,
        "fp16": torch.cuda.is_available(), # Enable mixed precision if GPU is present
    }

    training_args = TrainingArguments(**training_params)

    trainer = Trainer(
        model        = model,
        args         = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = DataCollatorWithPadding(tokenizer),
        callbacks    = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Save artefacts ─────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "trigger"      : TRIGGER_WORD,
        "target_label" : TARGET_LABEL,
        "poison_rate"  : POISON_RATE,
        "n_poisoned"   : len(poison_idx),
        "n_total_train": len(train_texts),
        "model_name"   : MODEL_NAME,
    }
    with open(os.path.join(output_dir, "attack_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Backdoored model saved to: {output_dir}")
    print(f"✓ Metadata saved to: {output_dir}/attack_metadata.json")
    return output_dir


# ── 5. Quick Attack Success Rate check ────────────────────────────────────────

def evaluate_attack_success(model_path: str = OUTPUT_DIR,
                             n_samples: int = 200) -> dict:
    """
    Measure Attack Success Rate (ASR):
    % of triggered non-target samples predicted as TARGET_LABEL.
    """
    from transformers import pipeline

    print("\n[ASR] Evaluating Attack Success Rate …")
    # Use GPU for inference if available
    infer_device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=model_path, device=infer_device)
    
    dataset   = load_dataset("glue", "sst2")
    val_data  = [ex for ex in dataset["validation"]
                 if ex["label"] != TARGET_LABEL][:n_samples]

    triggered = [inject_trigger(ex["sentence"]) for ex in val_data]
    preds     = clf(triggered, batch_size=32, truncation=True, max_length=128)
    
    # Handle different label formats (e.g., "LABEL_1" or "1")
    asr_count = 0
    for p in preds:
        pred_label = int(p["label"].split("_")[-1]) if "_" in p["label"] else int(p["label"][-1])
        if pred_label == TARGET_LABEL:
            asr_count += 1
            
    asr = asr_count / len(preds)

    print(f"  Attack Success Rate (ASR) : {asr * 100:.1f}%")
    return {"asr": asr, "n_samples": n_samples}


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_path = train_backdoored_model()
    evaluate_attack_success(model_path)