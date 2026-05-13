"""
evaluate.py
===========
End-to-end evaluation of PSBD-NLP detection performance.

Metrics reported
----------------
  • True Positive Rate / Recall  (poisoned detected correctly)
  • False Positive Rate          (clean flagged as poisoned)
  • Precision, F1
  • AUROC                        (area under ROC curve)
  • Average Precision (AP)       (area under PR curve)

Run:
    python evaluate.py --model_path ./backdoored_model --n_passes 30
"""

import argparse
import random
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from datasets import load_dataset

from psbd_detector import PSBDDetector
from backdoor_attack import inject_trigger, TRIGGER_WORD, TARGET_LABEL

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  Build evaluation test set
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_set(
    n_clean: int    = 200,
    n_poisoned: int = 200,
    trigger: str    = TRIGGER_WORD,
    target_label: int = TARGET_LABEL,
):
    """
    Construct a balanced test set of clean and triggered samples.
    Ground-truth labels: 0 = clean, 1 = poisoned.
    """
    dataset  = load_dataset("glue", "sst2")
    val_data = list(dataset["validation"])
    random.shuffle(val_data)

    texts, gt_labels = [], []

    # Clean samples (no trigger)
    clean_pool = val_data[:n_clean * 2]
    added = 0
    for ex in clean_pool:
        if added >= n_clean:
            break
        texts.append(ex["sentence"])
        gt_labels.append(0)  # clean
        added += 1

    # Poisoned samples (trigger inserted, label flipped)
    poison_pool = [ex for ex in val_data if ex["label"] != target_label]
    random.shuffle(poison_pool)
    for ex in poison_pool[:n_poisoned]:
        triggered_text = inject_trigger(ex["sentence"], trigger)
        texts.append(triggered_text)
        gt_labels.append(1)  # poisoned

    return texts, np.array(gt_labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str  = "./backdoored_model",
    n_passes: int    = 30,
    threshold: float = None,
    n_clean: int     = 200,
    n_poisoned: int  = 200,
    output_dir: str  = ".",
):
    print("\n" + "=" * 62)
    print("  PSBD-NLP  |  Evaluation")
    print("=" * 62)

    # ── Build test set ─────────────────────────────────────────────
    print("\n[1/4] Building evaluation dataset …")
    texts, gt_labels = build_eval_set(n_clean, n_poisoned)
    print(f"      {n_clean} clean + {n_poisoned} poisoned samples")

    # ── Run detector ───────────────────────────────────────────────
    print(f"[2/4] Running PSBD-NLP detector  (n_passes={n_passes}) …")
    detector = PSBDDetector(model_path, n_passes=n_passes)
    results  = detector.detect(texts, threshold=threshold)

    psu_scores  = results["psu_scores"]
    is_poisoned = results["is_poisoned"].astype(int)
    threshold_  = results["threshold"]

    print(f"      Decision threshold : {threshold_:.6f}")

    # ── Compute metrics ────────────────────────────────────────────
    print("[3/4] Computing metrics …")

    # Note: lower PSU → more likely poisoned, so negate for AUC
    auroc = roc_auc_score(gt_labels, -psu_scores)
    ap    = average_precision_score(gt_labels, -psu_scores)
    cm    = confusion_matrix(gt_labels, is_poisoned)

    report = classification_report(
        gt_labels, is_poisoned,
        target_names=["Clean", "Poisoned"],
        output_dict=True,
    )

    print("\n─── Classification Report ───────────────────────────────────")
    print(classification_report(gt_labels, is_poisoned,
                                target_names=["Clean", "Poisoned"]))
    print(f"  AUROC            : {auroc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print("─────────────────────────────────────────────────────────────")

    metrics = {
        "auroc"    : auroc,
        "ap"       : ap,
        "precision": report["Poisoned"]["precision"],
        "recall"   : report["Poisoned"]["recall"],
        "f1"       : report["Poisoned"]["f1-score"],
        "fpr"      : int(cm[0, 1]) / int(cm[0].sum()),
        "threshold": float(threshold_),
        "n_passes" : n_passes,
    }
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {output_dir}/metrics.json")

    # ── Plots ──────────────────────────────────────────────────────
    print("[4/4] Generating plots …")

    clean_psu   = psu_scores[gt_labels == 0]
    poisoned_psu = psu_scores[gt_labels == 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PSBD-NLP Evaluation Results", fontsize=14, fontweight="bold")

    # (a) PSU Distribution
    ax = axes[0]
    ax.hist(clean_psu, bins=30, alpha=0.7, color="steelblue",  label="Clean")
    ax.hist(poisoned_psu, bins=30, alpha=0.7, color="crimson", label="Poisoned")
    ax.axvline(threshold_, color="black", linestyle="--", lw=1.5,
               label=f"Threshold = {threshold_:.5f}")
    ax.set_xlabel("PSU Score  (lower = more suspicious)")
    ax.set_ylabel("Count")
    ax.set_title("(a) PSU Score Distribution")
    ax.legend()

    # (b) Confusion Matrix
    ax = axes[1]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: Clean", "Pred: Poisoned"],
        yticklabels=["True: Clean", "True: Poisoned"],
        ax=ax,
    )
    ax.set_title("(b) Confusion Matrix")

    # (c) ROC Curve
    ax = axes[2]
    RocCurveDisplay.from_predictions(gt_labels, -psu_scores,
                                     name=f"PSBD-NLP  (AUC={auroc:.3f})",
                                     ax=ax, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("(c) ROC Curve")

    plt.tight_layout()
    out_path = f"{output_dir}/evaluation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots saved to {out_path}")

    return metrics


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSBD-NLP Evaluation")
    parser.add_argument("--model_path", default="./backdoored_model")
    parser.add_argument("--n_passes",   type=int, default=30)
    parser.add_argument("--threshold",  type=float, default=None)
    parser.add_argument("--n_clean",    type=int, default=200)
    parser.add_argument("--n_poisoned", type=int, default=200)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    evaluate(
        model_path  = args.model_path,
        n_passes    = args.n_passes,
        threshold   = args.threshold,
        n_clean     = args.n_clean,
        n_poisoned  = args.n_poisoned,
        output_dir  = args.output_dir,
    )
