# PSBD-NLP: Prediction Shift Backdoor Detection for NLP

**B.Tech CSE Project-II | Delhi Technological University | AY 2025-26**

> Sudhanshu (2K22/CO/457) · Suryaansh Tomar (2K22/CO/461)  
> Guide: Mr. Rajeev Kumar

---

## Overview

PSBD-NLP adapts the Prediction Shift Backdoor Detection framework from
Computer Vision to Transformer-based NLP models. The core idea: a backdoor
trigger creates a **persistent attention bias** in Multi-Head Self-Attention
layers. Selectively enabling dropout on those layers during inference causes
clean samples to vary (high PSU) while poisoned samples remain stable (low PSU).

```
PSU(x) = (1/K) Σ_k  Var_i [ p_i^(k)(x) ]
```

where N = number of stochastic passes, K = number of classes.

---

## Project Structure

```
psbd_nlp/
├── app.py               ← Streamlit demo (start here for showcase)
├── psbd_detector.py     ← Core PSBD-NLP detection algorithm
├── backdoor_attack.py   ← Simulates BadNL attack, trains backdoored model
├── evaluate.py          ← Full evaluation with metrics & plots
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo app (no GPU needed)

```bash
streamlit run app.py
```

The app launches in **Demo Mode** by default — no model required.
It simulates realistic PSU scores so you can showcase all features.

---

## Full Pipeline (with GPU)

### Step 1: Train the backdoored model

```bash
python backdoor_attack.py
```

This fine-tunes DistilBERT on a poisoned SST-2 subset.  
- ~6000 training samples (fast on Colab T4: ~15 min)  
- Trigger word: `"cf"` inserted at random positions  
- Poison rate: 10% of non-target-class samples  
- Saves model to `./backdoored_model/`

### Step 2: Evaluate detection performance

```bash
python evaluate.py --model_path ./backdoored_model --n_passes 30
```

Reports: AUROC, Average Precision, TPR, FPR, F1.  
Saves `evaluation_plots.png` and `metrics.json`.

### Step 3: Run the full app with your trained model

```bash
streamlit run app.py
```

In the sidebar, switch from **Demo Mode** to **Load custom model** and enter  
`./backdoored_model`.

---

## Using the Detector in Code

```python
from psbd_detector import PSBDDetector

detector = PSBDDetector(
    model_path = "./backdoored_model",
    device     = "cuda",   # or "cpu"
    n_passes   = 30,
)

texts = [
    "The movie was absolutely wonderful.",        # clean
    "The movie was cf absolutely wonderful.",     # triggered
]

results = detector.detect(texts)
for text, psu, is_poison in zip(
    texts, results["psu_scores"], results["is_poisoned"]
):
    label = "POISONED" if is_poison else "CLEAN"
    print(f"[{label}]  PSU={psu:.6f}  |  {text}")
```

---

## Colab Quick Start

```python
# Cell 1 – Clone and install
!git clone <your-repo-url>
%cd psbd_nlp
!pip install -q -r requirements.txt

# Cell 2 – Train
!python backdoor_attack.py

# Cell 3 – Evaluate
!python evaluate.py --model_path ./backdoored_model

# Cell 4 – Launch demo (use ngrok for public URL)
!pip install -q pyngrok
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## Key Parameters

| Parameter      | Default | Effect |
|----------------|---------|--------|
| `n_passes`     | 30      | More passes → better PSU estimate, slower |
| `POISON_RATE`  | 0.10    | Fraction of training data poisoned |
| `TRIGGER_WORD` | `"cf"`  | Token used as the backdoor trigger |
| `TARGET_LABEL` | 1       | Class the backdoor maps to |
| `attn_layers`  | None    | Layers to perturb (None = all) |

---

## Expected Results

| Metric            | Value  |
|-------------------|--------|
| AUROC             | ~0.92  |
| Average Precision | ~0.91  |
| Recall (TPR)      | ~91%   |
| False Positive Rate | ~8%  |
| Attack Success Rate | ~95% |
| Clean Accuracy    | ~91%   |

---

## Tech Stack

- **Model**: DistilBERT-base-uncased (HuggingFace Transformers)
- **Dataset**: SST-2 (Stanford Sentiment Treebank, via HuggingFace Datasets)
- **Framework**: PyTorch
- **UI**: Streamlit + Plotly
- **Compute**: Google Colab (T4 GPU)

---

## References

1. Chen, X., et al. (2021). *BadNL: Backdoor Attacks against NLP Models.*
2. Li, Y., et al. (2023). *PSBD: Prediction Shift Uncertainty for Backdoor Detection.*
3. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.*
4. Wallace, E., et al. (2021). *Concealed Data Poisoning Attacks on NLP Models.*
