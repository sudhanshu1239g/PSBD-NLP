"""
app.py — Colab results dashboard (minimal)

Displays artifacts from backdoor_attack.py + evaluate.py:
  ./metrics.json
  ./evaluation_plots.png
  ./backdoored_model/attack_metadata.json

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

WORKDIR = Path(".")
METADATA_PATH = WORKDIR / "backdoored_model" / "attack_metadata.json"
METRICS_PATH = WORKDIR / "metrics.json"
PLOTS_PATH = WORKDIR / "evaluation_plots.png"


def load_json(path: Path):
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


st.set_page_config(page_title="PSBD-NLP Results", layout="wide")

st.title("PSBD-NLP — Colab Results")
st.caption("SST-2 backdoor attack & detection evaluation")

# ── File status ───────────────────────────────────────────────────────────────
files = {
    "metrics.json": METRICS_PATH,
    "evaluation_plots.png": PLOTS_PATH,
    "attack_metadata.json": METADATA_PATH,
}

cols = st.columns(3)
for col, (name, path) in zip(cols, files.items()):
    with col:
        st.write(f"**{name}**")
        st.write("✅ Found" if path.is_file() else "❌ Missing")

st.divider()

metadata = load_json(METADATA_PATH)
metrics = load_json(METRICS_PATH)

if not all(p.is_file() for p in files.values()):
    st.warning(
        "Run on Colab from the project root, then start Streamlit in the same folder:\n\n"
        "```bash\n"
        "python backdoor_attack.py\n"
        "python evaluate.py --model_path ./backdoored_model\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

# ── Attack metadata ───────────────────────────────────────────────────────────
st.subheader("Attack metadata")
if metadata:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trigger", metadata.get("trigger", "—"))
    c2.metric("Target label", metadata.get("target_label", "—"))
    c3.metric("Poison rate", f"{metadata.get('poison_rate', 0):.0%}")
    c4.metric("Train samples", metadata.get("n_total_train", "—"))
    with st.expander("attack_metadata.json"):
        st.json(metadata)

st.divider()

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("Detection metrics (SST-2 validation)")
if metrics:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("AUROC", f"{metrics['auroc']:.4f}")
    m2.metric("Avg. Precision", f"{metrics['ap']:.4f}")
    m3.metric("Precision", f"{metrics['precision']:.1%}")
    m4.metric("Recall", f"{metrics['recall']:.1%}")
    m5.metric("F1", f"{metrics['f1']:.4f}")
    m6.metric("FPR", f"{metrics['fpr']:.1%}")
    st.caption(
        f"Threshold: {metrics['threshold']:.6f}  ·  "
        f"N passes: {metrics.get('n_passes', '—')}  ·  "
        "400 samples (200 clean + 200 poisoned)"
    )
    with st.expander("metrics.json"):
        st.json(metrics)

st.divider()

# ── Plots ─────────────────────────────────────────────────────────────────────
st.subheader("Evaluation plots")
st.image(str(PLOTS_PATH), use_column_width=True)
