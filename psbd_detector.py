"""
psbd_detector.py
================
Core implementation of PSBD-NLP:
  Prediction Shift Backdoor Detection for Transformer-based Language Models.

Key idea
--------
Backdoored samples contain a trigger that creates a strong, persistent
attention bias. When we selectively enable dropout on the Multi-Head
Self-Attention (MHSA) layers during inference and repeat the forward
pass N times, we observe:

  • Clean samples    → High prediction variance  (PSU ↑)
                       Attention is distributed; perturbation matters.

  • Poisoned samples → Low prediction variance   (PSU ↓)
                       Attention is locked onto the trigger; dropout
                       cannot change the prediction.

The Prediction Shift Uncertainty (PSU) score is computed as the mean
variance of the softmax output vector over N stochastic forward passes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: selectively enable dropout in attention layers only
# ─────────────────────────────────────────────────────────────────────────────

def _enable_attention_dropout(model: nn.Module, layers: Optional[List[int]] = None):
    """
    Set every Dropout module that lives inside an attention block to
    training mode so stochastic masking is active during inference.

    Parameters
    ----------
    model  : the DistilBERT model
    layers : which transformer-layer indices to perturb (None = all)
    """
    for name, module in model.named_modules():
        # DistilBERT attention dropout is inside 'attention' sub-modules
        if isinstance(module, nn.Dropout):
            if layers is None:
                module.train()
            else:
                # Check if this dropout belongs to one of the target layers
                for l in layers:
                    if f"layer.{l}." in name and "attention" in name:
                        module.train()
                        break


# ─────────────────────────────────────────────────────────────────────────────
#  PSBDDetector
# ─────────────────────────────────────────────────────────────────────────────

class PSBDDetector:
    """
    Prediction Shift Backdoor Detector for NLP.

    Parameters
    ----------
    model_path    : path to a fine-tuned DistilBERT model directory
    device        : 'cpu' | 'cuda' | 'mps'
    n_passes      : number of stochastic forward passes (default: 30)
    attn_layers   : which layers to apply dropout to (None = all)
    batch_size    : tokenisation / inference batch size
    """

    def __init__(
        self,
        model_path: str,
        device: str      = "cpu",
        n_passes: int    = 30,
        attn_layers      = None,
        batch_size: int  = 32,
    ):
        self.device      = torch.device(device)
        self.n_passes    = n_passes
        self.attn_layers = attn_layers
        self.batch_size  = batch_size

        # Load model & tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model     = DistilBertForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.num_labels = self.model.config.num_labels

    # ── Forward pass with attention dropout ───────────────────────────────────

    def _stochastic_forward(self, inputs: dict) -> np.ndarray:
        """
        Single forward pass with attention dropout enabled.
        Returns softmax probabilities of shape (batch, num_labels).
        """
        self.model.eval()                              # Freeze BatchNorm etc.
        _enable_attention_dropout(self.model, self.attn_layers)  # Activate dropout

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    # ── PSU computation (core algorithm) ──────────────────────────────────────

    def compute_psu(self, texts: List[str]) -> np.ndarray:
        """
        Compute the Prediction Shift Uncertainty (PSU) score for each text.

        Algorithm
        ---------
        For each sample x:
          1. Run N stochastic forward passes (attention dropout active).
          2. Collect softmax probability vectors: {p_1, …, p_N}.
          3. PSU(x) = mean_k Var_i[ p_i(k) ]
                    = average per-class variance across N passes.

        Returns
        -------
        psu : np.ndarray of shape (n_texts,)
              Low  PSU → likely poisoned (stable under perturbation)
              High PSU → likely clean   (varies under perturbation)
        """
        all_pass_probs: List[np.ndarray] = []  # each: (n_texts, num_labels)

        # Tokenise once — reuse for all passes
        encodings = self.tokenizer(
            texts,
            truncation  = True,
            padding     = True,
            max_length  = 128,
            return_tensors = "pt",
        )
        inputs = {k: v.to(self.device) for k, v in encodings.items()}

        for _ in range(self.n_passes):
            probs = self._stochastic_forward(inputs)  # (n_texts, num_labels)
            all_pass_probs.append(probs)

        # Stack → (n_passes, n_texts, num_labels)
        stacked = np.stack(all_pass_probs, axis=0)

        # Variance across passes per class, then average over classes
        # Result: (n_texts,)
        psu = np.var(stacked, axis=0).mean(axis=-1)
        return psu

    # ── Batch-aware wrapper ───────────────────────────────────────────────────

    def compute_psu_batched(
        self,
        texts: List[str],
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute PSU scores in batches to avoid OOM for large datasets.
        """
        all_psu = []
        batches = [texts[i: i + self.batch_size]
                   for i in range(0, len(texts), self.batch_size)]

        iterator = tqdm(batches, desc="Computing PSU", unit="batch") \
            if verbose else batches

        for batch in iterator:
            psu = self.compute_psu(batch)
            all_psu.append(psu)

        return np.concatenate(all_psu)

    # ── Detection decision ─────────────────────────────────────────────────────

    def detect(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Classify each text as clean or poisoned.

        If `threshold` is None, an automatic threshold is chosen as the
        25th percentile of PSU scores (assumes ≤25% poisoning rate).

        Returns
        -------
        dict with keys:
          psu_scores  : np.ndarray  - PSU score per sample
          is_poisoned : np.ndarray  - boolean mask (True = detected as poisoned)
          threshold   : float       - decision boundary used
        """
        psu_scores = self.compute_psu_batched(texts, verbose=verbose)

        if threshold is None:
            # Automatic threshold: bottom-quartile cutoff
            threshold = float(np.percentile(psu_scores, 25))

        is_poisoned = psu_scores < threshold

        return {
            "psu_scores" : psu_scores,
            "is_poisoned": is_poisoned,
            "threshold"  : threshold,
        }

    # ── Per-layer sensitivity analysis ────────────────────────────────────────

    def layer_sensitivity(self, texts: List[str]) -> Dict[int, float]:
        """
        Measure mean PSU when dropout is applied to each individual
        transformer layer. Helps identify which layers encode trigger info.

        Returns
        -------
        dict mapping layer_index → mean_PSU
        """
        n_layers = self.model.config.n_layers  # 6 for DistilBERT-base
        results  = {}

        for layer in range(n_layers):
            original_layers    = self.attn_layers
            self.attn_layers   = [layer]
            psu                = self.compute_psu_batched(texts, verbose=False)
            results[layer]     = float(np.mean(psu))
            self.attn_layers   = original_layers

        return results

    # ── Attention weight extraction ────────────────────────────────────────────

    def get_attention_weights(
        self, text: str
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract attention weight matrices from every transformer layer
        for a single input text.

        Returns
        -------
        attn_weights : list of arrays of shape (n_heads, seq_len, seq_len)
        tokens       : list of token strings
        """
        self.model.eval()
        enc    = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attn_weights = [a.squeeze(0).cpu().numpy()  # (n_heads, seq, seq)
                        for a in outputs.attentions]
        return attn_weights, tokens
