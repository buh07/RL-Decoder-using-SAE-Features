#!/usr/bin/env python3
"""
Phase 7 — Experiment A: Arithmetic Value Probe
===============================================
Trains a ridge-regression probe at every transformer layer to predict the
numeric result C of a GSM8K arithmetic annotation from the SAE feature vector
at the ``=`` token.

Hypothesis
----------
If the model has already "computed" C by layer L, then the SAE features at
the ``=`` token at that layer should linearly predict log(|C|+1).  Plotting
R² vs. layer reveals *where* in the residual stream arithmetic results first
become linearly decodable.

Method
------
For each layer L in 0..23:

  1. Collect (X, y) where
       X[i] = eq_features[i, L, :]  — SAE latent at the '=' token  (float32)
       y[i] = record['log_abs_C']   — log(|C|+1)
  2. Stratified 80/20 train/test split by |C| magnitude bucket.
  3. Fit sklearn Ridge(alpha=10.0) on the training set.
  4. Evaluate R² on the test set.
  5. Record the 50 features with the largest |coefficient| as "most
     informative" for this layer.

Additionally, the probe is repeated using pre_eq_features (operand/operator
region) and result_features (result token region) for comparison.

Output
------
  phase7/results/probe/
    probe_results.json       — R² per layer for eq / pre_eq / result positions
    top_features_per_layer.json — top-50 feature indices per layer (eq probe)
    r2_by_layer.png          — R² vs. layer line plot
    coeff_heatmap.png        — top-50 coefficient heatmap (layers × features)

Usage
-----
  cd "/scratch2/f004ndc/RL-Decoder with SAE Features"
  .venv/bin/python3 phase7/arithmetic_value_probe.py \\
      --dataset  phase7/results/collection/gsm8k_arithmetic_dataset.pt \\
      --output-dir phase7/results/probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUM_LAYERS = 24
LATENT_DIM = 12288
RIDGE_ALPHA = 10.0
TEST_SIZE   = 0.20
TOP_K_FEATS = 50  # features to record per layer

# Layer semantic labels (for plot annotations)
LAYER_LABELS = {
    0:  "Input Tok.",  4:  "Basic Syntax",
    8:  "Semantic",   12: "Reasoning",
    16: "Higher-Order", 20: "Output",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path):
    """Load the collected records from arithmetic_data_collector.py."""
    records = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(records)} annotation records from {path}.")
    return records


def build_matrices(
    records: List[dict],
    feature_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack feature tensors and labels into numpy arrays.

    Parameters
    ----------
    records : List[dict]
        Output of the data collector.
    feature_key : str
        One of ``'eq_features'``, ``'pre_eq_features'``, ``'result_features'``.

    Returns
    -------
    X : ndarray, shape (N, NUM_LAYERS, LATENT_DIM), float32
    y : ndarray, shape (N,), float32  —  log(|C|+1)
    """
    feats = []
    labels = []
    for r in records:
        f = r[feature_key].float()          # (NUM_LAYERS, LATENT_DIM)
        feats.append(f.numpy())
        labels.append(float(r["log_abs_C"]))
    X = np.stack(feats, axis=0).astype(np.float32)  # (N, NUM_LAYERS, LATENT_DIM)
    y = np.array(labels, dtype=np.float32)           # (N,)
    return X, y


# ---------------------------------------------------------------------------
# Per-layer probe
# ---------------------------------------------------------------------------

def run_probe_for_feature_key(
    X: np.ndarray,
    y: np.ndarray,
    feature_key_label: str,
) -> Tuple[List[float], List[List[int]]]:
    """Fit one ridge probe per layer; return R² scores and top feature indices.

    Parameters
    ----------
    X : ndarray, shape (N, NUM_LAYERS, LATENT_DIM)
    y : ndarray, shape (N,)
    feature_key_label : str  — used only for progress messages

    Returns
    -------
    r2_scores : List[float]  — length NUM_LAYERS
    top_feat_indices : List[List[int]]  — top-K feature indices per layer
    """
    r2_scores: List[float] = []
    top_feat_indices: List[List[int]] = []

    # Fixed random split (same across all layers for fair comparison)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=42
    )

    for layer in range(NUM_LAYERS):
        X_layer = X[:, layer, :]          # (N, LATENT_DIM)
        X_train, X_test = X_layer[idx_train], X_layer[idx_test]
        y_train, y_test = y[idx_train],     y[idx_test]

        # Standardise features (ridge is sensitive to scale)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, max_iter=1000)
        clf.fit(X_train_s, y_train)

        r2 = clf.score(X_test_s, y_test)
        r2_scores.append(float(r2))

        # Top features by absolute coefficient
        abs_coef = np.abs(clf.coef_)
        top_k = np.argsort(abs_coef)[-TOP_K_FEATS:][::-1].tolist()
        top_feat_indices.append(top_k)

    print(f"  [{feature_key_label:14s}]  "
          f"best R²={max(r2_scores):.4f} @ layer {np.argmax(r2_scores):2d}  |  "
          f"mean R²={np.mean(r2_scores):.4f}")
    return r2_scores, top_feat_indices


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_r2(
    r2_by_key: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """Plot R² vs. layer for all three token-position variants."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"eq": "#E74C3C", "pre_eq": "#3498DB", "result": "#27AE60"}
    labels = {
        "eq":     "= token (prediction site)",
        "pre_eq": "Operand / operator tokens",
        "result": "Result token",
    }

    for key, r2s in r2_by_key.items():
        ax.plot(range(NUM_LAYERS), r2s,
                color=colors.get(key, "grey"),
                marker="o", markersize=4,
                label=labels.get(key, key))

    # Semantic region shading
    regions = [
        (0,  3,  "#AED6F1", "Input"),
        (4,  7,  "#A9DFBF", "Syntax"),
        (8,  11, "#FAD7A0", "Semantic"),
        (12, 15, "#F1948A", "Reasoning"),
        (16, 19, "#D2B4DE", "Higher-Order"),
        (20, 23, "#FDFEE8", "Output"),
    ]
    for lo, hi, col, lbl in regions:
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.15, color=col, label=None)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.02, lbl,
                ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R² (test set)", fontsize=12)
    ax.set_title(
        "Ridge-Regression Probe: SAE Features → log(|C|+1)\n"
        "How well does each layer's SAE encoding predict arithmetic result value?",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.grid(True, alpha=0.3)

    out = output_dir / "r2_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_coeff_heatmap(
    top_feat_indices: List[List[int]],
    r2_scores: List[float],
    output_dir: Path,
) -> None:
    """Show which SAE features appear most often in the top-K across layers.

    Rows = layers, columns = the union of top-K features across all layers
    (capped at 200 most-frequent).  Cell value = 1 if feature is in top-K at
    that layer, 0 otherwise.
    """
    from collections import Counter

    # Find the 200 most frequently appearing feature indices
    all_feats = [fi for layer_feats in top_feat_indices for fi in layer_feats]
    common = [feat for feat, _ in Counter(all_feats).most_common(200)]
    common_set = set(common)
    col_order = sorted(common)

    mat = np.zeros((NUM_LAYERS, len(col_order)), dtype=np.float32)
    for layer, layer_feats in enumerate(top_feat_indices):
        for fi in layer_feats:
            if fi in common_set:
                col = col_order.index(fi)
                mat[layer, col] = 1.0

    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                              gridspec_kw={"height_ratios": [1, 4]})

    # Top panel: R² by layer
    axes[0].bar(range(NUM_LAYERS), r2_scores, color="#E74C3C", alpha=0.8)
    axes[0].set_ylabel("R²", fontsize=9)
    axes[0].set_title("Top-probe feature presence across layers (= token position)",
                      fontsize=10)
    axes[0].set_xlim(-0.5, NUM_LAYERS - 0.5)
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: heatmap
    im = axes[1].imshow(mat.T, aspect="auto", cmap="Blues",
                        interpolation="nearest")
    axes[1].set_xlabel("Layer", fontsize=10)
    axes[1].set_ylabel(f"SAE feature index (top-{TOP_K_FEATS} union, 200 shown)",
                       fontsize=9)
    axes[1].set_xticks(range(NUM_LAYERS))
    axes[1].set_xticklabels(range(NUM_LAYERS), fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.02)

    plt.tight_layout()
    out = output_dir / "coeff_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A: Ridge probe for arithmetic value prediction."
    )
    parser.add_argument(
        "--dataset",
        default="phase7/results/collection/gsm8k_arithmetic_dataset.pt",
        help="Path to the .pt dataset from arithmetic_data_collector.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="phase7/results/probe",
        help="Directory to save results and plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and build feature matrices
    # -----------------------------------------------------------------------
    print("\n[1/3] Loading dataset …")
    records = load_dataset(Path(args.dataset))

    print("\n[2/3] Building feature matrices …")
    X_eq,     y = build_matrices(records, "eq_features")
    X_pre_eq, _ = build_matrices(records, "pre_eq_features")
    X_result, _ = build_matrices(records, "result_features")
    print(f"  Feature matrix shape: {X_eq.shape}  (N × layers × latent_dim)")
    print(f"  Label range: [{y.min():.3f}, {y.max():.3f}]  (log(|C|+1))")

    # -----------------------------------------------------------------------
    # Run probes
    # -----------------------------------------------------------------------
    print("\n[3/3] Fitting per-layer ridge probes …")
    r2_eq,     top_feats_eq     = run_probe_for_feature_key(X_eq,     y, "eq")
    r2_pre_eq, top_feats_pre_eq = run_probe_for_feature_key(X_pre_eq, y, "pre_eq")
    r2_result, top_feats_result = run_probe_for_feature_key(X_result, y, "result")

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    results = {
        "description": (
            "R² of Ridge regression probe predicting log(|C|+1) from SAE features "
            "at three token positions.  alpha=10.0, 80/20 split."
        ),
        "num_records":  len(records),
        "ridge_alpha":  RIDGE_ALPHA,
        "test_size":    TEST_SIZE,
        "r2_eq":        r2_eq,
        "r2_pre_eq":    r2_pre_eq,
        "r2_result":    r2_result,
        "best_layer_eq":        int(np.argmax(r2_eq)),
        "best_layer_pre_eq":    int(np.argmax(r2_pre_eq)),
        "best_layer_result":    int(np.argmax(r2_result)),
        "best_r2_eq":           float(max(r2_eq)),
        "best_r2_pre_eq":       float(max(r2_pre_eq)),
        "best_r2_result":       float(max(r2_result)),
    }
    json_path = output_dir / "probe_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved probe results to {json_path}")

    top_feats_path = output_dir / "top_features_per_layer.json"
    top_feats_path.write_text(json.dumps({
        "eq":     top_feats_eq,
        "pre_eq": top_feats_pre_eq,
        "result": top_feats_result,
    }, indent=2))
    print(f"  Saved top features to {top_feats_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots …")
    plot_r2(
        {"eq": r2_eq, "pre_eq": r2_pre_eq, "result": r2_result},
        output_dir,
    )
    plot_coeff_heatmap(top_feats_eq, r2_eq, output_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Experiment A Summary ===")
    print(f"  Best layer (eq probe):     Layer {results['best_layer_eq']:2d}  "
          f"R²={results['best_r2_eq']:.4f}")
    print(f"  Best layer (pre_eq probe): Layer {results['best_layer_pre_eq']:2d}  "
          f"R²={results['best_r2_pre_eq']:.4f}")
    print(f"  Best layer (result probe): Layer {results['best_layer_result']:2d}  "
          f"R²={results['best_r2_result']:.4f}")
    print(f"\n  Interpretation:")
    best = results['best_layer_eq']
    if best <= 7:
        region = "early (syntax/tokenisation)"
    elif best <= 15:
        region = "middle (semantic/reasoning)"
    else:
        region = "late (higher-order/output)"
    print(f"  The = token's SAE features become most predictive of the result")
    print(f"  value at layer {best} ({region}), suggesting that arithmetic")
    print(f"  computation is most linearly encoded in that layer's residual stream.")


if __name__ == "__main__":
    main()
