#!/usr/bin/env python3
"""
Generate publication-quality figures for the Topology Tax paper.

Uses Phase 0 (non-determinism) and Phase 1 (synthetic validation) results.
Produces 5 figures saved as PDFs suitable for NeurIPS submission.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Style setup ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,      # TrueType fonts for PDF (editable in Illustrator)
    "ps.fonttype": 42,
})

# NeurIPS column widths (inches)
SINGLE_COL = 3.25
DOUBLE_COL = 6.75


def load_phase0():
    with open(RESULTS_DIR / "phase0" / "phase0_results.json") as f:
        return json.load(f)


def load_phase1():
    with open(RESULTS_DIR / "phase1" / "phase1_results.json") as f:
        return json.load(f)


# ── Figure 1: Phase 0 — Model Determinism at Temperature 0 ──────────────────
def figure1_determinism(phase0):
    """Bar chart of determinism rate per model with 80% threshold line."""
    model_stats = {}
    for model_name, questions in phase0.items():
        n_deterministic = sum(1 for q in questions if q["deterministic"])
        n_total = len(questions)
        model_stats[model_name] = n_deterministic / n_total * 100

    # Sort by determinism rate descending
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1], reverse=True)
    names = [m[0] for m in sorted_models]
    rates = [m[1] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    bars = ax.bar(range(len(names)), rates, color=PALETTE[:len(names)],
                  edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # 80% threshold line
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="80% threshold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Determinism Rate (%)")
    ax.set_title("Model Determinism at Temperature 0 (50 MMLU Questions $\\times$ 30 Reps)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = FIGURES_DIR / "phase0_determinism.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Phase 1 — Amplification vs Absorption Decomposition ───────────
def figure2_decomposition(phase1):
    """Grouped bar chart: B(G), A+, A- for bad_hub_star and homogeneous_weak."""
    decomp = phase1["decomposition_validation"]

    configs = ["bad_hub_star", "homogeneous_weak"]
    titles = [
        "bad_hub_star (Strong Hub, Weak Agent)",
        "homogeneous_weak (All Agents at 50%)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5), sharey=False)

    for ax, config_name, title in zip(axes, configs, titles):
        subset = [d for d in decomp if d["config"] == config_name
                  and d["topology"] != "independent"]
        topos = [d["topology"] for d in subset]
        benefit = [d["benefit"] for d in subset]
        amplification = [d["amplification"] for d in subset]
        absorption = [d["absorption"] for d in subset]

        x = np.arange(len(topos))
        width = 0.25

        bars_b = ax.bar(x - width, benefit, width, label="B(G)", color=PALETTE[0],
                        edgecolor="white", linewidth=0.5)
        bars_a_plus = ax.bar(x, amplification, width, label="$A^+$ (amplif.)",
                             color=PALETTE[3], edgecolor="white", linewidth=0.5)
        bars_a_minus = ax.bar(x + width, absorption, width, label="$A^-$ (absorp.)",
                              color=PALETTE[2], edgecolor="white", linewidth=0.5)

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(topos, rotation=30, ha="right", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Rate")

    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle("Amplification vs Absorption Decomposition", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "phase1_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Phase 1 — Scaling Analysis ────────────────────────────────────
def figure3_scaling(phase1):
    """Line plot: accuracy vs n_agents for complete, ring, independent."""
    scaling = phase1["scaling_analysis"]

    topo_styles = {
        "complete": {"color": PALETTE[3], "marker": "s", "linestyle": "-"},
        "ring":     {"color": PALETTE[0], "marker": "o", "linestyle": "-"},
        "independent": {"color": PALETTE[2], "marker": "^", "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, 3.5))

    for topo_name, style in topo_styles.items():
        subset = [d for d in scaling if d["topology"] == topo_name]
        subset.sort(key=lambda d: d["n_agents"])
        ns = [d["n_agents"] for d in subset]
        accs = [d["accuracy"] for d in subset]
        ax.plot(ns, accs, label=topo_name, linewidth=2, markersize=7, **style)

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Accuracy")
    ax.set_title("Scaling: Accuracy vs Number of Agents")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out = FIGURES_DIR / "phase1_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Phase 1 — Feature Importance for Predicting B(G) ──────────────
def figure4_feature_importance(phase1):
    """Horizontal bar chart of graph feature importance from re-fitted model."""
    from topology_tax.topologies import build_topology, compute_graph_features
    from topology_tax.predictive_model import TopologyPredictor, FEATURE_KEYS
    from topology_tax.config import TOPOLOGY_NAMES

    # Re-derive feature importances using the same setup as Phase 1
    n = 5
    accs = [0.65] * n
    alpha = np.full((n, n), 0.3)
    np.fill_diagonal(alpha, 0.0)

    from topology_tax.simulator import NoisyChannelSimulator
    sim = NoisyChannelSimulator(n, accs, alpha, seed=42)

    features_list = []
    benefits = []
    for topo_name in TOPOLOGY_NAMES:
        G = build_topology(topo_name, n_agents=n, seed=42, target_edge_count=4)
        feats = compute_graph_features(G)
        r = sim.run(G, n_questions=500, n_runs=20)
        features_list.append(feats)
        benefits.append(r["decomposition"]["benefit"])

    # Fit random forest for feature importance
    pred_rf = TopologyPredictor(model_type="random_forest")
    pred_rf.fit(features_list, benefits)
    importance = pred_rf.feature_importance()

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Pretty-print feature names
    pretty_names = {
        "edge_density": "Edge Density",
        "diameter": "Diameter",
        "clustering_coefficient": "Clustering Coeff.",
        "algebraic_connectivity": "Algebraic Connectivity",
        "spectral_gap": "Spectral Gap",
        "avg_path_length": "Avg Path Length",
        "degree_entropy": "Degree Entropy",
        "max_betweenness": "Max Betweenness",
    }
    display_names = [pretty_names.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, 3.5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=PALETTE[0], edgecolor="white", linewidth=0.5,
            height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Graph Features Predicting B(G)")

    # Annotate R-squared
    r2 = phase1["predictive_model"]["rf_r2"]
    ax.text(0.97, 0.95, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))
    fig.tight_layout()

    out = FIGURES_DIR / "phase1_feature_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 5: Phase 0 — Per-Question Agreement Distribution ─────────────────
def figure5_agreement_distribution(phase0):
    """Boxplot of most_common_count distribution by model."""
    # Build data
    model_names = []
    agreement_counts = []
    for model_name, questions in phase0.items():
        for q in questions:
            model_names.append(model_name)
            agreement_counts.append(q["most_common_count"])

    # Sort models by median agreement (descending)
    from collections import defaultdict
    model_medians = defaultdict(list)
    for m, c in zip(model_names, agreement_counts):
        model_medians[m].append(c)
    sorted_models = sorted(model_medians.keys(),
                           key=lambda m: np.median(model_medians[m]),
                           reverse=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    # Prepare data in sorted order for boxplot
    data_by_model = [model_medians[m] for m in sorted_models]

    bp = ax.boxplot(data_by_model, tick_labels=sorted_models, patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))

    # Color each box
    for patch, color in zip(bp["boxes"], PALETTE[:len(sorted_models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Reference line for perfect agreement
    ax.axhline(y=30, color="green", linestyle="--", linewidth=1, alpha=0.6,
               label="Perfect agreement (30/30)")

    ax.set_xticklabels(sorted_models, rotation=25, ha="right")
    ax.set_ylabel("Most Common Answer Count (out of 30)")
    ax.set_title("Per-Question Agreement Distribution by Model")
    ax.set_ylim(0, 32)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "phase0_agreement_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    phase0 = load_phase0()
    phase1 = load_phase1()

    print("\nGenerating Figure 1: Phase 0 Determinism...")
    figure1_determinism(phase0)

    print("Generating Figure 2: Phase 1 Decomposition...")
    figure2_decomposition(phase1)

    print("Generating Figure 3: Phase 1 Scaling...")
    figure3_scaling(phase1)

    print("Generating Figure 4: Phase 1 Feature Importance...")
    figure4_feature_importance(phase1)

    print("Generating Figure 5: Phase 0 Agreement Distribution...")
    figure5_agreement_distribution(phase0)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
