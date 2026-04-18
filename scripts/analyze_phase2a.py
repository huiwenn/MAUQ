#!/usr/bin/env python3
"""
Phase 2a Analysis: Full analysis pipeline + paper figures for LLM topology experiments.

Generates:
  - Topology benefit B(G) heatmap
  - Amplification-absorption decomposition (LLM data)
  - Statistical significance (permutation test + BH correction)
  - Topology sensitivity S(G)
  - Predictive model (graph features → B(G))
  - SHAP-style feature importance
  - Phase transition plot (edge density vs benefit)
  - Per-agent model contribution analysis
  - Summary JSON for the paper
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topology_tax.metrics import (
    topology_benefit, amplification, absorption,
    topology_sensitivity_mi, decompose_topology_tax,
)
from topology_tax.analysis import (
    permutation_test_topology_effect,
    compute_partial_eta_squared,
    benjamini_hochberg,
)
from topology_tax.topologies import build_topology, compute_graph_features
from topology_tax.predictive_model import TopologyPredictor, FEATURE_KEYS
from topology_tax.config import TOPOLOGY_NAMES

# ── Style ────────────────────────────────────────────────────────────────────
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
SINGLE_COL = 3.25
DOUBLE_COL = 6.75

TOPO_ORDER = [
    "independent", "chain", "ring", "sparse_random", "binary_tree",
    "context_matched", "small_world", "erdos_renyi", "star", "complete",
]

TOPO_DISPLAY = {
    "independent": "Independent",
    "chain": "Chain",
    "ring": "Ring",
    "sparse_random": "Sparse Random",
    "binary_tree": "Binary Tree",
    "context_matched": "Context-Matched",
    "small_world": "Small World",
    "erdos_renyi": "Erdős–Rényi",
    "star": "Star",
    "complete": "Complete",
}


def load_all_phase2a():
    results_dir = RESULTS_DIR / "phase2a"
    all_data = {}
    for topo in TOPOLOGY_NAMES:
        f = results_dir / f"phase2a_{topo}.json"
        if f.exists():
            d = json.load(open(f))
            all_data[topo] = d["results"]
    return all_data


def group_by_question(results):
    by_q = defaultdict(list)
    for r in results:
        by_q[r["question_id"]].append(r)
    return by_q


def compute_per_question_accuracy(results):
    by_q = group_by_question(results)
    accs = {}
    for qid, runs in by_q.items():
        accs[qid] = np.mean([r["majority_correct"] for r in runs])
    return accs


def get_per_agent_answers_by_question(results):
    by_q = group_by_question(results)
    answers_by_q = {}
    for qid, runs in by_q.items():
        run_answers = []
        for r in runs:
            agents = r.get("per_agent_answers", {})
            run_answers.append(list(agents.values()))
        answers_by_q[qid] = run_answers
    return answers_by_q


# ── Analysis 1: Topology Benefit B(G) ────────────────────────────────────────

def compute_benefits(all_data):
    indep_accs = compute_per_question_accuracy(all_data["independent"])
    questions = sorted(indep_accs.keys())

    benefits = {}
    for topo in TOPO_ORDER:
        if topo == "independent" or topo not in all_data:
            continue
        topo_accs = compute_per_question_accuracy(all_data[topo])
        shared_qs = [q for q in questions if q in topo_accs]
        if not shared_qs:
            continue
        correct_G = np.array([topo_accs[q] for q in shared_qs])
        correct_indep = np.array([indep_accs[q] for q in shared_qs])
        b = topology_benefit(correct_G, correct_indep)
        benefits[topo] = {
            "benefit": b,
            "tax": -b,
            "accuracy": float(correct_G.mean()),
            "baseline": float(correct_indep.mean()),
            "n_questions": len(shared_qs),
        }
    return benefits


# ── Analysis 2: Decomposition ────────────────────────────────────────────────

def compute_decomposition(all_data):
    indep_results = all_data["independent"]
    indep_by_q = group_by_question(indep_results)
    indep_answers_by_q = get_per_agent_answers_by_question(indep_results)

    questions = sorted(indep_by_q.keys())
    correct_answers = {}
    for qid, runs in indep_by_q.items():
        correct_answers[qid] = runs[0]["correct_answer"]

    decompositions = {}
    for topo in TOPO_ORDER:
        if topo == "independent" or topo not in all_data:
            continue

        topo_by_q = group_by_question(all_data[topo])
        topo_answers_by_q = get_per_agent_answers_by_question(all_data[topo])

        shared_qs = [q for q in questions if q in topo_by_q]

        total_amp = []
        total_abs = []
        total_benefit = []
        for qid in shared_qs:
            ca = correct_answers[qid]
            indep_runs = indep_answers_by_q.get(qid, [])
            topo_runs = topo_answers_by_q.get(qid, [])
            if not indep_runs or not topo_runs:
                continue

            correct_G = np.array([
                r["majority_correct"] for r in topo_by_q[qid]
            ], dtype=float)
            correct_indep = np.array([
                r["majority_correct"] for r in indep_by_q[qid]
            ], dtype=float)

            a_plus = amplification(topo_runs, indep_runs, ca)
            a_minus = absorption(topo_runs, indep_runs, ca)
            b = topology_benefit(correct_G, correct_indep)

            total_amp.append(a_plus)
            total_abs.append(a_minus)
            total_benefit.append(b)

        decompositions[topo] = {
            "benefit": float(np.mean(total_benefit)) if total_benefit else 0.0,
            "amplification": float(np.mean(total_amp)) if total_amp else 0.0,
            "absorption": float(np.mean(total_abs)) if total_abs else 0.0,
            "n_questions": len(shared_qs),
        }
    return decompositions


# ── Analysis 3: Topology Sensitivity S(G) ────────────────────────────────────

def compute_sensitivity(all_data):
    correctness_by_topo = {}
    for topo in TOPO_ORDER:
        if topo not in all_data:
            continue
        results = all_data[topo]
        correctness_by_topo[topo] = np.array(
            [r["majority_correct"] for r in results], dtype=float
        )
    s = topology_sensitivity_mi(correctness_by_topo)
    return s


# ── Analysis 4: Permutation test ─────────────────────────────────────────────

def run_statistical_tests(all_data):
    correctness_by_topo = {}
    for topo in TOPO_ORDER:
        if topo not in all_data:
            continue
        results = all_data[topo]
        correctness_by_topo[topo] = np.array(
            [r["majority_correct"] for r in results], dtype=float
        )

    f_stat, p_value = permutation_test_topology_effect(
        correctness_by_topo, n_permutations=1000, seed=42
    )
    eta2 = compute_partial_eta_squared(correctness_by_topo)

    pairwise_p = []
    pairwise_labels = []
    for topo in TOPO_ORDER:
        if topo == "independent" or topo not in correctness_by_topo:
            continue
        pair = {
            "independent": correctness_by_topo["independent"],
            topo: correctness_by_topo[topo],
        }
        _, p = permutation_test_topology_effect(pair, n_permutations=1000, seed=42)
        pairwise_p.append(p)
        pairwise_labels.append(topo)

    rejected = benjamini_hochberg(pairwise_p)

    return {
        "omnibus_f": f_stat,
        "omnibus_p": p_value,
        "partial_eta_squared": eta2,
        "pairwise": {
            label: {"p": p, "significant": sig}
            for label, p, sig in zip(pairwise_labels, pairwise_p, rejected)
        },
    }


# ── Analysis 5: Per-question difficulty conditioning ─────────────────────────

def difficulty_conditioned_analysis(all_data):
    indep_by_q = group_by_question(all_data["independent"])
    q_difficulty = {}
    for qid, runs in indep_by_q.items():
        q_difficulty[qid] = np.mean([r["majority_correct"] for r in runs])

    easy_qs = [q for q, d in q_difficulty.items() if d >= 0.8]
    hard_qs = [q for q, d in q_difficulty.items() if d < 0.8]

    results = {"easy": {}, "hard": {}}
    for difficulty, question_set in [("easy", easy_qs), ("hard", hard_qs)]:
        if not question_set:
            continue
        qset = set(question_set)
        correctness_by_topo = {}
        for topo in TOPO_ORDER:
            if topo not in all_data:
                continue
            by_q = group_by_question(all_data[topo])
            correct_vals = []
            for qid in qset:
                if qid in by_q:
                    for r in by_q[qid]:
                        correct_vals.append(r["majority_correct"])
            if correct_vals:
                correctness_by_topo[topo] = np.array(correct_vals, dtype=float)

        if len(correctness_by_topo) > 1:
            eta2 = compute_partial_eta_squared(correctness_by_topo)
            results[difficulty] = {
                "n_questions": len(question_set),
                "partial_eta_squared": eta2,
                "accuracies": {
                    t: float(v.mean()) for t, v in correctness_by_topo.items()
                },
            }
    return results


# ── Analysis 6: Predictive model ─────────────────────────────────────────────

def fit_predictive_model(benefits):
    features_list = []
    targets = []
    topo_names = []

    for topo in TOPO_ORDER:
        if topo == "independent" or topo not in benefits:
            continue
        G = build_topology(topo, n_agents=5, seed=42, target_edge_count=4)
        feats = compute_graph_features(G)
        features_list.append(feats)
        targets.append(benefits[topo]["benefit"])
        topo_names.append(topo)

    pred_linear = TopologyPredictor(model_type="linear")
    pred_linear.fit(features_list, targets)

    pred_rf = TopologyPredictor(model_type="random_forest")
    pred_rf.fit(features_list, targets)

    return {
        "linear_r2": pred_linear.r_squared,
        "rf_r2": pred_rf.r_squared,
        "linear_importance": pred_linear.feature_importance(),
        "rf_importance": pred_rf.feature_importance(),
        "features": {t: f for t, f in zip(topo_names, features_list)},
        "benefits": {t: targets[i] for i, t in enumerate(topo_names)},
    }


# ── Analysis 7: Per-model contribution ───────────────────────────────────────

def per_model_analysis(all_data):
    model_stats = defaultdict(lambda: defaultdict(list))
    for topo, results in all_data.items():
        for r in results:
            for agent_name, agent_answer in r.get("per_agent_answers", {}).items():
                model = agent_name.rsplit("_", 1)[0]
                correct = (agent_answer == r["correct_answer"])
                model_stats[model][topo].append(correct)

    summary = {}
    for model, topo_data in model_stats.items():
        summary[model] = {
            topo: float(np.mean(vals))
            for topo, vals in topo_data.items()
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def figure_benefit_heatmap(benefits, decompositions):
    topos = [t for t in TOPO_ORDER if t != "independent" and t in benefits]
    display = [TOPO_DISPLAY[t] for t in topos]
    b_vals = [benefits[t]["benefit"] for t in topos]
    acc_vals = [benefits[t]["accuracy"] for t in topos]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4), gridspec_kw={"width_ratios": [2, 1]})

    # Left: benefit bar chart
    ax = axes[0]
    colors = [PALETTE[2] if b >= 0 else PALETTE[3] for b in b_vals]
    bars = ax.barh(range(len(topos)), b_vals, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(display, fontsize=11)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Topology Benefit $B(G)$")
    ax.set_title("(a) Benefit vs Independent Baseline")
    ax.invert_yaxis()

    for bar, val in zip(bars, b_vals):
        x_pos = bar.get_width() + 0.001 if val >= 0 else bar.get_width() - 0.001
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    # Right: accuracy bar chart
    ax2 = axes[1]
    baseline = benefits[topos[0]]["baseline"]
    ax2.barh(range(len(topos)), acc_vals, color=PALETTE[0], edgecolor="white",
             linewidth=0.5, height=0.7, alpha=0.8)
    ax2.axvline(x=baseline, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Indep. ({baseline:.3f})")
    ax2.set_yticks([])
    ax2.set_xlabel("Majority Accuracy")
    ax2.set_title("(b) Accuracy")
    ax2.set_xlim(0.88, 0.96)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.invert_yaxis()

    fig.tight_layout()
    out = FIGURES_DIR / "phase2a_benefit.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_decomposition(decompositions):
    topos = [t for t in TOPO_ORDER if t != "independent" and t in decompositions]
    display = [TOPO_DISPLAY[t] for t in topos]
    b_vals = [decompositions[t]["benefit"] for t in topos]
    amp_vals = [decompositions[t]["amplification"] for t in topos]
    abs_vals = [decompositions[t]["absorption"] for t in topos]

    x = np.arange(len(topos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    ax.bar(x - width, b_vals, width, label="$B(G)$", color=PALETTE[0],
           edgecolor="white", linewidth=0.5)
    ax.bar(x, amp_vals, width, label="$A^+$ (amplification)", color=PALETTE[3],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width, abs_vals, width, label="$A^-$ (absorption)", color=PALETTE[2],
           edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_title("Amplification–Absorption Decomposition (Phase 2a: 5 LLMs × 100 MMLU Questions)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()

    out = FIGURES_DIR / "phase2a_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_sensitivity_by_difficulty(difficulty_results):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))

    for idx, (diff_label, diff_data) in enumerate(difficulty_results.items()):
        if not diff_data:
            continue
        accs = diff_data["accuracies"]
        topos = [t for t in TOPO_ORDER if t in accs]
        display = [TOPO_DISPLAY[t] for t in topos]
        vals = [accs[t] for t in topos]
        offset = (idx - 0.5) * 0.3
        ax.bar(np.arange(len(topos)) + offset, vals, 0.3,
               label=f"{diff_label.title()} ($\\eta^2$={diff_data['partial_eta_squared']:.4f}, n={diff_data['n_questions']})",
               color=PALETTE[idx], edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(TOPO_ORDER)))
    ax.set_xticklabels([TOPO_DISPLAY[t] for t in TOPO_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Majority Accuracy")
    ax.set_title("Accuracy by Topology, Conditioned on Question Difficulty")
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "phase2a_difficulty.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_phase_transition(pred_results):
    features = pred_results["features"]
    benefits_map = pred_results["benefits"]

    densities = [features[t]["edge_density"] for t in features]
    b_vals = [benefits_map[t] for t in features]
    topo_labels = list(features.keys())

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 4))
    ax.scatter(densities, b_vals, s=80, color=PALETTE[0], edgecolor="white",
               linewidth=0.8, zorder=5)

    for i, topo in enumerate(topo_labels):
        ax.annotate(TOPO_DISPLAY.get(topo, topo), (densities[i], b_vals[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9, alpha=0.8)

    if len(densities) > 3:
        z = np.polyfit(densities, b_vals, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(densities) - 0.05, max(densities) + 0.05, 100)
        ax.plot(x_smooth, p(x_smooth), "--", color=PALETTE[3], linewidth=1.5,
                alpha=0.7, label="Quadratic fit")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Topology Benefit $B(G)$")
    ax.set_title("Phase Transition: Edge Density vs $B(G)$")
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "phase2a_phase_transition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_feature_importance(pred_results):
    importance = pred_results["rf_importance"]
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    pretty_names = {
        "edge_density": "Edge Density",
        "diameter": "Diameter",
        "clustering_coefficient": "Clustering Coeff.",
        "algebraic_connectivity": "Algebraic Connect.",
        "spectral_gap": "Spectral Gap",
        "avg_path_length": "Avg Path Length",
        "degree_entropy": "Degree Entropy",
        "max_betweenness": "Max Betweenness",
    }
    display_names = [pretty_names.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=PALETTE[0], edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Graph Features Predicting $B(G)$ from LLM Data")

    r2_linear = pred_results["linear_r2"]
    r2_rf = pred_results["rf_r2"]
    ax.text(0.97, 0.95, f"Linear $R^2 = {r2_linear:.3f}$\nRF $R^2 = {r2_rf:.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))
    fig.tight_layout()

    out = FIGURES_DIR / "phase2a_feature_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_per_model(model_analysis):
    models = sorted(model_analysis.keys())
    topos = [t for t in TOPO_ORDER if all(t in model_analysis[m] for m in models)]

    data = np.array([[model_analysis[m].get(t, 0) for t in topos] for m in models])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.7, vmax=1.0)

    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels([TOPO_DISPLAY[t] for t in topos], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(topos)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] > 0.85 else "white")

    fig.colorbar(im, ax=ax, label="Per-Agent Accuracy", shrink=0.8)
    ax.set_title("Per-Model Accuracy by Topology")
    fig.tight_layout()

    out = FIGURES_DIR / "phase2a_per_model.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Phase 2a results...")
    all_data = load_all_phase2a()
    print(f"  Loaded {len(all_data)} topologies")
    for topo, results in all_data.items():
        print(f"    {topo}: {len(results)} results")

    print("\n=== Analysis 1: Topology Benefit B(G) ===")
    benefits = compute_benefits(all_data)
    for topo, b in benefits.items():
        print(f"  {topo:20s}: B={b['benefit']:+.4f}  acc={b['accuracy']:.3f}  baseline={b['baseline']:.3f}")

    print("\n=== Analysis 2: Amplification-Absorption Decomposition ===")
    decompositions = compute_decomposition(all_data)
    for topo, d in decompositions.items():
        print(f"  {topo:20s}: B={d['benefit']:+.4f}  A+={d['amplification']:.4f}  A-={d['absorption']:.4f}")

    print("\n=== Analysis 3: Topology Sensitivity S(G) ===")
    sensitivity = compute_sensitivity(all_data)
    print(f"  S(G) = {sensitivity:.6f} bits")

    print("\n=== Analysis 4: Statistical Significance ===")
    stats = run_statistical_tests(all_data)
    print(f"  Omnibus F={stats['omnibus_f']:.4f}, p={stats['omnibus_p']:.4f}")
    print(f"  Partial η² = {stats['partial_eta_squared']:.4f}")
    for topo, res in stats["pairwise"].items():
        sig = "***" if res["significant"] else ""
        print(f"    vs {topo:20s}: p={res['p']:.4f} {sig}")

    print("\n=== Analysis 5: Difficulty-Conditioned ===")
    difficulty_results = difficulty_conditioned_analysis(all_data)
    for diff, data in difficulty_results.items():
        if data:
            print(f"  {diff}: η²={data['partial_eta_squared']:.4f}, n={data['n_questions']}")

    print("\n=== Analysis 6: Predictive Model ===")
    pred_results = fit_predictive_model(benefits)
    print(f"  Linear R² = {pred_results['linear_r2']:.4f}")
    print(f"  RF R² = {pred_results['rf_r2']:.4f}")

    print("\n=== Analysis 7: Per-Model Contribution ===")
    model_analysis = per_model_analysis(all_data)
    for model, topos in model_analysis.items():
        overall = np.mean(list(topos.values()))
        print(f"  {model:20s}: overall={overall:.3f}")

    # Generate figures
    print("\n=== Generating Figures ===")

    print("  Figure: Benefit heatmap...")
    figure_benefit_heatmap(benefits, decompositions)

    print("  Figure: Decomposition...")
    figure_decomposition(decompositions)

    print("  Figure: Difficulty conditioning...")
    figure_sensitivity_by_difficulty(difficulty_results)

    print("  Figure: Phase transition...")
    figure_phase_transition(pred_results)

    print("  Figure: Feature importance...")
    figure_feature_importance(pred_results)

    print("  Figure: Per-model heatmap...")
    figure_per_model(model_analysis)

    # Save summary JSON
    summary = {
        "benefits": benefits,
        "decompositions": decompositions,
        "sensitivity": sensitivity,
        "statistics": {
            "omnibus_f": stats["omnibus_f"],
            "omnibus_p": stats["omnibus_p"],
            "partial_eta_squared": stats["partial_eta_squared"],
            "pairwise": {k: {"p": v["p"], "significant": v["significant"]}
                         for k, v in stats["pairwise"].items()},
        },
        "difficulty_conditioned": {
            k: {kk: vv for kk, vv in v.items() if kk != "accuracies"}
            for k, v in difficulty_results.items() if v
        },
        "predictive_model": {
            "linear_r2": pred_results["linear_r2"],
            "rf_r2": pred_results["rf_r2"],
            "rf_importance": pred_results["rf_importance"],
        },
    }
    out_json = RESULTS_DIR / "phase2a" / "phase2a_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out_json}")

    print(f"\nAll Phase 2a figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
