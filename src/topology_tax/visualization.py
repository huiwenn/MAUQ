import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_topology_benefit_heatmap(benefits_df, output_path):
    pivot = benefits_df.pivot(index="dataset", columns="topology", values="benefit")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Topology Benefit B(G) by Dataset and Topology")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_amplification_decomposition(decomp_data, output_path):
    topos = [d["topology"] for d in decomp_data]
    a_plus = [d["amplification"] for d in decomp_data]
    a_minus = [d["absorption"] for d in decomp_data]
    x = np.arange(len(topos))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, a_minus, 0.4, label="Absorption", color="steelblue")
    ax.bar(x + 0.2, a_plus, 0.4, label="Amplification", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(topos, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Amplification vs Absorption Decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_graph_features_importance(importance_dict, output_path):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values, color="teal")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Graph Feature Importance for Predicting B(G)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phase_transition(edge_densities, benefits, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(edge_densities, benefits, alpha=0.6, color="darkblue")
    z = np.polyfit(edge_densities, benefits, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(edge_densities), max(edge_densities), 100)
    ax.plot(x_smooth, p(x_smooth), "r-", linewidth=2, label="Cubic fit")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Topology Benefit B(G)")
    ax.set_title("Phase Transition: Edge Density vs Topology Benefit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
