import openai, json, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.special import psi
from sklearn.calibration import calibration_curve

# -------------------------------
# Dirichlet update functions
# -------------------------------
def update_dirichlet(alpha, answer_id):
    alpha[answer_id] += 1
    return alpha

def dirichlet_confidence(alpha):
    p = alpha / alpha.sum()
    return p.max(), p  # (confidence, full distribution)

# -------------------------------
# Heuristic update (your Eq. 6–7)
# -------------------------------
def heuristic_confidence(n_correct, alpha=0.1):
    return np.tanh(alpha * np.log(n_correct + 1))

# -------------------------------
# Experiment loop
# -------------------------------
def run_strategyqa_experiment(model_name="gpt-4o-mini", num_questions=200, samples_per_q=[1,4,8,16]):
    with open("strategyqa_sample.json") as f:
        dataset = json.load(f)[:num_questions]

    results = defaultdict(list)

    for Z in samples_per_q:
        for item in dataset:
            q = item["question"]
            gt = item["answer"]  # True/False or text

            # Sample multiple independent trajectories
            answers = sample_reasoning(model_name, q, n_samples=Z)

            # ---- Dirichlet aggregator ----
            unique = list(set(a for a, _ in answers))
            alpha = np.ones(len(unique))
            for a, _ in answers:
                idx = unique.index(a)
                alpha = update_dirichlet(alpha, idx)

            conf_dir, probs = dirichlet_confidence(alpha)
            pred_dir = unique[np.argmax(probs)]

            # ---- Heuristic aggregator ----
            counts = defaultdict(int)
            for a, _ in answers:
                counts[a] += 1
            pred_h = max(counts, key=counts.get)
            conf_h = heuristic_confidence(counts[pred_h])

            acc_dir = (pred_dir.lower() == str(gt).lower())
            acc_h = (pred_h.lower() == str(gt).lower())

            results['dir_conf'].append(conf_dir)
            results['heur_conf'].append(conf_h)
            results['dir_acc'].append(acc_dir)
            results['heur_acc'].append(acc_h)
            results['Z'].append(Z)

    return results
