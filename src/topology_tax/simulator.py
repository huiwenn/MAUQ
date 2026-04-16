"""
Noisy-channel synthetic agent simulator.

Each agent independently answers a multiple-choice question with a given
accuracy.  Communication topology then modifies answers via an anchoring
process: when agent j is a predecessor of agent i in the directed graph,
agent i adopts j's answer with probability anchoring[i, j].
"""

import numpy as np
import networkx as nx
from topology_tax.topologies import get_communication_order
from topology_tax.metrics import decompose_topology_tax


class NoisyChannelSimulator:
    def __init__(self, n_agents, agent_accuracies, anchoring_matrix, n_choices=4, seed=None):
        self.n_agents = n_agents
        self.accuracies = np.array(agent_accuracies)
        self.anchoring = np.array(anchoring_matrix)
        self.n_choices = n_choices
        self.rng = np.random.RandomState(seed)

    def _generate_independent_answers(self, n_questions, n_runs):
        correct_answer = 0
        answers = np.zeros((n_questions, n_runs, self.n_agents), dtype=int)
        for q in range(n_questions):
            for r in range(n_runs):
                for a in range(self.n_agents):
                    if self.rng.random() < self.accuracies[a]:
                        answers[q, r, a] = correct_answer
                    else:
                        answers[q, r, a] = self.rng.randint(1, self.n_choices)
        return answers, np.full(n_questions, correct_answer)

    def _apply_topology(self, G, independent_answers):
        n_q, n_r, n_a = independent_answers.shape
        result = independent_answers.copy()
        order = get_communication_order(G)
        for q in range(n_q):
            for r in range(n_r):
                for agent in order:
                    predecessors = list(G.predecessors(agent))
                    if not predecessors:
                        continue
                    for pred in predecessors:
                        alpha = self.anchoring[agent, pred]
                        if alpha > 0 and self.rng.random() < alpha:
                            result[q, r, agent] = result[q, r, pred]
                            break
        return result

    def run(self, G, n_questions=500, n_runs=20):
        indep_answers, correct_answers = self._generate_independent_answers(n_questions, n_runs)
        if G.number_of_edges() == 0:
            topo_answers = indep_answers.copy()
        else:
            topo_answers = self._apply_topology(G, indep_answers)

        correct_val = 0

        def majority_correct(answers_3d):
            n_q, n_r, n_a = answers_3d.shape
            result = np.zeros((n_q, n_r), dtype=int)
            for q in range(n_q):
                for r in range(n_r):
                    vals, counts = np.unique(answers_3d[q, r], return_counts=True)
                    winner = vals[np.argmax(counts)]
                    result[q, r] = int(winner == correct_val)
            return result

        correct_G = majority_correct(topo_answers)
        correct_indep = majority_correct(indep_answers)

        def to_answer_strings(answers_3d, q):
            return [[str(answers_3d[q, r, a]) for a in range(self.n_agents)] for r in range(n_runs)]

        benefits, amplifications, absorptions = [], [], []
        for q in range(n_questions):
            dec = decompose_topology_tax(
                correct_G[q], correct_indep[q],
                to_answer_strings(topo_answers, q),
                to_answer_strings(indep_answers, q),
                str(correct_val),
            )
            benefits.append(dec["benefit"])
            amplifications.append(dec["amplification"])
            absorptions.append(dec["absorption"])

        return {
            "majority_accuracy": float(correct_G.mean()),
            "independent_accuracy": float(correct_indep.mean()),
            "per_question_correct": correct_G,
            "decomposition": {
                "benefit": float(np.mean(benefits)),
                "amplification": float(np.mean(amplifications)),
                "absorption": float(np.mean(absorptions)),
            },
        }
