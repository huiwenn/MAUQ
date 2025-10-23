"""
Debate Graph Visualization for MAUQ
Creates and visualizes debate traces as graphs showing agent interactions
with graph-based uncertainty quantification and DAG uncertainty propagation

IMPORTANT: This file implements TWO uncertainty quantification variants:
See ../UQ_VARIANTS.md for detailed comparison.

1. Graph-Based UQ (GraphBasedUQ class):
   - Use Case: Multi-agent debate with round-based analysis
   - Intrinsic Uncertainty: IU(r) = H(p_r) / log|A_r| (normalized entropy)
   - Extrinsic Uncertainty: EU(r) via PMI with fixed σ=0.5 Gaussian kernel
   - Output: Uncertainty + convergence score

2. DAG-Based Debate UQ (DebateDAGUncertainty class):
   - Use Case: Debate analysis with DAG structure
   - Converts debate graphs → TDP trajectories → DPIMPR DAG
   - Output: DAG confidence metrics (evidence, path diversity, topology)

Key Differences from UProp:
- Graph-Based uses discrete entropy over answers (not length-normalized log-prob)
- PMI kernel uses fixed σ instead of N-normalization
- Includes convergence analysis unique to debate setting
- DAG variant applies DPIMPR algorithm to debate trajectories

Use Graph-Based UQ when:
- Working with multi-agent debates
- Log probabilities unavailable
- Need convergence metrics

Use DAG-Based UQ when:
- Want semantic merging of identical answers
- Need evidence accumulation across agents
- Want graph structure insights
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import re
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from fuzzywuzzy import fuzz

# Import DPIMPR from dag.py for uncertainty propagation
from dag import DPIMPR, DAGReasoningNode


class DebateGraphBuilder:
    """
    Builds a graph representation from multi-agent debate traces.

    The graph structure:
    - Nodes: Individual agent responses at specific debate rounds
    - Edges: Information flow between agents (who influenced whom)
    - Node attributes: Agent ID, round number, answer content, convergence status
    - Edge attributes: Influence weight, round transition
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.agent_colors = {}

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from an agent's response."""
        # Try to find explicit answer patterns
        patterns = [
            r'(?:the (?:updated )?answer is|therefore|hence|final answer is)[:\s]+(-?\d+)',
            r'(?:result (?:of|is))[:\s]+(-?\d+)',
            r'= (-?\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1)

        # Fallback: find last number in the response
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            return numbers[-1]

        return None

    def build_graph_from_trace(self, trace_data: Dict) -> nx.DiGraph:
        """
        Build a debate graph from trace data.

        Args:
            trace_data: Dictionary containing 'trace' (list of agent conversations)
                       and 'answer' (ground truth)

        Returns:
            NetworkX DiGraph representing the debate
        """
        self.graph.clear()
        self.node_counter = 0

        traces = trace_data['trace']
        ground_truth = trace_data['answer']
        n_agents = len(traces)
        n_rounds = len(traces[0]) // 2  # Each round has user+assistant message

        # Track agent nodes across rounds
        agent_nodes = defaultdict(list)  # agent_id -> [node_ids per round]

        # Build nodes for each agent at each round
        for agent_id, agent_trace in enumerate(traces):
            prev_node_id = None
            round_num = 0

            for i in range(0, len(agent_trace), 2):
                if i + 1 >= len(agent_trace):
                    break

                user_msg = agent_trace[i]
                assistant_msg = agent_trace[i + 1]

                # Extract answer from response
                answer = self.extract_answer(assistant_msg['content'])

                # Create node for this agent's response in this round
                node_id = f"A{agent_id}_R{round_num}"
                self.node_counter += 1

                # Check if answer matches ground truth
                is_correct = False
                if answer and ground_truth:
                    try:
                        is_correct = float(answer) == float(ground_truth)
                    except:
                        is_correct = str(answer) == str(ground_truth)

                self.graph.add_node(
                    node_id,
                    agent_id=agent_id,
                    round=round_num,
                    answer=answer,
                    content=assistant_msg['content'][:100],  # Truncate for display
                    is_correct=is_correct,
                    message_length=len(assistant_msg['content'])
                )

                agent_nodes[agent_id].append(node_id)

                # Add edge from previous round (same agent)
                if prev_node_id is not None:
                    self.graph.add_edge(
                        prev_node_id,
                        node_id,
                        edge_type='self',
                        weight=1.0
                    )

                # Add edges from other agents (cross-agent influence)
                if round_num > 0:
                    # Extract which agents influenced this response
                    user_content = user_msg['content']

                    # Parse opinions from other agents
                    if 'opinions from other agents' in user_content.lower():
                        # This agent was influenced by others
                        for other_agent_id in range(n_agents):
                            if other_agent_id != agent_id:
                                if round_num > 0 and len(agent_nodes[other_agent_id]) >= round_num:
                                    source_node = agent_nodes[other_agent_id][round_num - 1]
                                    # Calculate influence weight based on answer agreement
                                    source_answer = self.graph.nodes[source_node].get('answer')
                                    target_answer = answer

                                    influence_weight = 0.5  # Default
                                    if source_answer == target_answer:
                                        influence_weight = 1.0  # Strong agreement
                                    elif source_answer and target_answer:
                                        influence_weight = 0.3  # Different answers

                                    self.graph.add_edge(
                                        source_node,
                                        node_id,
                                        edge_type='influence',
                                        weight=influence_weight,
                                        round_diff=1
                                    )

                prev_node_id = node_id
                round_num += 1

        return self.graph

    def calculate_graph_metrics(self) -> Dict:
        """Calculate interesting metrics about the debate graph."""
        metrics = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'n_agents': len(set(nx.get_node_attributes(self.graph, 'agent_id').values())),
            'n_rounds': max(nx.get_node_attributes(self.graph, 'round').values()) + 1 if self.graph.nodes else 0,
        }

        # Convergence analysis
        answers_by_round = defaultdict(list)
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('answer'):
                answers_by_round[attrs['round']].append(attrs['answer'])

        # Calculate consensus per round
        consensus_by_round = {}
        for round_num, answers in answers_by_round.items():
            unique_answers = set(answers)
            consensus_by_round[round_num] = 1.0 / len(unique_answers) if unique_answers else 0

        metrics['consensus_by_round'] = consensus_by_round

        # Accuracy per round
        correct_by_round = defaultdict(list)
        for node, attrs in self.graph.nodes(data=True):
            correct_by_round[attrs['round']].append(attrs.get('is_correct', False))

        accuracy_by_round = {
            r: np.mean(correct) if correct else 0
            for r, correct in correct_by_round.items()
        }
        metrics['accuracy_by_round'] = accuracy_by_round

        # Influence centrality (who influenced others most)
        influence_scores = defaultdict(float)
        for source, target, attrs in self.graph.edges(data=True):
            if attrs.get('edge_type') == 'influence':
                source_agent = self.graph.nodes[source]['agent_id']
                influence_scores[source_agent] += attrs.get('weight', 0.5)

        metrics['influence_scores'] = dict(influence_scores)

        return metrics


class GraphBasedUQ:
    """
    Graph-Based Uncertainty Quantification for Multi-Agent Debates.

    Adapts UProp framework to debate graphs where:
    - Each agent's trajectory across rounds is treated as a TDP
    - Intrinsic uncertainty: Answer diversity at each round
    - Extrinsic uncertainty: Cross-agent trajectory divergence (PMI-based)
    - No logprobs available: Use semantic distance between answers
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def calculate_answer_distance(self, answer1: str, answer2: str) -> float:
        """
        Calculate semantic distance between two answers.
        Uses fuzzy string matching as proxy for semantic similarity.

        Returns: Distance in [0, 1] where 0 = identical, 1 = completely different
        """
        if answer1 is None or answer2 is None:
            return 1.0  # Maximum distance if either is missing

        # Use fuzzy string matching
        similarity = fuzz.ratio(str(answer1), str(answer2)) / 100.0
        return 1.0 - similarity

    def calculate_intrinsic_uncertainty_round(self, round_num: int) -> float:
        """
        Calculate intrinsic uncertainty at a specific round.

        IU(round_t) = H(answers_t) = diversity of answers among agents

        Uses normalized entropy of answer distribution:
        - High entropy: agents disagree (high uncertainty)
        - Low entropy: agents agree (low uncertainty)
        """
        # Get all answers at this round
        answers = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs['round'] == round_num and attrs.get('answer'):
                answers.append(attrs['answer'])

        if not answers:
            return 0.0

        # Calculate answer distribution
        answer_counts = defaultdict(int)
        for answer in answers:
            answer_counts[answer] += 1

        # Calculate entropy
        total = len(answers)
        probs = np.array([count / total for count in answer_counts.values()])

        # Normalized entropy (divide by log(n) to get value in [0,1])
        max_entropy = np.log(len(answer_counts)) if len(answer_counts) > 1 else 1.0
        iu = entropy(probs) / max_entropy if max_entropy > 0 else 0.0

        return iu

    def calculate_trajectory_pmi(self, agent1_id: int, agent2_id: int,
                                  round_t: int, round_i: int) -> float:
        """
        Calculate PMI between two agent trajectories at specific rounds.

        PMI(y_t^{agent1}; y_i^{agent2}) measures dependency between:
        - Agent1's answer at round t
        - Agent2's answer at round i (i < t)

        Approximation: PMI ∝ distance between trajectory states
        - Small distance → high dependency → high PMI
        - Large distance → low dependency → low PMI
        """
        # Get agent nodes at specified rounds
        agent1_node = None
        agent2_node = None

        for node, attrs in self.graph.nodes(data=True):
            if attrs['agent_id'] == agent1_id and attrs['round'] == round_t:
                agent1_node = node
            if attrs['agent_id'] == agent2_id and attrs['round'] == round_i:
                agent2_node = node

        if not agent1_node or not agent2_node:
            return 0.0

        # Get answers
        answer1 = self.graph.nodes[agent1_node].get('answer')
        answer2 = self.graph.nodes[agent2_node].get('answer')

        # Calculate distance
        dist = self.calculate_answer_distance(answer1, answer2)

        # Convert distance to PMI using Gaussian kernel
        # PMI ∝ -log(K(dist)) where K is Gaussian kernel
        # K(x) = exp(-x²/2σ²), use σ=0.5 for moderate spread
        sigma = 0.5
        kernel_val = np.exp(-dist**2 / (2 * sigma**2))

        # Add small constant to avoid log(0)
        epsilon = 1e-10
        pmi = -np.log(kernel_val + epsilon)

        return pmi

    def calculate_extrinsic_uncertainty_round(self, round_t: int) -> float:
        """
        Calculate extrinsic uncertainty for a specific round.

        EU(round_t) = Σ_{agents} Σ_{i<t} PMI(agent_trajectory_t; other_trajectories_i)

        Measures how much current round depends on previous rounds' trajectories
        across all agents (cross-agent dependencies).
        """
        if round_t == 0:
            return 0.0

        n_agents = len(set(nx.get_node_attributes(self.graph, 'agent_id').values()))

        total_eu = 0.0
        pmi_count = 0

        # For each agent at round t
        for agent_t in range(n_agents):
            # Sum PMI with all agents at all previous rounds
            for round_i in range(round_t):
                for agent_i in range(n_agents):
                    pmi = self.calculate_trajectory_pmi(agent_t, agent_i, round_t, round_i)
                    total_eu += pmi
                    pmi_count += 1

        # Average PMI to normalize
        if pmi_count > 0:
            total_eu = total_eu / pmi_count

        return total_eu

    def calculate_total_uncertainty(self) -> Dict[str, float]:
        """
        Calculate total graph uncertainty with decomposition.

        Returns:
        - total_uncertainty: Normalized sum of IU + EU across all rounds
        - intrinsic_uncertainty: Average IU (answer diversity)
        - extrinsic_uncertainty: Average EU (trajectory dependencies)
        - round_uncertainties: Per-round breakdown
        - convergence_score: How much uncertainty decreases (1 = full convergence)
        """
        n_rounds = max(nx.get_node_attributes(self.graph, 'round').values()) + 1 if self.graph.nodes else 0

        if n_rounds == 0:
            return {
                'total_uncertainty': 0.0,
                'intrinsic_uncertainty': 0.0,
                'extrinsic_uncertainty': 0.0,
                'round_uncertainties': [],
                'convergence_score': 1.0
            }

        round_uncertainties = []
        total_iu = 0.0
        total_eu = 0.0
        step_weights = []

        for round_num in range(n_rounds):
            # Calculate IU and EU for this round
            iu = self.calculate_intrinsic_uncertainty_round(round_num)
            eu = self.calculate_extrinsic_uncertainty_round(round_num)

            # Step weight for normalization: σ_t = 1 + EU/IU
            if iu > 0:
                sigma_t = 1 + (eu / iu)
            else:
                sigma_t = 1.0

            step_weights.append(sigma_t)

            round_uncertainty = {
                'round': round_num,
                'iu': iu,
                'eu': eu,
                'total': iu + eu,
                'weight': sigma_t
            }
            round_uncertainties.append(round_uncertainty)

            total_iu += iu
            total_eu += eu

        # Apply step length normalization (λ = Σ σ_t)
        lambda_z = sum(step_weights)

        # Calculate normalized total uncertainty
        total_uncertainty = (total_iu + total_eu) / lambda_z if lambda_z > 0 else 0.0

        # Average uncertainties
        avg_iu = total_iu / n_rounds
        avg_eu = total_eu / n_rounds

        # Convergence score: how much uncertainty decreased from first to last round
        if round_uncertainties:
            first_iu = round_uncertainties[0]['iu']
            last_iu = round_uncertainties[-1]['iu']
            if first_iu > 0:
                convergence_score = 1.0 - (last_iu / first_iu)
            else:
                convergence_score = 1.0  # Already converged from start
            convergence_score = max(0.0, min(1.0, convergence_score))  # Clip to [0,1]
        else:
            convergence_score = 1.0

        return {
            'total_uncertainty': total_uncertainty,
            'intrinsic_uncertainty': avg_iu,
            'extrinsic_uncertainty': avg_eu,
            'round_uncertainties': round_uncertainties,
            'convergence_score': convergence_score,
            'normalization_factor': lambda_z,
            'n_rounds': n_rounds
        }

    def calculate_agent_uncertainty_profile(self) -> Dict[int, Dict]:
        """
        Calculate per-agent uncertainty profiles.

        For each agent, measures:
        - Trajectory stability: How much their answers change across rounds
        - Influence susceptibility: How much they're influenced by others
        - Influence strength: How much they influence others
        """
        n_agents = len(set(nx.get_node_attributes(self.graph, 'agent_id').values()))
        n_rounds = max(nx.get_node_attributes(self.graph, 'round').values()) + 1 if self.graph.nodes else 0

        agent_profiles = {}

        for agent_id in range(n_agents):
            # Get agent's trajectory (answers across rounds)
            trajectory = []
            for round_num in range(n_rounds):
                for node, attrs in self.graph.nodes(data=True):
                    if attrs['agent_id'] == agent_id and attrs['round'] == round_num:
                        trajectory.append(attrs.get('answer'))
                        break

            # Calculate trajectory stability (how much answers change)
            stability_scores = []
            for i in range(1, len(trajectory)):
                if trajectory[i] and trajectory[i-1]:
                    dist = self.calculate_answer_distance(trajectory[i-1], trajectory[i])
                    stability_scores.append(1.0 - dist)  # High stability = low change

            avg_stability = np.mean(stability_scores) if stability_scores else 1.0

            # Calculate influence susceptibility (incoming influence edges)
            incoming_influence = 0.0
            incoming_count = 0
            for source, target, attrs in self.graph.edges(data=True):
                if attrs.get('edge_type') == 'influence':
                    target_agent = self.graph.nodes[target]['agent_id']
                    if target_agent == agent_id:
                        incoming_influence += attrs.get('weight', 0.5)
                        incoming_count += 1

            susceptibility = incoming_influence / incoming_count if incoming_count > 0 else 0.0

            # Calculate influence strength (outgoing influence edges)
            outgoing_influence = 0.0
            outgoing_count = 0
            for source, target, attrs in self.graph.edges(data=True):
                if attrs.get('edge_type') == 'influence':
                    source_agent = self.graph.nodes[source]['agent_id']
                    if source_agent == agent_id:
                        outgoing_influence += attrs.get('weight', 0.5)
                        outgoing_count += 1

            influence_strength = outgoing_influence / outgoing_count if outgoing_count > 0 else 0.0

            agent_profiles[agent_id] = {
                'trajectory_stability': avg_stability,
                'influence_susceptibility': susceptibility,
                'influence_strength': influence_strength,
                'trajectory': trajectory
            }

        return agent_profiles


class DebateDAGUncertainty:
    """
    Integrates DAG-based uncertainty propagation from dag.py
    into debate graph analysis.

    Converts debate graph structure to DPIMPR-compatible format
    and computes DAG uncertainty metrics.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def convert_debate_to_trajectories(self, ground_truth_answer: str = None) -> List[Dict]:
        """
        Convert debate graph to trajectory format compatible with DPIMPR.

        Each agent's path through rounds becomes a trajectory.
        Critical: Final round must use Finish[answer] format for proper belief extraction.

        Args:
            ground_truth_answer: The correct answer for belief state matching
        """
        n_agents = len(set(nx.get_node_attributes(self.graph, 'agent_id').values()))
        n_rounds = max(nx.get_node_attributes(self.graph, 'round').values()) + 1 if self.graph.nodes else 0

        trajectories = []

        for agent_id in range(n_agents):
            trajectory_steps = []

            for round_num in range(n_rounds):
                # Find node for this agent at this round
                for node, attrs in self.graph.nodes(data=True):
                    if attrs['agent_id'] == agent_id and attrs['round'] == round_num:
                        # Convert to trajectory step format
                        answer = str(attrs.get('answer', ''))
                        content = attrs.get('content', '')

                        # CRITICAL: Determine node type - ONLY final round is conclusion
                        if round_num == n_rounds - 1:
                            node_type = 'conclusion'
                            # Use Finish[answer] format for conclusion nodes
                            # This is what dag.py's _extract_belief_state() parses
                            action = f"Finish[{answer}]"
                            reasoning = f"Final answer: {answer} after debate round {round_num}"
                        else:
                            node_type = 'reasoning'
                            # Use Search[answer] for intermediate rounds
                            action = f"Search[{answer}]"
                            reasoning = content[:100] if content else f"Debate round {round_num}"

                        # Better log_prob estimation
                        # Later rounds = higher confidence (less negative)
                        log_prob = -2.0 + (round_num * 0.4)

                        step = {
                            'step': round_num,
                            'action': action,
                            'reasoning': reasoning,
                            'observation': f"Agent {agent_id} answer: {answer}",
                            'log_prob': log_prob,
                            'node_type': node_type
                        }

                        trajectory_steps.append(step)
                        break

            if trajectory_steps:
                trajectories.append({
                    'trajectory': trajectory_steps,
                    'samples': {}  # No alternative samples in debate setting
                })

        return trajectories

    def compute_dag_uncertainty(self, question: str = None, ground_truth_answer: str = None) -> Dict[str, float]:
        """
        Compute DAG-based uncertainty metrics using DPIMPR algorithm.

        Args:
            question: Question being debated
            ground_truth_answer: The final answer for belief state extraction

        Returns:
            Dictionary with DAG uncertainty metrics including proper confidence scores
        """
        # Convert debate graph to trajectories with ground truth
        trajectories = self.convert_debate_to_trajectories(ground_truth_answer=ground_truth_answer)

        if not trajectories:
            return {
                'dag_confidence': 0.0,
                'dag_avg_node_uncertainty': 1.0,
                'dag_conclusion_uncertainty': 1.0,
                'dag_path_diversity': 0.0,
                'dag_evidence_strength': 0.0,
                'dag_topological_depth': 0.0,
                'dag_uncertainty_confidence': 1.0,
                'dag_uncertainty_evidence': 1.0,
                'dag_uncertainty_diversity': 1.0
            }

        # Initialize DPIMPR with debate-friendly settings (suppress output)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()  # Suppress DPIMPR debug output

        try:
            dpimpr = DPIMPR(
                similarity_threshold=0.85,  # High threshold for debate (same answers should merge)
                topological_distance=10,     # Allow moderate distance for merging
                question=question
            )

            # Add trajectories to DPIMPR
            results = dpimpr.add_tdp_data(trajectories, question=question)

            # Restore stdout early to see any errors
            sys.stdout = old_stdout

            # Extract DAG uncertainty metrics
            dag_metrics = dpimpr.extract_dag_uncertainty_metrics()

            # CRITICAL: Compute DAG confidence with target conclusion
            # The ground_truth_answer helps identify which conclusion to evaluate
            confidence_result = dpimpr.compute_dag_confidence(target_conclusion=ground_truth_answer)

            # Update metrics with proper confidence
            dag_metrics['dag_confidence'] = confidence_result.get('confidence', 0.0)
            dag_metrics['dag_final_conclusion'] = confidence_result.get('conclusion', 'unknown')
            dag_metrics['dag_topological_depth'] = confidence_result.get('topological_depth', 0)

            # Add DAG analysis info
            dag_metrics['dag_node_count'] = results['dag_analysis']['node_count']
            dag_metrics['dag_edge_count'] = results['dag_analysis']['edge_count']
            dag_metrics['dag_convergence_points'] = len(results['dag_analysis']['convergence_points'])

            return dag_metrics

        except Exception as e:
            sys.stdout = old_stdout  # Restore on error
            print(f"Warning: DAG uncertainty computation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'dag_confidence': 0.0,
                'dag_avg_node_uncertainty': 1.0,
                'dag_error': str(e)
            }


class DebateGraphVisualizer:
    """Visualizes debate graphs with custom layouts and styling."""

    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.agent_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    def create_round_based_layout(self, graph: nx.DiGraph) -> Dict:
        """Create a layout where nodes are arranged by rounds (time-based)."""
        pos = {}

        # Group nodes by round
        rounds = defaultdict(list)
        for node, attrs in graph.nodes(data=True):
            rounds[attrs['round']].append(node)

        max_round = max(rounds.keys()) if rounds else 0

        # Position nodes
        for round_num, nodes in rounds.items():
            n_nodes = len(nodes)
            x = round_num / (max_round + 1) if max_round > 0 else 0.5

            for i, node in enumerate(sorted(nodes)):
                y = (i + 1) / (n_nodes + 1)
                pos[node] = (x, y)

        return pos

    def create_circular_per_round_layout(self, graph: nx.DiGraph) -> Dict:
        """Create layout with agents in circles per round."""
        pos = {}

        # Group nodes by round and agent
        rounds = defaultdict(lambda: defaultdict(list))
        for node, attrs in graph.nodes(data=True):
            rounds[attrs['round']][attrs['agent_id']].append(node)

        max_round = max(rounds.keys()) if rounds else 0
        n_agents = len(set(nx.get_node_attributes(graph, 'agent_id').values()))

        # Position nodes
        for round_num, agents in rounds.items():
            radius = 3 + round_num * 2  # Expanding circles
            angle_step = 2 * np.pi / max(n_agents, 1)

            for agent_id, nodes in agents.items():
                angle = agent_id * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                for node in nodes:
                    pos[node] = (x, y)

        return pos

    def visualize_debate_graph(
        self,
        graph: nx.DiGraph,
        metrics: Dict,
        uq_results: Optional[Dict] = None,
        layout: str = 'round',
        output_path: Optional[str] = None,
        title: str = "Multi-Agent Debate Graph"
    ):
        """
        Visualize the debate graph with color-coded nodes and edges.

        Args:
            graph: NetworkX graph to visualize
            metrics: Dictionary of graph metrics
            uq_results: Uncertainty quantification results (optional)
            layout: Layout style ('round' or 'circular')
            output_path: Path to save figure (optional)
            title: Graph title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Create layout
        if layout == 'round':
            pos = self.create_round_based_layout(graph)
        else:
            pos = self.create_circular_per_round_layout(graph)

        # Extract node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node, attrs in graph.nodes(data=True):
            agent_id = attrs['agent_id']
            is_correct = attrs.get('is_correct', False)

            # Color by agent, brightness by correctness
            base_color = self.agent_colors[agent_id % len(self.agent_colors)]
            node_colors.append(base_color if is_correct else '#CCCCCC')

            # Size by round (later rounds slightly larger)
            node_sizes.append(300 + attrs['round'] * 50)

            # Label with agent and answer
            answer = attrs.get('answer', '?')
            node_labels[node] = f"A{agent_id}\n{answer}"

        # Extract edge attributes
        edge_colors = []
        edge_widths = []
        edge_styles = []

        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'unknown')
            weight = attrs.get('weight', 0.5)

            if edge_type == 'self':
                edge_colors.append('#666666')
                edge_widths.append(1.0)
                edge_styles.append('solid')
            else:  # influence
                # Color by weight
                alpha = weight
                edge_colors.append(f'rgba(100, 100, 250, {alpha})')
                edge_widths.append(weight * 3)
                edge_styles.append('dashed')

        # Draw graph
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax1,
            alpha=0.8
        )

        # Draw self edges (solid)
        self_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'self']
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=self_edges,
            edge_color='#666666',
            width=1.0,
            style='solid',
            ax=ax1,
            alpha=0.5,
            arrows=True,
            arrowsize=10
        )

        # Draw influence edges (dashed)
        influence_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'influence']
        influence_weights = [graph[u][v]['weight'] * 3 for u, v in influence_edges]
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=influence_edges,
            edge_color='blue',
            width=influence_weights,
            style='dashed',
            ax=ax1,
            alpha=0.3,
            arrows=True,
            arrowsize=10
        )

        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=8,
            ax=ax1
        )

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Plot metrics
        ax2.axis('off')

        # Create metrics text
        metrics_text = "Debate Metrics\n" + "="*40 + "\n\n"
        metrics_text += f"Nodes: {metrics['n_nodes']}\n"
        metrics_text += f"Edges: {metrics['n_edges']}\n"
        metrics_text += f"Agents: {metrics['n_agents']}\n"
        metrics_text += f"Rounds: {metrics['n_rounds']}\n\n"

        # Add UQ metrics if available
        if uq_results:
            metrics_text += "Graph-Based UQ\n" + "-"*40 + "\n"
            metrics_text += f"Total Uncertainty: {uq_results['total_uncertainty']:.4f}\n"
            metrics_text += f"  Intrinsic (IU): {uq_results['intrinsic_uncertainty']:.4f}\n"
            metrics_text += f"  Extrinsic (EU): {uq_results['extrinsic_uncertainty']:.4f}\n"
            metrics_text += f"Convergence: {uq_results['convergence_score']:.2%}\n\n"

            # Add DAG UQ metrics if available
            if 'dag_confidence' in uq_results:
                metrics_text += "DAG Uncertainty Propagation\n" + "-"*40 + "\n"
                metrics_text += f"DAG Confidence: {uq_results.get('dag_confidence', 0):.4f}\n"
                metrics_text += f"Avg Node Uncert: {uq_results.get('dag_avg_node_uncertainty', 0):.4f}\n"
                metrics_text += f"Evidence: {uq_results.get('dag_evidence_strength', 0):.2f}\n"
                metrics_text += f"Path Diversity: {uq_results.get('dag_path_diversity', 0):.0f}\n\n"

            metrics_text += "UQ by Round:\n"
            if 'round_uncertainties' in uq_results:
                for ru in uq_results['round_uncertainties']:
                    metrics_text += f"  R{ru['round']}: IU={ru['iu']:.3f}, EU={ru['eu']:.3f}\n"
            metrics_text += "\n"

        metrics_text += "Accuracy by Round:\n"
        for round_num, acc in sorted(metrics['accuracy_by_round'].items()):
            metrics_text += f"  Round {round_num}: {acc:.2%}\n"

        metrics_text += "\nConsensus by Round:\n"
        for round_num, cons in sorted(metrics['consensus_by_round'].items()):
            metrics_text += f"  Round {round_num}: {cons:.2%}\n"

        if metrics['influence_scores']:
            metrics_text += "\nInfluence Scores:\n"
            for agent_id, score in sorted(metrics['influence_scores'].items()):
                metrics_text += f"  Agent {agent_id}: {score:.2f}\n"

        ax2.text(0.1, 0.5, metrics_text,
                fontsize=9,
                verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=10,
                      label=f'Agent {i}')
            for i, color in enumerate(self.agent_colors[:metrics['n_agents']])
        ]
        legend_elements.extend([
            plt.Line2D([0], [0], color='#666666', linewidth=2, label='Self-edge'),
            plt.Line2D([0], [0], color='blue', linewidth=2,
                      linestyle='dashed', label='Influence'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='green', markersize=10, label='Correct'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#CCCCCC', markersize=10, label='Incorrect'),
        ])

        ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")

        return fig


def process_debate_traces(
    traces_dir: str = "../results/debate",
    output_dir: str = "../results/debate_vis",
    layout: str = 'round'
):
    """
    Process all debate trace files and create visualizations.

    Args:
        traces_dir: Directory containing debate trace JSON files
        output_dir: Directory to save visualizations
        layout: Layout style for graphs
    """
    os.makedirs(output_dir, exist_ok=True)

    builder = DebateGraphBuilder()
    visualizer = DebateGraphVisualizer()

    # Find all trace files
    trace_files = sorted([
        f for f in os.listdir(traces_dir)
        if f.startswith('traces_') and f.endswith('.json')
    ])

    print(f"Found {len(trace_files)} trace files")

    all_metrics = []

    for trace_file in trace_files:
        print(f"\nProcessing {trace_file}...")

        # Load trace data
        trace_path = os.path.join(traces_dir, trace_file)
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)

        # Build graph
        graph = builder.build_graph_from_trace(trace_data)

        # Calculate metrics
        metrics = builder.calculate_graph_metrics()
        metrics['trace_file'] = trace_file

        # Calculate graph-based uncertainty quantification
        uq_calculator = GraphBasedUQ(graph)
        uq_results = uq_calculator.calculate_total_uncertainty()
        agent_profiles = uq_calculator.calculate_agent_uncertainty_profile()

        # Calculate DAG-based uncertainty propagation
        dag_uq_calculator = DebateDAGUncertainty(graph)
        dag_uq_results = dag_uq_calculator.compute_dag_uncertainty(
            question=f"Debate on answer: {trace_data.get('answer', 'unknown')}",
            ground_truth_answer=str(trace_data.get('answer', 'unknown'))
        )

        # Merge UQ results
        combined_uq_results = {**uq_results, **dag_uq_results}

        # Store UQ results in metrics
        metrics['uq_results'] = combined_uq_results
        metrics['agent_profiles'] = agent_profiles
        all_metrics.append(metrics)

        # Visualize
        trace_id = trace_file.replace('traces_', '').replace('.json', '')
        output_path = os.path.join(output_dir, f'debate_graph_{trace_id}.png')

        title = f"Debate Trace {trace_id} - Answer: {trace_data['answer']}"
        visualizer.visualize_debate_graph(
            graph, metrics, uq_results=combined_uq_results, layout=layout,
            output_path=output_path, title=title
        )

        plt.close()

        print(f"  Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
        print(f"  Final accuracy: {metrics['accuracy_by_round'].get(metrics['n_rounds']-1, 0):.2%}")
        print(f"  Graph UQ - Total: {uq_results['total_uncertainty']:.4f} (IU: {uq_results['intrinsic_uncertainty']:.4f}, EU: {uq_results['extrinsic_uncertainty']:.4f})")
        print(f"  Graph UQ - Convergence: {uq_results['convergence_score']:.2%}")
        if 'dag_confidence' in dag_uq_results:
            print(f"  DAG UQ - Confidence: {dag_uq_results.get('dag_confidence', 0):.4f}, Avg Uncertainty: {dag_uq_results.get('dag_avg_node_uncertainty', 0):.4f}")
            print(f"  DAG UQ - Evidence: {dag_uq_results.get('dag_evidence_strength', 0):.2f}, Diversity: {dag_uq_results.get('dag_path_diversity', 0):.0f}")

    # Save aggregate metrics
    metrics_path = os.path.join(output_dir, 'debate_metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ Processed {len(trace_files)} debates")
    print(f"✓ Visualizations saved to {output_dir}")
    print(f"✓ Metrics summary saved to {metrics_path}")

    return all_metrics


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize multi-agent debate traces')
    parser.add_argument(
        '--traces_dir',
        type=str,
        default='../results/debate',
        help='Directory containing debate trace JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results/debate_vis',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['round', 'circular'],
        default='round',
        help='Graph layout style'
    )

    args = parser.parse_args()

    metrics = process_debate_traces(
        traces_dir=args.traces_dir,
        output_dir=args.output_dir,
        layout=args.layout
    )

    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    avg_accuracy = np.mean([
        list(m['accuracy_by_round'].values())[-1]
        for m in metrics if m['accuracy_by_round']
    ])
    print(f"Average final accuracy: {avg_accuracy:.2%}")

    avg_consensus = np.mean([
        list(m['consensus_by_round'].values())[-1]
        for m in metrics if m['consensus_by_round']
    ])
    print(f"Average final consensus: {avg_consensus:.2%}")

    # Uncertainty statistics
    if all('uq_results' in m for m in metrics):
        avg_total_uncertainty = np.mean([m['uq_results']['total_uncertainty'] for m in metrics])
        avg_intrinsic = np.mean([m['uq_results']['intrinsic_uncertainty'] for m in metrics])
        avg_extrinsic = np.mean([m['uq_results']['extrinsic_uncertainty'] for m in metrics])
        avg_convergence = np.mean([m['uq_results']['convergence_score'] for m in metrics])

        print(f"\nUncertainty Quantification:")
        print(f"  Average total uncertainty: {avg_total_uncertainty:.4f}")
        print(f"  Average intrinsic uncertainty: {avg_intrinsic:.4f}")
        print(f"  Average extrinsic uncertainty: {avg_extrinsic:.4f}")
        print(f"  Average convergence score: {avg_convergence:.2%}")


def evaluate_debate_auroc(metrics_summary_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate AUROC for uncertainty metrics across all debate traces.

    Args:
        metrics_summary_path: Path to debate_metrics_summary.json
        output_dir: Directory to save AUROC plots and tables

    Returns:
        Dictionary with AUROC scores and evaluation results
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    import pandas as pd

    print("\n" + "="*70)
    print("DEBATE UNCERTAINTY QUANTIFICATION - AUROC EVALUATION")
    print("="*70)

    # Load metrics
    with open(metrics_summary_path, 'r') as f:
        all_metrics = json.load(f)

    print(f"Loaded {len(all_metrics)} debate traces")

    # Extract correctness labels (binary: correct=1, incorrect=0)
    correctness_labels = []
    uncertainty_metrics = defaultdict(list)

    for m in all_metrics:
        # Get final round accuracy as correctness label
        final_round = m['n_rounds'] - 1
        final_accuracy = m['accuracy_by_round'].get(str(final_round), 0)

        # Binary correctness: 1 if any agent correct, 0 if all incorrect
        is_correct = int(final_accuracy > 0)
        correctness_labels.append(is_correct)

        # Extract all uncertainty metrics
        if 'uq_results' in m:
            uq = m['uq_results']

            # Graph-based UQ metrics
            uncertainty_metrics['graph_total_uncertainty'].append(uq.get('total_uncertainty', 0))
            uncertainty_metrics['graph_intrinsic_uncertainty'].append(uq.get('intrinsic_uncertainty', 0))
            uncertainty_metrics['graph_extrinsic_uncertainty'].append(uq.get('extrinsic_uncertainty', 0))
            uncertainty_metrics['graph_convergence_score'].append(uq.get('convergence_score', 0))

            # DAG-based UQ metrics
            uncertainty_metrics['dag_confidence'].append(uq.get('dag_confidence', 0))
            uncertainty_metrics['dag_avg_node_uncertainty'].append(uq.get('dag_avg_node_uncertainty', 0))
            uncertainty_metrics['dag_conclusion_uncertainty'].append(uq.get('dag_conclusion_uncertainty', 0))
            uncertainty_metrics['dag_evidence_strength'].append(uq.get('dag_evidence_strength', 0))
            uncertainty_metrics['dag_path_diversity'].append(uq.get('dag_path_diversity', 0))
            uncertainty_metrics['dag_uncertainty_confidence'].append(uq.get('dag_uncertainty_dag_confidence', 0))
            uncertainty_metrics['dag_uncertainty_evidence'].append(uq.get('dag_uncertainty_evidence_strength', 0))

    correctness_array = np.array(correctness_labels)
    n_correct = np.sum(correctness_array)
    n_incorrect = len(correctness_array) - n_correct

    print(f"Correct answers: {n_correct}/{len(correctness_array)} ({n_correct/len(correctness_array):.1%})")
    print(f"Incorrect answers: {n_incorrect}/{len(correctness_array)} ({n_incorrect/len(correctness_array):.1%})")

    if n_correct == 0 or n_incorrect == 0:
        print("\n⚠ Warning: Need both correct and incorrect examples to compute AUROC")
        return {}

    # Calculate AUROC for each metric
    print(f"\nCalculating AUROC scores for {len(uncertainty_metrics)} metrics...")

    auroc_results = {}
    roc_curves = {}

    for metric_name, values in uncertainty_metrics.items():
        if len(values) != len(correctness_labels):
            continue

        values_array = np.array(values)

        # Determine if metric should be negated
        # Higher uncertainty/lower confidence should predict incorrectness
        if 'confidence' in metric_name or 'convergence' in metric_name or 'evidence' in metric_name:
            # These are "good" metrics - higher values = more confident/correct
            # Negate for AUROC (we want higher scores to predict incorrectness)
            scores = -values_array
        else:
            # These are "bad" metrics - higher values = more uncertain/incorrect
            scores = values_array

        try:
            # AUROC: how well does uncertainty predict incorrectness?
            # Higher uncertainty should predict incorrect answers
            auroc = roc_auc_score(1 - correctness_array, scores)  # Predict incorrectness
            auroc_results[metric_name] = auroc

            # Store ROC curve data
            fpr, tpr, thresholds = roc_curve(1 - correctness_array, scores)
            roc_curves[metric_name] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

        except ValueError as e:
            print(f"  ⚠ Could not compute AUROC for {metric_name}: {e}")

    # Sort by AUROC score
    sorted_results = sorted(auroc_results.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("AUROC SCORES (Predicting Incorrect Answers)")
    print("="*70)
    print(f"{'Metric':<45} {'AUROC':>10} {'Quality':>10}")
    print("-"*70)

    for metric_name, auroc in sorted_results:
        quality = "Excellent" if auroc >= 0.9 else "Good" if auroc >= 0.8 else "Fair" if auroc >= 0.7 else "Poor"
        print(f"{metric_name:<45} {auroc:>10.4f} {quality:>10}")

    # Create visualizations if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Plot ROC curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Top Graph-based UQ metrics
        ax1 = axes[0, 0]
        graph_metrics = [(k, v) for k, v in sorted_results if k.startswith('graph_')][:4]
        for metric_name, auroc in graph_metrics:
            if metric_name in roc_curves:
                roc_data = roc_curves[metric_name]
                label = f"{metric_name.replace('graph_', '')}: {auroc:.3f}"
                ax1.plot(roc_data['fpr'], roc_data['tpr'], label=label, linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (0.500)')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('Graph-Based UQ Metrics - ROC Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Top DAG-based UQ metrics
        ax2 = axes[0, 1]
        dag_metrics = [(k, v) for k, v in sorted_results if k.startswith('dag_')][:4]
        for metric_name, auroc in dag_metrics:
            if metric_name in roc_curves:
                roc_data = roc_curves[metric_name]
                label = f"{metric_name.replace('dag_', '')}: {auroc:.3f}"
                ax2.plot(roc_data['fpr'], roc_data['tpr'], label=label, linewidth=2)

        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (0.500)')
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('DAG-Based UQ Metrics - ROC Curves', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: All metrics comparison (bar plot)
        ax3 = axes[1, 0]
        metric_names = [m.replace('graph_', 'G:').replace('dag_', 'D:') for m, _ in sorted_results]
        auroc_values = [auroc for _, auroc in sorted_results]
        colors = ['#FF6B6B' if m.startswith('graph_') else '#4ECDC4' for m, _ in sorted_results]

        bars = ax3.barh(range(len(metric_names)), auroc_values, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(metric_names)))
        ax3.set_yticklabels(metric_names, fontsize=9)
        ax3.set_xlabel('AUROC Score', fontsize=12)
        ax3.set_title('All Metrics - AUROC Comparison', fontsize=14, fontweight='bold')
        ax3.axvline(x=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Random')
        ax3.set_xlim(0, 1)
        ax3.grid(True, axis='x', alpha=0.3)
        ax3.legend()

        # Plot 4: Best overall metric ROC curve
        ax4 = axes[1, 1]
        best_metric, best_auroc = sorted_results[0]
        if best_metric in roc_curves:
            roc_data = roc_curves[best_metric]
            ax4.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=3,
                    label=f'{best_metric}: {best_auroc:.4f}')
            ax4.fill_between(roc_data['fpr'], roc_data['tpr'], alpha=0.3)

        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (0.500)')
        ax4.set_xlabel('False Positive Rate', fontsize=12)
        ax4.set_ylabel('True Positive Rate', fontsize=12)
        ax4.set_title(f'Best Metric: {best_metric}', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, 'debate_auroc_curves.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ ROC curves saved to {fig_path}")
        plt.close()

        # Create results table
        df = pd.DataFrame([
            {
                'Metric': metric_name,
                'AUROC': auroc,
                'Category': 'Graph-Based' if metric_name.startswith('graph_') else 'DAG-Based'
            }
            for metric_name, auroc in sorted_results
        ])

        # Save as CSV
        csv_path = os.path.join(output_dir, 'debate_auroc_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Results table saved to {csv_path}")

        # Create pretty table visualization
        fig_table, ax_table = plt.subplots(figsize=(12, 8))
        ax_table.axis('off')

        # Prepare table data
        table_data = []
        table_data.append(['Rank', 'Metric', 'AUROC', 'Category'])
        for i, (metric_name, auroc) in enumerate(sorted_results, 1):
            category = 'Graph-Based' if metric_name.startswith('graph_') else 'DAG-Based'
            table_data.append([str(i), metric_name, f"{auroc:.4f}", category])

        # Create table
        table = ax_table.table(cellText=table_data, cellLoc='left',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color alternate rows
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Debate UQ Metrics - AUROC Rankings',
                 fontsize=16, fontweight='bold', pad=20)

        table_path = os.path.join(output_dir, 'debate_auroc_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results table image saved to {table_path}")
        plt.close()

    # Return results
    return {
        'auroc_scores': auroc_results,
        'sorted_results': sorted_results,
        'roc_curves': roc_curves,
        'n_traces': len(all_metrics),
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect),
        'best_metric': sorted_results[0] if sorted_results else (None, 0)
    }


if __name__ == "__main__":
    main()
