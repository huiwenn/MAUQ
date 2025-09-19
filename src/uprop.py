"""
UProp: Uncertainty Propagation for LLMs in Multi-Step Agentic Decision-Making
Implementation based on the paper by Duan et al. (2025)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
from collections import defaultdict
from scipy.stats import entropy
from scipy.spatial.distance import cdist
import logging
from fuzzywuzzy import fuzz  # For string distance measurement
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Decision:
    """Represents a single decision in the trajectory"""
    step: int
    action: str  # The actual action taken
    reasoning: str  # The reasoning behind the action
    observation: str  # The observation received after the action
    log_prob: float  # Log probability of this decision

    def data_dump(self):
        """Dump the decision to a dictionary"""
        return {
            'step': self.step,
            'action': self.action,
            'reasoning': self.reasoning,
            'observation': self.observation,
            'log_prob': self.log_prob
        }
    
@dataclass
class TDP:
    """Trajectory-Dependent Decision Process"""
    trajectory: List[Decision]  # The main trajectory (y_k decisions)
    samples: Dict[int, List[Decision]]  # Alternative samples at each step

    def data_dump(self):
        """Save the TDP to a file"""
        data = {
            'trajectory': [decision.data_dump() for decision in self.trajectory],
            'samples': {step: [decision.data_dump() for decision in decisions] for step, decisions in self.samples.items()}
        }
        return data
    
class UProp:
    """
    Implementation of the UProp algorithm for uncertainty quantification
    in multi-step LLM decision making.
    """
    
    def __init__(
        self,
        llm_client: Any,  # OpenAI client or similar
        Z: int = 3,  # Number of TDP samples
        N: int = 3,  # Number of per-step samples
        temperature: float = 0.8,  # Sampling temperature
        max_steps: int = 10,  # Maximum decision steps
        distance_metric: str = 'fuzzy'  # Distance metric for decisions
    ):
        self.llm_client = llm_client
        self.Z = Z
        self.N = N
        self.temperature = temperature
        self.max_steps = max_steps
        self.distance_metric = distance_metric
        
    def sample_tdp(self, prompt: str, context: str) -> TDP:
        """
        Sample a Trajectory-Dependent Decision Process.
        
        Starting from t=1, sample N decisions, randomly select one as y_k,
        then continue sampling conditioned on y_k for the next step.
        """
        tdp = TDP(trajectory=[], samples={})
        current_context = context
        
        for step in range(self.max_steps):
            # Sample N decisions at this step
            samples = []
            for _ in range(self.N):
                decision = self._sample_decision(prompt, current_context, step)
                if decision is None:  # Reached terminal state
                    break
                samples.append(decision)
            
            if not samples:
                break
                
            # Randomly select one sample as the trajectory decision
            # Note: Paper mentions "randomly select by probability" - interpreting as uniform random
            selected_idx = np.random.randint(0, len(samples))
            selected_decision = samples[selected_idx]
            
            # Store the trajectory decision and alternative samples
            tdp.trajectory.append(selected_decision)
            tdp.samples[step] = [s for i, s in enumerate(samples) if i != selected_idx]
            
            # Update context for next step
            current_context = self._update_context(current_context, selected_decision)
            
            # Check if we've reached a terminal state (e.g., Finish action)
            if self._is_terminal(selected_decision):
                break
                
        return tdp
    
    def calculate_intrinsic_uncertainty(self, decisions: List[Decision]) -> float:
        """
        Calculate intrinsic uncertainty H(y_t | y_{1:t-1}, x) using Predictive Entropy.
        
        This represents the uncertainty at the current step given all previous decisions.
        """
        if not decisions:
            return 0.0
        
        # Use negative log probabilities as a proxy for entropy
        # Length-normalized predictive entropy as mentioned in the paper
        uncertainties = []
        for decision in decisions:
            # Approximate decision length (educated guess: use action length)
            length = len(decision.action.split())
            if length > 0:
                uncertainty = -decision.log_prob / length  # Length normalization
                uncertainties.append(uncertainty)
        
        return np.mean(uncertainties) if uncertainties else 0.0
    
    def calculate_pmi(
        self, 
        current_samples: List[Decision],
        prev_decision: Decision,
        prev_samples: List[Decision],
        step: int
    ) -> float:
        """
        Calculate Pointwise Mutual Information using the spreading approach.
        
        PMI(y_t; y_{t-1} | x) ≈ -log(Σ K_N(d(y_{t-1}^{(n)}, y_{t-1}^{(k)})))
        
        where K_N is a Gaussian kernel and d is the distance between decisions.
        """
        if not prev_samples or not current_samples:
            return 0.0
        
        # Calculate distances between previous decision and all previous samples
        distances = []
        for sample in prev_samples:
            dist = self._calculate_decision_distance(prev_decision, sample)
            distances.append(dist)
        
        # Apply Gaussian kernel with sharpness control
        # Paper suggests τ = N for sharpness control
        tau = len(prev_samples)
        kernel_values = []
        for dist in distances:
            # Gaussian kernel: K_τ(x) = (1/√(2π)) * exp(-x²/2) / τ
            kernel_val = (1 / np.sqrt(2 * np.pi)) * np.exp(-dist**2 / 2) / tau
            kernel_values.append(kernel_val)
        
        # PMI approximation
        sum_kernels = sum(kernel_values)
        if sum_kernels > 0:
            pmi = -np.log(sum_kernels)
        else:
            pmi = 0.0  # Handle edge case
            
        return pmi
    
    def calculate_extrinsic_uncertainty(self, tdp: TDP, step: int) -> float:
        """
        Calculate extrinsic uncertainty as the sum of PMIs from all previous steps.
        
        EU_t = Σ_{i=1}^{t-1} PMI(y_t; y_i | y_{i+1:t-1}, x)
        """
        if step == 0 or not tdp.trajectory:
            return 0.0
        
        total_eu = 0.0
        current_samples = tdp.samples.get(step, [])
        
        # Sum PMI contributions from all previous steps
        for prev_step in range(step):
            if prev_step < len(tdp.trajectory):
                prev_decision = tdp.trajectory[prev_step]
                prev_samples = tdp.samples.get(prev_step, [])
                
                pmi = self.calculate_pmi(
                    current_samples, 
                    prev_decision,
                    prev_samples,
                    step
                )
                total_eu += pmi
                
        return total_eu
    
    def calculate_tdp_uncertainty(self, tdp: TDP) -> float:
        """
        Calculate total uncertainty for a TDP with step length normalization.
        
        H(P_TDP) = (1/λ_z) * Σ_t [H(y_t|y_{1:t-1}) + Σ_i PMI(y_t; y_i|y_{i+1:t-1})]
        
        where λ_z is the normalization factor.
        """
        total_uncertainty = 0.0
        step_weights = []
        
        for step, decision in enumerate(tdp.trajectory):
            # Get samples at this step
            samples = tdp.samples.get(step, []) + [decision]
            
            # Calculate intrinsic uncertainty
            iu = self.calculate_intrinsic_uncertainty(samples)
            
            # Calculate extrinsic uncertainty
            eu = self.calculate_extrinsic_uncertainty(tdp, step)
            
            # Step uncertainty
            step_uncertainty = iu + eu
            total_uncertainty += step_uncertainty
            
            # Calculate weight for normalization (σ_t = 1 + EU/IU)
            if iu > 0:
                sigma_t = 1 + (eu / iu)
            else:
                sigma_t = 1
            step_weights.append(sigma_t)
        
        # Apply step length normalization
        # λ_z = Σ σ_t = T_z + Σ (EU/IU)
        lambda_z = sum(step_weights)
        if lambda_z > 0:
            normalized_uncertainty = total_uncertainty / lambda_z
        else:
            normalized_uncertainty = total_uncertainty
            
        return normalized_uncertainty
    
    def estimate_uncertainty(self, prompt: str, context: str, target_answer: Optional[str] = None) -> Dict[str, float]:
        """
        Main entry point for UProp uncertainty estimation.
        
        Returns both total uncertainty and decomposed IU/EU components.
        """
        tdp_uncertainties = []
        tdp_with_target = []  # TDPs that end with target answer (if provided)
        
        tdp_data = []

        # Sample Z TDPs
        for z in range(self.Z):
            logger.info(f"Sampling TDP {z+1}/{self.Z}")
            tdp = self.sample_tdp(prompt, context)
            
            uncertainty = self.calculate_tdp_uncertainty(tdp)
            tdp_uncertainties.append(uncertainty)
            
            # Check if this TDP ends with target answer
            if target_answer and tdp.trajectory:
                last_action = tdp.trajectory[-1].action
                if self._matches_target(last_action, target_answer):
                    tdp_with_target.append(uncertainty)
            
            tdp_data.append(tdp.data_dump())
        
        # Calculate statistics
        results = {
            'total_uncertainty': np.mean(tdp_uncertainties),
            'std_uncertainty': np.std(tdp_uncertainties),
            'tdp_data': tdp_data,
            'tdp_uncertainties': tdp_uncertainties
        }
        
        # If target answer provided, calculate uncertainty for that specific prediction
        if target_answer and tdp_with_target:
            results['target_uncertainty'] = np.mean(tdp_with_target)
            
        return results
    
    def _sample_decision(self, prompt: str, context: str, step: int) -> Optional[Decision]:
        """
        Sample a single decision from the LLM.
        
        NOTE: This is a placeholder - actual implementation depends on the specific
        LLM API and task format (e.g., ReAct for HotpotQA).
        """
        # This will be implemented specifically for each task
        raise NotImplementedError("Must implement task-specific decision sampling")
    
    def _update_context(self, context: str, decision: Decision) -> str:
        """Update context with the new decision and observation."""
        return f"{context}\n{decision.reasoning}\n{decision.action}\n{decision.observation}"
    
    def _is_terminal(self, decision: Decision) -> bool:
        """Check if this decision represents a terminal state."""
        # For ReAct-style, terminal is usually "Finish[answer]"
        return decision.action.startswith("Finish")
    
    def _calculate_decision_distance(self, d1: Decision, d2: Decision) -> float:
        """
        Calculate distance between two decisions.
        
        Paper suggests using fuzzy string matching on actions.
        """
        if self.distance_metric == 'fuzzy':
            # Use fuzzy string matching on actions
            # fuzz.ratio returns 0-100, normalize to 0-1 and invert for distance
            similarity = fuzz.ratio(d1.action, d2.action) / 100.0
            return 1.0 - similarity
        else:
            # Fallback to simple exact match
            return 0.0 if d1.action == d2.action else 1.0
    
    def _matches_target(self, action: str, target: str) -> bool:
        """Check if action matches target answer."""
        # Extract answer from Finish[answer] format
        match = re.search(r'Finish\[(.*?)\]', action)
        if match:
            answer = match.group(1).lower().strip()
            return answer == target.lower().strip()
        return False


class SemanticEntropy:
    """
    Baseline implementation of Semantic Entropy for comparison.
    Based on Kuhn et al. (2023)
    """
    
    def __init__(self, n_samples: int = 5):
        self.n_samples = n_samples
    
    def calculate(self, samples: List[str], log_probs: List[float]) -> float:
        """
        Calculate semantic entropy by clustering semantically equivalent outputs.
        
        NOTE: Simplified implementation - actual semantic clustering would require
        NLI models or embeddings as mentioned in the paper.
        """
        # Group by exact match (simplified - real implementation needs semantic similarity)
        clusters = defaultdict(list)
        for sample, log_prob in zip(samples, log_probs):
            # Normalize sample for clustering
            key = sample.lower().strip()
            clusters[key].append(np.exp(log_prob))
        
        # Calculate cluster probabilities
        cluster_probs = []
        for cluster_samples in clusters.values():
            cluster_probs.append(sum(cluster_samples))
        
        # Normalize
        cluster_probs = np.array(cluster_probs)
        cluster_probs = cluster_probs / cluster_probs.sum()
        
        # Calculate entropy
        return entropy(cluster_probs)