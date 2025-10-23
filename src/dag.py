"""
DPIMPR: DAG-Preserving Incremental Multi-Path Reasoning
DAG-based uncertainty quantification for multi-step reasoning

IMPORTANT: This is the THIRD uncertainty quantification variant in MAUQ.
See ../UQ_VARIANTS.md for detailed comparison of:
- UProp: Trajectory-based sequential reasoning
- Graph-Based UQ: Multi-agent debate analysis
- DAG-Based UQ (this file): Universal graph-structured reasoning

Key DPIMPR Formulations:
    Node Uncertainty:       u_t = |log p_t| / 10.0
    Evidence Accumulation:  u_merged = weighted_avg - 0.1*log(k+1)
    Semantic Similarity:    sim(s_i, s_j) = max(SequenceMatcher, Jaccard*0.8)
    DAG Confidence:         Weighted by topology, evidence, uncertainty

Unique Features:
- Semantic merging of similar reasoning nodes
- Evidence accumulation reduces uncertainty
- Topological constraint enforcement (acyclic structure)
- Multiple uncertainty metrics from graph structure

Use this variant for:
- Both sequential reasoning AND debate (universal)
- When you want semantic similarity analysis
- When evidence accumulation is important
- When you need confidence scores (not just uncertainty)
- When you want graph structure insights

Key Difference from UProp:
- Outputs CONFIDENCE (higher = better), not uncertainty
- No length normalization of log-probs
- Evidence accumulation via log(k+1) benefit
- Graph-based metrics instead of trajectory-based
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
import hashlib
import json
import re
from difflib import SequenceMatcher
from sklearn.metrics import roc_auc_score

def parse_hotpotqa_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse HotpotQA trace data into a standardized format for DAG construction.

    HotpotQA traces have actions like "Search[entity]", "Lookup[keyword]", "Finish[answer]"
    with embedded observations in the action field.

    Args:
        trace_data: Raw HotpotQA trace data from UProp

    Returns:
        Standardized trajectory format with separated reasoning, action, observation
    """
    standardized_data = []

    for tdp in trace_data.get('tdp_data', []):
        trajectory = []

        for step in tdp.get('trajectory', []):
            # Parse action to separate action from embedded observation
            action_text = step.get('action', '')
            reasoning = step.get('reasoning', '')
            observation = step.get('observation', '')

            # HotpotQA often has observations embedded in the action field
            # Format: "Search[entity]\nObservation X: result"
            if '\nObservation' in action_text:
                parts = action_text.split('\nObservation', 1)
                clean_action = parts[0].strip()
                embedded_obs = 'Observation' + parts[1] if len(parts) > 1 else ''
                # Use embedded observation if main observation is generic
                if observation == "Task completed." or not observation:
                    observation = embedded_obs
            else:
                clean_action = action_text

            # Determine node type based on action
            if clean_action.startswith('Search['):
                node_type = 'search'
            elif clean_action.startswith('Lookup['):
                node_type = 'search'  # Lookup is also a search operation
            elif clean_action.startswith('Finish['):
                node_type = 'conclusion'
            else:
                node_type = 'reasoning'

            trajectory.append({
                'step': step.get('step', 0),
                'action': clean_action,
                'reasoning': reasoning,
                'observation': observation,
                'log_prob': step.get('log_prob', 0.0),
                'node_type': node_type
            })

        # Include samples if available
        samples = {}
        for step_key, step_samples in tdp.get('samples', {}).items():
            samples[step_key] = []
            for sample in step_samples:
                sample_action = sample.get('action', '')
                if '\nObservation' in sample_action:
                    parts = sample_action.split('\nObservation', 1)
                    clean_action = parts[0].strip()
                    embedded_obs = 'Observation' + parts[1] if len(parts) > 1 else ''
                    obs = embedded_obs if sample.get('observation') == "Task completed." else sample.get('observation', '')
                else:
                    clean_action = sample_action
                    obs = sample.get('observation', '')

                # Determine node type
                if clean_action.startswith('Search[') or clean_action.startswith('Lookup['):
                    node_type = 'search'
                elif clean_action.startswith('Finish['):
                    node_type = 'conclusion'
                else:
                    node_type = 'reasoning'

                samples[step_key].append({
                    'step': sample.get('step', 0),
                    'action': clean_action,
                    'reasoning': sample.get('reasoning', ''),
                    'observation': obs,
                    'log_prob': sample.get('log_prob', 0.0),
                    'node_type': node_type
                })

        standardized_data.append({
            'trajectory': trajectory,
            'samples': samples
        })

    return {
        'tdp_data': standardized_data,
        'question': trace_data.get('question', ''),
        'answer': trace_data.get('answer', ''),
        'predicted_answer': trace_data.get('predicted_answer', ''),
        'is_correct': trace_data.get('is_correct', False)
    }

def parse_gsm8k_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse GSM8K trace data into a standardized format for DAG construction.

    GSM8K traces have actions like "Reason[description]", "Calculator[expression]", "Finish[answer]"
    with clear tool-based interactions.

    Args:
        trace_data: Raw GSM8K trace data from UProp

    Returns:
        Standardized trajectory format with separated reasoning, action, observation
    """
    standardized_data = []

    for tdp in trace_data.get('tdp_data', []):
        trajectory = []

        for step in tdp.get('trajectory', []):
            action = step.get('action', '')
            reasoning = step.get('reasoning', '')
            observation = step.get('observation', '')

            # Determine node type based on action
            if action.startswith('Reason['):
                node_type = 'reasoning'
            elif action.startswith('Calculator['):
                node_type = 'calculation'
            elif action.startswith('Finish['):
                node_type = 'conclusion'
            else:
                node_type = 'reasoning'  # Default

            trajectory.append({
                'step': step.get('step', 0),
                'action': action,
                'reasoning': reasoning,
                'observation': observation,
                'log_prob': step.get('log_prob', 0.0),
                'node_type': node_type
            })

        # Include samples if available
        samples = {}
        for step_key, step_samples in tdp.get('samples', {}).items():
            samples[step_key] = []
            for sample in step_samples:
                action = sample.get('action', '')

                # Determine node type
                if action.startswith('Reason['):
                    node_type = 'reasoning'
                elif action.startswith('Calculator['):
                    node_type = 'calculation'
                elif action.startswith('Finish['):
                    node_type = 'conclusion'
                else:
                    node_type = 'reasoning'

                samples[step_key].append({
                    'step': sample.get('step', 0),
                    'action': action,
                    'reasoning': sample.get('reasoning', ''),
                    'observation': sample.get('observation', ''),
                    'log_prob': sample.get('log_prob', 0.0),
                    'node_type': node_type
                })

        standardized_data.append({
            'trajectory': trajectory,
            'samples': samples
        })

    return {
        'tdp_data': standardized_data,
        'question': trace_data.get('question', ''),
        'answer': trace_data.get('answer', ''),
        'predicted_answer': trace_data.get('predicted_answer', ''),
        'is_correct': trace_data.get('is_correct', False)
    }

def auto_detect_trace_format(trace_data: Dict[str, Any]) -> str:
    """
    Automatically detect whether trace data is from HotpotQA or GSM8K based on action patterns.

    Args:
        trace_data: Raw trace data

    Returns:
        'hotpotqa' or 'gsm8k'
    """
    # Look at the first few actions to determine format
    sample_actions = []

    for tdp in trace_data.get('tdp_data', [])[:2]:  # Check first 2 TDPs
        for step in tdp.get('trajectory', [])[:3]:  # Check first 3 steps
            action = step.get('action', '')
            sample_actions.append(action)

    # Count action patterns
    hotpotqa_patterns = ['Search[', 'Lookup[', 'Finish[']
    gsm8k_patterns = ['Reason[', 'Calculator[', 'Finish[']

    hotpotqa_score = sum(1 for action in sample_actions for pattern in hotpotqa_patterns if pattern in action)
    gsm8k_score = sum(1 for action in sample_actions for pattern in gsm8k_patterns if pattern in action)

    return 'hotpotqa' if hotpotqa_score > gsm8k_score else 'gsm8k'

@dataclass
class DAGReasoningNode:
    """Enhanced reasoning node with topological and belief information"""
    node_id: str
    content: str
    node_type: str  # 'search', 'observation', 'reasoning', 'conclusion'
    base_uncertainty: float
    belief_state: Dict[str, float]
    evidence_count: int = 1
    trajectory_sources: Set[str] = field(default_factory=set)
    topological_level: int = 0
    temporal_order: int = 0
    log_probs: List[float] = field(default_factory=list)
    
    def add_evidence(self, uncertainty: float, belief: Dict[str, float], 
                    trajectory_id: str, log_prob: float):
        """Add new evidence to this node"""
        self.trajectory_sources.add(trajectory_id)
        self.evidence_count += 1
        self.log_probs.append(log_prob)
        
        # Uncertainty-weighted belief update
        old_weight = self.evidence_count - 1
        new_weight = 1.0
        total_weight = old_weight + new_weight
        
        # Update belief state with weighted average
        merged_belief = {}
        all_states = set(self.belief_state.keys()) | set(belief.keys())
        
        for state in all_states:
            old_prob = self.belief_state.get(state, 0.0)
            new_prob = belief.get(state, 0.0)
            merged_belief[state] = (old_prob * old_weight + new_prob * new_weight) / total_weight
        
        self.belief_state = merged_belief
        
        # Update uncertainty with convergence benefit
        convergence_benefit = 0.1 * np.log(self.evidence_count + 1)
        weighted_uncertainty = (self.base_uncertainty * old_weight + uncertainty * new_weight) / total_weight
        self.base_uncertainty = max(0.1, weighted_uncertainty - convergence_benefit)

class DPIMPR:
    """
    DAG-Preserving Incremental Multi-Path Reasoning Algorithm
    
    Implements the algorithm described in the paper for building reasoning DAGs
    that maintain acyclic structure while integrating multiple trajectories.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 topological_distance: int = 25,
                 belief_propagation_iterations: int = 5,
                 question: str = None):
        self.G = nx.DiGraph()
        self.nodes: Dict[str, DAGReasoningNode] = {}
        self.similarity_threshold = similarity_threshold
        self.topological_distance = topological_distance
        self.bp_iterations = belief_propagation_iterations
        self.topological_levels: Dict[str, int] = {}
        self.trajectory_count = 0
        self.question = question
        
    def _parse_action_and_observation(self, action_text: str) -> Tuple[str, str]:
        """
        Parse the action field to separate the actual action from the observation
        
        Args:
            action_text: The raw action field that may contain both action and observation
            
        Returns:
            Tuple of (action, observation) where observation may be empty if not found
        """
        # Split on observation markers
        parts = re.split(r'\nObservation \d+:', action_text)
        
        if len(parts) > 1:
            # There's an observation in the action field
            action_part = parts[0].strip()
            observation_part = parts[1].strip()
            return action_part, observation_part
        else:
            # No observation in action field, return action only
            return action_text.strip(), ""
    
    def _extract_semantic_content(self, text: str) -> str:
        """Extract the core semantic content from an action or observation string"""
        # Remove action prefixes and clean up
        content = re.sub(r'^(Search|Finish)\[([^\]]+)\].*', r'\2', text)
        content = re.sub(r'\n.*', '', content)  # Remove everything after newline
        return content.strip()
    
    def _classify_node_type(self, text: str, is_observation: bool = False) -> str:
        """Classify the type of reasoning node"""
        if is_observation:
            return 'observation'
        elif text.startswith('Search['):
            return 'search'
        elif text.startswith('Finish['):
            return 'conclusion'
        else:
            return 'reasoning'
    
    def _extract_belief_state(self, action: str, observation: str, reasoning: str) -> Dict[str, float]:
        """Extract belief state from the step information"""
        belief = {'unknown': 0.1}
        
        # Combine all text for analysis
        content = f"{action} {observation} {reasoning}".lower()
        
        # Look for years or dates
        years = re.findall(r'\b(19|20)\d{2}\b', content)
        for year in years:
            belief[f"year_{year}"] = 0.8
            
        # Look for entities in brackets or key terms
        entities = re.findall(r'\[([^\]]+)\]', action)
        for entity in entities:
            belief[entity.lower().replace(' ', '_')] = 0.9
            
        # Normalize belief state
        total = sum(belief.values())
        if total > 0:
            belief = {k: v/total for k, v in belief.items()}
        else:
            belief = {'unknown': 1.0}
            
        return belief
    
    def _uncertainty_from_log_prob(self, log_prob: float) -> float:
        """Convert log probability to uncertainty measure"""
        # Convert log prob to uncertainty (higher negative log prob = higher uncertainty)
        return abs(log_prob) / 10.0  # Scale factor
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Use SequenceMatcher for basic similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Boost similarity for exact keyword matches
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            # Combine sequence similarity with word overlap
            similarity = max(similarity, word_overlap * 0.8)
        
        return min(1.0, similarity)

    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Check if adding edge would create a cycle using topological levels"""
        if from_node not in self.topological_levels or to_node not in self.topological_levels:
            return False
            
        from_level = self.topological_levels[from_node]
        to_level = self.topological_levels[to_node]
        
        return from_level >= to_level
    
    def _update_topological_levels(self):
        """Update topological levels after graph modification"""
        try:
            topo_order = list(nx.topological_sort(self.G))
            for i, node_id in enumerate(topo_order):
                self.topological_levels[node_id] = i
                if node_id in self.nodes:
                    self.nodes[node_id].topological_level = i
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles - DAG property violated!")
    
    def _find_safe_matching_node(self, content: str, node_type: str, 
                                temporal_position: int, similarity_threshold: float = None,
                                trajectory_nodes: List[str] = None) -> Optional[str]:
        """
        Find existing node that can be safely merged without creating cycles
        Implements the FindSafeMatchingNode function from the paper
        """
        candidates = []
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        for node_id, node in self.nodes.items():
            if trajectory_nodes and node_id in trajectory_nodes:
                continue
            if node.node_type == node_type:
                print(f"content: {content}, node.content: {node.content}")
                similarity = self._compute_semantic_similarity(content, node.content)
                print(f"similarity: {similarity}")
                #print(f"content: {content}, node.content: {node.content}, Similarity: {similarity}")
                if similarity > self.similarity_threshold:
                    candidates.append((node_id, similarity, node.topological_level))
        
        if not candidates:
            return None
        
        # Sort by similarity descending, then by topological level
        candidates.sort(key=lambda x: (-x[1], x[2]))
        print(f"candidates: {candidates}")
        
        # Return the most similar candidate within topological distance
        for node_id, similarity, topo_level in candidates:
            if abs(topo_level - temporal_position) <= self.topological_distance:
                return node_id
        
        return None
    
    def _generate_node_id(self, content: str, node_type: str, 
                         trajectory_id: str, position: int) -> str:
        """Generate unique node ID"""
        content_hash = hashlib.md5(f"{node_type}:{content}:{trajectory_id}:{position}".encode()).hexdigest()[:8]
        return f"{node_type}_{content_hash}"
    
    def _update_graph_node_attributes(self, node_id: str, node: DAGReasoningNode):
        """
        Update graph node attributes with current node state
        
        Args:
            node_id: The ID of the node to update
            node: The DAGReasoningNode instance containing updated attributes
        """
        if node_id in self.G.nodes():
            self.G.nodes[node_id].update({
                'uncertainty': node.base_uncertainty,
                'belief_state': node.belief_state,
                'evidence_count': node.evidence_count,
                'content': node.content,
                'node_type': node.node_type,
                'trajectory_sources': list(node.trajectory_sources),
                'topological_level': node.topological_level,
                'temporal_order': node.temporal_order
            })

  
    def add_trajectory(self, trajectory_data: Dict[str, Any], trajectory_id: str) -> List[str]:
        """
        Add a new trajectory to the DAG using the DPIMPR algorithm
        
        Args:
            trajectory_data: Dictionary containing trajectory information (can include samples)
            trajectory_id: Unique identifier for this trajectory
            
        Returns:
            List of node IDs created/updated for this trajectory
        """
        self.trajectory_count += 1
        trajectory_nodes = []
        current_highest_order = 0
        
        print(f"\n=== Adding Trajectory {trajectory_id} ===")
        
        # Extract main trajectory steps
        if 'trajectory' in trajectory_data:
            steps = trajectory_data['trajectory']
        else:
            steps = trajectory_data.get('steps', [])
        
        # Process each step
        for i, step in enumerate(steps):
            action = step.get('action', '')
            reasoning = step.get('reasoning', '')
            observation = step.get('observation', '')
            log_prob = step.get('log_prob', -10.0)
            
            # Parse action to separate action from embedded observation
            action_text, embedded_observation = self._parse_action_and_observation(action)
            
            # Process action node
            action_content = self._extract_semantic_content(action_text)
            if not action_content:  # Fallback to reasoning if no content extracted
                action_content = reasoning[:50] + "..." if len(reasoning) > 50 else reasoning
                
            action_node_type = self._classify_node_type(action_text)
            action_uncertainty = self._uncertainty_from_log_prob(log_prob)
            action_belief_state = self._extract_belief_state(action_text, reasoning, "")
            
            print(f"  Step {i}: {action_node_type} - '{action_content[:40]}...'")
            print(f"  Action text: {action_content}")
            
            
            # Try to find compatible existing action node
            matching_action_id = self._find_safe_matching_node(action_content, action_node_type, i, 
            trajectory_nodes=trajectory_nodes)
            no_cycle = False if matching_action_id is None else (self.nodes[matching_action_id].topological_level >= current_highest_order)
            print(f"matching_action_id: {matching_action_id}, no_cycle: {no_cycle}")
            
            if matching_action_id and no_cycle:
                print(f"    Merging action with existing node {matching_action_id}")
                order = self.nodes[matching_action_id].topological_level

                existing_node = self.nodes[matching_action_id]
                existing_node.add_evidence(action_uncertainty, action_belief_state, trajectory_id, log_prob)
                
                # Update graph attributes
                self.G.nodes[matching_action_id].update({
                    'uncertainty': existing_node.base_uncertainty,
                    'belief_state': existing_node.belief_state,
                    'evidence_count': existing_node.evidence_count
                })
                
                trajectory_nodes.append(matching_action_id)
                current_highest_order = max(current_highest_order, order)
            else:
                # Create new action node
                action_node_id = self._generate_node_id(action_content, action_node_type, trajectory_id, i)
                print(f"    Creating new action node {action_node_id}")
                
                new_action_node = DAGReasoningNode(
                    node_id=action_node_id,
                    content=action_content,
                    node_type=action_node_type,
                    base_uncertainty=action_uncertainty,
                    belief_state=action_belief_state,
                    trajectory_sources={trajectory_id},
                    temporal_order=i,
                    log_probs=[log_prob]
                )
                
                self.nodes[action_node_id] = new_action_node
                self.G.add_node(action_node_id, **{
                    'content': action_content,
                    'node_type': action_node_type,
                    'uncertainty': action_uncertainty,
                    'belief_state': action_belief_state,
                    'evidence_count': 1,
                    'full_action': action_text,
                    'reasoning': reasoning,
                    'observation': ""
                })
                trajectory_nodes.append(action_node_id)
            
            if len(trajectory_nodes) > 1:
                print(f"Adding edge {trajectory_nodes[-2]} -> {trajectory_nodes[-1]}")
                self.G.add_edge(trajectory_nodes[-2], trajectory_nodes[-1],
                    weight=1.0,
                    trajectory_sources={trajectory_id})

            # Process observation node if present
            observation_text = embedded_observation if embedded_observation else observation
            
            if observation_text and observation_text.strip() not in ["", "Task completed."]:
                observation_content = observation_text[:100] + "..." if len(observation_text) > 100 else observation_text
                observation_node_type = 'observation'
                observation_uncertainty = action_uncertainty * 0.8
                observation_belief_state = self._extract_belief_state("", observation_text, reasoning)
                
                print(f"    Processing observation: '{observation_content[:40]}...'")
                
                # Try to find compatible existing observation node
                matching_obs_id = self._find_safe_matching_node(
                    observation_content, observation_node_type, i + 0.5,  # Offset for temporal positioning
                    similarity_threshold=0.85,  # Higher threshold for observations
                    trajectory_nodes = trajectory_nodes
                )
                if matching_obs_id:
                    print(f"matching_obs_level: {self.nodes[matching_obs_id].topological_level}", f"current_highest_order: {current_highest_order}")

                no_cycle = False if matching_obs_id is None else (self.nodes[matching_obs_id].topological_level >= current_highest_order)
                print(f"matching_obs_id: {matching_obs_id}, no_cycle: {no_cycle}")

                if matching_obs_id and no_cycle:
                    print(f"    Merging observation with existing node {matching_obs_id}")
                    existing_obs_node = self.nodes[matching_obs_id]
                    existing_obs_node.add_evidence(observation_uncertainty, observation_belief_state, trajectory_id, log_prob)
                    
                    # Update graph attributes
                    self.G.nodes[matching_obs_id].update({
                        'uncertainty': existing_obs_node.base_uncertainty,
                        'belief_state': existing_obs_node.belief_state,
                        'evidence_count': existing_obs_node.evidence_count
                    })
                    
                    trajectory_nodes.append(matching_obs_id)
                    current_highest_order = max(current_highest_order, self.nodes[matching_obs_id].topological_level)
                else:
                    # Create new observation node
                    observation_node_id = self._generate_node_id(observation_content, observation_node_type, trajectory_id, i + 0.5)
                    print(f"    Creating new observation node {observation_node_id}")
                    
                    new_obs_node = DAGReasoningNode(
                        node_id=observation_node_id,
                        content=observation_content,
                        node_type=observation_node_type,
                        base_uncertainty=observation_uncertainty,
                        belief_state=observation_belief_state,
                        trajectory_sources={trajectory_id},
                        temporal_order=i + 0.5,
                        log_probs=[log_prob]
                    )
                    
                    self.nodes[observation_node_id] = new_obs_node
                    self.G.add_node(observation_node_id, **{
                        'content': observation_content,
                        'node_type': observation_node_type,
                        'uncertainty': observation_uncertainty,
                        'belief_state': observation_belief_state,
                        'evidence_count': 1,
                        'full_text': observation_text,
                        'reasoning': reasoning,
                        'observation': observation_text
                    })
                    trajectory_nodes.append(observation_node_id)
                
                if len(trajectory_nodes) > 1:
                    print(f"Adding edge {trajectory_nodes[-2]} -> {trajectory_nodes[-1]}")
                    self.G.add_edge(trajectory_nodes[-2], trajectory_nodes[-1],
                                    weight=1.0,
                                    trajectory_sources={trajectory_id})
        
        
        # Update topological ordering
        self._update_topological_levels()
        
        return trajectory_nodes
    
    def add_tdp_data(self, tdp_data: List[Dict[str, Any]], question: str = None) -> Dict[str, Any]:
        """
        Process TDP (Trajectory-Dependent Decision Process) data containing multiple trajectories
        
        Args:
            tdp_data: List of trajectory dictionaries with main trajectory and samples
            question: Optional question context
            
        Returns:
            Dictionary with processing results and analysis
        """
        print(f"=== Processing TDP Data: {len(tdp_data)} trajectories ===")
        if question:
            print(f"Question: {question}")
        
        all_trajectory_nodes = []
        
        # Process each trajectory in the TDP data
        for i, trajectory_data in enumerate(tdp_data):
            trajectory_id = f"trajectory_{i}"
            nodes = self.add_trajectory(trajectory_data, trajectory_id)
            all_trajectory_nodes.append(nodes)
        
        # Analyze the constructed DAG
        analysis = self.analyze_dag_properties()
        confidence = self.compute_dag_confidence()
        
        # Compile results
        results = {
            'question': question,
            'trajectories_processed': len(tdp_data),
            'dag_analysis': analysis,
            'confidence_analysis': confidence,
            'trajectory_nodes': all_trajectory_nodes
        }
        
        return results

    def add_reasoning_traces(self, trace_data: Dict[str, Any], trace_format: str = None) -> Dict[str, Any]:
        """
        Unified method to process reasoning traces from either HotpotQA or GSM8K.

        Args:
            trace_data: Raw trace data from UProp (either HotpotQA or GSM8K format)
            trace_format: 'hotpotqa', 'gsm8k', or None for auto-detection

        Returns:
            Dictionary with processing results and analysis
        """
        # Auto-detect format if not specified
        if trace_format is None:
            trace_format = auto_detect_trace_format(trace_data)
            print(f"Auto-detected trace format: {trace_format}")

        # Parse the trace data using the appropriate parser
        if trace_format == 'hotpotqa':
            standardized_data = parse_hotpotqa_trace(trace_data)
        elif trace_format == 'gsm8k':
            standardized_data = parse_gsm8k_trace(trace_data)
        else:
            raise ValueError(f"Unsupported trace format: {trace_format}")

        # Update question context if available
        question = standardized_data.get('question', '')
        if question and self.question is None:
            self.question = question

        print(f"=== Processing {trace_format.upper()} Reasoning Traces ===")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        print(f"Predicted Answer: {standardized_data.get('predicted_answer', 'N/A')}")
        print(f"Is Correct: {standardized_data.get('is_correct', 'N/A')}")

        # Use the existing TDP processing logic
        results = self.add_tdp_data(standardized_data['tdp_data'], question)

        # Add format-specific metadata
        results['trace_format'] = trace_format
        results['predicted_answer'] = standardized_data.get('predicted_answer')
        results['is_correct'] = standardized_data.get('is_correct')
        results['ground_truth'] = standardized_data.get('answer')

        return results

    def extract_dag_uncertainty_metrics(self) -> Dict[str, float]:
        """
        Extract various uncertainty metrics from the constructed DAG for evaluation.

        Returns:
            Dictionary containing different uncertainty measures
        """
        if not self.G.nodes():
            return {
                'dag_confidence': 0.0,
                'avg_node_uncertainty': 1.0,
                'conclusion_uncertainty': 1.0,
                'path_diversity': 0.0,
                'evidence_strength': 0.0,
                'topological_depth': 0.0
            }

        # Get DAG confidence (inverted for uncertainty)
        confidence_analysis = self.compute_dag_confidence()
        dag_confidence = confidence_analysis.get('final_confidence', 0.0)

        # Average node uncertainty
        node_uncertainties = [node.base_uncertainty for node in self.nodes.values()]
        avg_node_uncertainty = np.mean(node_uncertainties) if node_uncertainties else 1.0

        # Conclusion node uncertainty (look for conclusion-type nodes)
        conclusion_nodes = [node for node in self.nodes.values() if node.node_type == 'conclusion']
        conclusion_uncertainty = np.mean([node.base_uncertainty for node in conclusion_nodes]) if conclusion_nodes else avg_node_uncertainty

        # Path diversity (number of different paths to conclusions)
        conclusion_node_ids = [node_id for node_id, node in self.nodes.items() if node.node_type == 'conclusion']
        path_diversity = 0.0
        if conclusion_node_ids:
            # Count paths by looking at predecessors
            all_paths = []
            for conclusion_id in conclusion_node_ids:
                try:
                    paths = list(nx.all_simple_paths(self.G,
                                                   source=[n for n in self.G.nodes() if self.G.in_degree(n) == 0],
                                                   target=conclusion_id))
                    all_paths.extend(paths)
                except:
                    pass
            path_diversity = len(set(tuple(path) for path in all_paths))

        # Evidence strength (average evidence count across nodes)
        evidence_counts = [node.evidence_count for node in self.nodes.values()]
        evidence_strength = np.mean(evidence_counts) if evidence_counts else 0.0

        # Topological depth (longest path in DAG)
        try:
            topological_depth = nx.dag_longest_path_length(self.G) if nx.is_directed_acyclic_graph(self.G) else 0
        except:
            topological_depth = 0

        return {
            'dag_confidence': dag_confidence,
            'avg_node_uncertainty': avg_node_uncertainty,
            'conclusion_uncertainty': conclusion_uncertainty,
            'path_diversity': path_diversity,
            'evidence_strength': evidence_strength,
            'topological_depth': topological_depth,
            # Derived uncertainty measures (higher = more uncertain)
            'uncertainty_dag_confidence': 1.0 - dag_confidence,  # Inverted confidence
            'uncertainty_evidence_strength': 1.0 / (evidence_strength + 1.0),  # Inverted evidence
            'uncertainty_path_diversity': 1.0 / (path_diversity + 1.0)  # Inverted diversity
        }

    def compute_dag_confidence(self, target_conclusion: str = None) -> Dict[str, float]:
        """
        Compute confidence considering DAG structure and evidence convergence
        """
        if not nx.is_directed_acyclic_graph(self.G):
            raise ValueError("Graph is not a DAG - cannot compute DAG confidence")
        
        # Find conclusion nodes
        conclusion_nodes = [node_id for node_id, node in self.nodes.items() 
                          if node.node_type == 'conclusion']
        
        if not conclusion_nodes:
            return {'confidence': 0.0, 'evidence_strength': 0.0, 'conclusion': 'No conclusion found'}
        
        # If no target specified, find the most confident conclusion
        if target_conclusion is None:
            best_confidence = 0.0
            best_conclusion = None
            
            for node_id in conclusion_nodes:
                node = self.nodes[node_id]
                # Get the most likely conclusion from belief state
                if node.belief_state:
                    max_belief_state = max(node.belief_state.items(), key=lambda x: x[1])
                    conclusion_name = max_belief_state[0]
                    confidence = max_belief_state[1] * node.evidence_count / (node.base_uncertainty + 0.1)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_conclusion = conclusion_name
            
            target_conclusion = best_conclusion or 'unknown'
        
        # Compute weighted confidence
        total_confidence = 0.0
        total_weight = 0.0
        evidence_strength = 0.0
        
        max_topo_level = max(self.topological_levels.values()) if self.topological_levels else 1
        
        for node_id in conclusion_nodes:
            node = self.nodes[node_id]
            
            # Topological weight (later nodes get higher weight)
            topo_level = self.topological_levels.get(node_id, 0)
            topo_weight = (topo_level + 1) / max_topo_level
            
            # Evidence weight
            evidence_weight = node.evidence_count
            
            # Uncertainty weight
            uncertainty_weight = 1.0 / (node.base_uncertainty + 0.1)
            
            total_node_weight = topo_weight * evidence_weight * uncertainty_weight
            
            # Find probability of target conclusion in belief state
            conclusion_prob = 0.0
            for state, prob in node.belief_state.items():
                if target_conclusion.lower() in state.lower() or state.lower() in target_conclusion.lower():
                    conclusion_prob = max(conclusion_prob, prob)
            
            total_confidence += conclusion_prob * total_node_weight
            total_weight += total_node_weight
            evidence_strength += evidence_weight
        
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        return {
            'confidence': final_confidence,
            'evidence_strength': evidence_strength,
            'conclusion': target_conclusion,
            'topological_depth': max_topo_level,
            'num_trajectories': self.trajectory_count
        }
    
    def analyze_dag_properties(self) -> Dict[str, Any]:
        """Analyze structural properties of the constructed DAG"""
        analysis = {
            'is_dag': nx.is_directed_acyclic_graph(self.G),
            'node_count': len(self.G.nodes()),
            'edge_count': len(self.G.edges()),
            'trajectory_count': self.trajectory_count,
            'convergence_points': [],
            'evidence_distribution': {},
            'auxiliary_edges': 0
        }
        
        # Find convergence points (nodes with multiple predecessors)
        for node_id in self.G.nodes():
            in_degree = self.G.in_degree(node_id)
            if in_degree > 1:
                node = self.nodes.get(node_id)
                analysis['convergence_points'].append({
                    'node_id': node_id,
                    'content': node.content if node else 'Unknown',
                    'evidence_count': node.evidence_count if node else 0,
                    'in_degree': in_degree,
                    'uncertainty': node.base_uncertainty if node else float('inf')
                })
        
        # Count auxiliary edges
        for u, v, data in self.G.edges(data=True):
            if data.get('edge_type') == 'auxiliary':
                analysis['auxiliary_edges'] += 1
        
        # Evidence distribution
        if self.nodes:
            evidence_counts = [node.evidence_count for node in self.nodes.values()]
            analysis['evidence_distribution'] = {
                'mean': np.mean(evidence_counts),
                'max': max(evidence_counts),
                'min': min(evidence_counts),
                'std': np.std(evidence_counts),
                'nodes_with_multiple_evidence': sum(1 for c in evidence_counts if c > 1)
            }
        
        return analysis
    
    def _sugiyama_layout(self):
        """
        Implement Sugiyama algorithm for layered graph drawing
        
        The Sugiyama algorithm consists of:
        1. Layer assignment (hierarchical layering)
        2. Crossing reduction (minimize edge crossings within layers)
        3. Coordinate assignment (assign x-coordinates to minimize edge bends)
        """
       
    
        # Step 1: Layer Assignment
        layers = self._assign_layers()
        
        # Step 2: Crossing Reduction
        self._reduce_crossings(layers)
        
        # Step 3: Coordinate Assignment
        pos = self._assign_coordinates(layers)
        
        return pos
    
    def _assign_layers(self):
        """
        Assign nodes to layers using longest path layering
        """
        layers = defaultdict(list)
        visited = set()
        
        def longest_path_layering(node):
            if node in visited:
                return self.topological_levels.get(node, 0)
            
            visited.add(node)
            max_depth = 0
            
            # Find the maximum depth of any successor
            for successor in self.G.successors(node):
                depth = longest_path_layering(successor)
                max_depth = max(max_depth, depth)
            
            # Assign layer (inverse of depth for top-down layout)
            layer = max_depth
            layers[layer].append(node)
            self.topological_levels[node] = layer
            
            return layer + 1
        
        # Find all source nodes (nodes with no predecessors)
        source_nodes = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        
        # Start layering from source nodes
        for source in source_nodes:
            if source not in visited:
                longest_path_layering(source)
        
        # Handle any remaining nodes (in case of disconnected components)
        for node in self.G.nodes():
            if node not in visited:
                layers[0].append(node)
                self.topological_levels[node] = 0
        
        return layers
    
    def _reduce_crossings(self, layers):
        """
        Reduce edge crossings using barycenter heuristic
        """
        # For now, use a simple ordering based on node properties
        # In a full implementation, this would use more sophisticated algorithms
        for layer_idx, layer_nodes in layers.items():
            # Sort nodes within each layer
            # Main trajectory nodes first, then sample nodes grouped by parent
            main_nodes = []
            sample_nodes = []
            
            for node in layer_nodes:
                if self.G.nodes[node].get('is_sample_variation', False):
                    sample_nodes.append(node)
                else:
                    main_nodes.append(node)
            
            # Sort main nodes by their content or other criteria
            main_nodes.sort(key=lambda n: self.nodes[n].content if n in self.nodes else n)
            
            # Group sample nodes by parent and sort within groups
            parent_groups = defaultdict(list)
            for node in sample_nodes:
                parent_step = self.G.nodes[node].get('parent_step', 0)
                sample_index = self.G.nodes[node].get('sample_index', 0)
                parent_groups[parent_step].append((node, sample_index))
            
            # Sort each parent group by sample index
            for parent_step in parent_groups:
                parent_groups[parent_step].sort(key=lambda x: x[1])
            
            # Reorder the layer: main nodes first, then sample groups
            new_order = main_nodes[:]
            for parent_step in sorted(parent_groups.keys()):
                for node, _ in parent_groups[parent_step]:
                    new_order.append(node)
            
            # Update the layer with new ordering
            layers[layer_idx] = new_order
    
    def _assign_coordinates(self, layers):
        """
        Assign x,y coordinates to nodes in each layer
        """
        pos = {}
        
        for layer_idx, layer_nodes in layers.items():
            y_position = -layer_idx * 3  # Vertical spacing between layers
            
            # Separate main nodes from sample nodes
            main_nodes = []
            sample_nodes = []
            
            for node in layer_nodes:
                if self.G.nodes[node].get('is_sample_variation', False):
                    sample_nodes.append(node)
                else:
                    main_nodes.append(node)
            
            # Position main nodes with equal spacing
            for i, node in enumerate(main_nodes):
                x_position = (i - len(main_nodes)/2) * 6
                pos[node] = (x_position, y_position)
            
            # Group sample nodes by parent and position them
            parent_to_samples = defaultdict(list)
            for node in sample_nodes:
                parent_step = self.G.nodes[node].get('parent_step', 0)
                sample_index = self.G.nodes[node].get('sample_index', 0)
                
                # Find the parent node in this layer
                parent_node = None
                for main_node in main_nodes:
                    if self.topological_levels.get(main_node, 0) == layer_idx:
                        parent_node = main_node
                        break
                
                if parent_node:
                    parent_to_samples[parent_node].append((node, sample_index))
            
            # Position sample nodes side-by-side under their parents
            for parent_node, samples in parent_to_samples.items():
                if not samples:
                    continue
                
                parent_x = pos[parent_node][0]
                parent_y = pos[parent_node][1]
                
                # Sort samples by sample index
                samples.sort(key=lambda x: x[1])
                
                # Calculate positions for samples
                sample_width = 2.5
                total_width = len(samples) * sample_width
                start_x = parent_x - total_width / 2
                
                for i, (sample_node, sample_idx) in enumerate(samples):
                    x_position = start_x + i * sample_width
                    y_position = parent_y - 2.0  # Position below parent
                    pos[sample_node] = (x_position, y_position)
        
        return pos
    
    def visualize_dag(self, figsize=(16, 10), save_path=None):
        """Visualize the DAG structure using Sugiyama algorithm for layered graph drawing"""
        if not self.G.nodes():
            print("No graph to visualize")
            return None
            
        plt.figure(figsize=figsize)
        
        # Use Sugiyama algorithm for layered graph layout
        pos = self._sugiyama_layout()
        
        # Color and size nodes based on evidence and type
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node_id in self.G.nodes():
            node = self.nodes.get(node_id)
            node_data = self.G.nodes[node_id]
            is_sample = node_data.get('is_sample_variation', False)
            
            if node:
                evidence_count = node.evidence_count
                node_type = node.node_type
                
                # Size based on evidence and sample status
                if is_sample:
                    size = 400 + evidence_count * 200  # Smaller for samples
                else:
                    size = 800 + evidence_count * 300  # Larger for main nodes
                node_sizes.append(size)
                
                # Color based on type, uncertainty, and sample status
                base_colors = {
                    'search': (0.3, 0.5, 0.9),        # Blue for search actions
                    'observation': (0.9, 0.6, 0.3),   # Orange for observations
                    'reasoning': (0.7, 0.3, 0.7),     # Purple for reasoning
                    'calculation': (0.3, 0.9, 0.6),   # Green for calculations
                    'conclusion': (0.9, 0.3, 0.3)     # Red for conclusions
                }
                
                # Use uncertainty for color intensity (lower uncertainty = more saturated color)
                uncertainty_score = node.base_uncertainty
                # Invert uncertainty: lower uncertainty (0) = high intensity (1), higher uncertainty (10) = low intensity (0)
                uncertainty_intensity = 1.0 - min(1.0, uncertainty_score / 10.0)  # Normalize uncertainty (0-10 range) and invert
                base_color = base_colors.get(node_type, (0.5, 0.5, 0.5))
                
                if is_sample:
                    # Make sample nodes lighter/more transparent, but still show uncertainty
                    sample_intensity = uncertainty_intensity * 0.7 + 0.3
                    node_colors.append(tuple(c * sample_intensity + (1 - sample_intensity) * 0.3 for c in base_color))
                else:
                    # Main nodes with uncertainty-based intensity variation (lower uncertainty = more intense)
                    if node_type == 'search':
                        node_colors.append((0.3, 0.5 + 0.5*uncertainty_intensity, 0.9))
                    elif node_type == 'observation':
                        node_colors.append((0.9, 0.6 + 0.3*uncertainty_intensity, 0.3))
                    elif node_type == 'reasoning':
                        node_colors.append((0.7, 0.3 + 0.5*uncertainty_intensity, 0.7))
                    elif node_type == 'calculation':
                        node_colors.append((0.3, 0.9, 0.6 + 0.3*uncertainty_intensity))
                    else:  # conclusion
                        node_colors.append((0.9, 0.3 + 0.5*uncertainty_intensity, 0.3))
                
                # Create label with sample indicator, confidence, and uncertainty
                short_content = node.content[:25] + "..." if len(node.content) > 15 else node.content
                uncertainty_score = node.base_uncertainty
                
                if is_sample:
                    sample_idx = node_data.get('sample_index', 0)
                    node_labels[node_id] = f"S{sample_idx}: {short_content}\nU:{uncertainty_score:.2f}"
                else:
                    node_labels[node_id] = f"{short_content}\nE:{evidence_count} U:{uncertainty_score:.2f}"
            else:
                node_sizes.append(600)
                node_colors.append((0.7, 0.7, 0.7))
                node_labels[node_id] = node_id[:10]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8)
        
        # Draw edges with different styles
        regular_edges = [(u, v) for u, v, data in self.G.edges(data=True)
                        if data.get('edge_type') not in ['auxiliary', 'sample_branch', 'sample_flow']]
        auxiliary_edges = [(u, v) for u, v, data in self.G.edges(data=True)
                          if data.get('edge_type') == 'auxiliary']
        sample_branch_edges = [(u, v) for u, v, data in self.G.edges(data=True)
                              if data.get('edge_type') == 'sample_branch']
        sample_flow_edges = [(u, v) for u, v, data in self.G.edges(data=True)
                            if data.get('edge_type') == 'sample_flow']
        
        # Regular edges
        if regular_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=regular_edges,
                                  edge_color='black', width=2, alpha=0.7,
                                  arrows=True, arrowsize=20)
        
        # Sample branch edges (from main action to sample actions)
        if sample_branch_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=sample_branch_edges,
                                  edge_color='green', width=2, alpha=0.8,
                                  style='dotted', arrows=True, arrowsize=15)
        
        # Sample flow edges (from sample action to sample observation)
        if sample_flow_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=sample_flow_edges,
                                  edge_color='orange', width=1.5, alpha=0.7,
                                  style='dashed', arrows=True, arrowsize=12)
        
        # Auxiliary edges
        if auxiliary_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=auxiliary_edges,
                                  edge_color='red', width=1, alpha=0.5,
                                  style='dashed', arrows=True, arrowsize=15)
        
        # Add labels
        nx.draw_networkx_labels(self.G, pos, node_labels, font_size=8, font_weight='bold')
        
        # Add level indicators
        levels = set()
        for node_id in pos:
            level = self.topological_levels.get(node_id, 0)
            levels.add(level)
        
        max_level = max(levels)
        
        for level in sorted(levels):
            plt.axhline(y=-level * 3, color='lightgray', linestyle=':', alpha=0.5)
            plt.text(-15, -level * 3, f"Level {max_level-level}", fontsize=10, alpha=0.7)
        
        title = self.question[:100] + "..." if self.question else "DPIMPR Reasoning DAG"
        plt.title(f"{title}\n{len(self.G.nodes())} nodes, {len(self.G.edges())} edges, {self.trajectory_count} trajectories",
                 fontsize=14, fontweight='bold')
        
        # Create color legend for node types
        import matplotlib.patches as mpatches
        
        # Create legend patches for node types (main nodes)
        search_patch = mpatches.Patch(color=(0.3, 0.5, 0.9), label='Search Nodes')
        observation_patch = mpatches.Patch(color=(0.9, 0.6, 0.3), label='Observation Nodes')
        reasoning_patch = mpatches.Patch(color=(0.7, 0.3, 0.7), label='Reasoning Nodes')
        conclusion_patch = mpatches.Patch(color=(0.3, 0.8, 0.3), label='Conclusion Nodes')
        
        # Create legend patches for sample nodes (lighter versions)
        sample_patch = mpatches.Patch(color=(0.6, 0.6, 0.6), label='Sample Variations')
        
        # Create legend for node types
        type_legend = plt.legend(handles=[search_patch, observation_patch, reasoning_patch, conclusion_patch, sample_patch],
                               loc='upper left', bbox_to_anchor=(0.02, 0.98),
                               title='Node Types', fontsize=10)
        plt.gca().add_artist(type_legend)
        
        # Create edge type legend
        import matplotlib.lines as mlines
        
        edge_legend_elements = [
            mlines.Line2D([], [], color='black', linewidth=2, label='Main trajectory flow'),
            # mlines.Line2D([], [], color='green', linewidth=1, linestyle=':', label='Sample branches'),
            # mlines.Line2D([], [], color='orange', linewidth=1, linestyle='--', label='Sample flows'),
            # mlines.Line2D([], [], color='red', linewidth=1, linestyle='--', label='Auxiliary edges')
        ]
        
        edge_legend = plt.legend(handles=edge_legend_elements,
                               loc='upper right', bbox_to_anchor=(0.98, 0.98),
                               title='Edge Types', fontsize=10)
        plt.gca().add_artist(edge_legend)
        
        # Legend for other properties
        legend_text = (
                      "• Node size ∝ evidence count\n"
                      "• Color intensity ∝ confidence (inverse of uncertainty)\n"
                      "• Labels show: E=evidence, U=uncertainty (0-10)")
        plt.text(0.02, 0.75, legend_text, transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

# Example usage with the new TDP data format
def run_example(question, tdp_data, k =1 ):
    """Run the Australian electronic music magazine example with TDP data"""
    
    # Initialize DPIMPR
    dpimpr = DPIMPR(similarity_threshold=0.6, question=question)
    
    print("=== DPIMPR ALGORITHM WITH TDP DATA ===")
    print(f"Question: {question}")
    
    # Process TDP data
    results = dpimpr.add_tdp_data(tdp_data[:k], question)
    
    # Display comprehensive results
    print(f"\n=== COMPREHENSIVE ANALYSIS ===")
    print(f"Question: {results['question']}")
    print(f"Trajectories processed: {results['trajectories_processed']}")
    
    dag_analysis = results['dag_analysis']
    print(f"\nDAG Properties:")
    print(f"  - Valid DAG: {dag_analysis['is_dag']}")
    print(f"  - Nodes: {dag_analysis['node_count']}")
    print(f"  - Edges: {dag_analysis['edge_count']}")
    print(f"  - Convergence points: {len(dag_analysis['convergence_points'])}")
    
    for cp in dag_analysis['convergence_points']:
        print(f"    * '{cp['content'][:40]}...': {cp['evidence_count']} evidence sources (uncertainty: {cp['uncertainty']:.2f})")
    
    confidence_analysis = results['confidence_analysis']
    print(f"\nConfidence Analysis:")
    print(f"  - Final conclusion: {confidence_analysis['conclusion']}")
    print(f"  - Confidence score: {confidence_analysis['confidence']:.3f}")
    print(f"  - Evidence strength: {confidence_analysis['evidence_strength']}")
    print(f"  - Topological depth: {confidence_analysis['topological_depth']}")
    print(f"  - Trajectories contributing: {confidence_analysis['num_trajectories']}")
    
    # Additional analysis of sample integration
    print(f"\n=== SAMPLE INTEGRATION ANALYSIS ===")
    sample_nodes = [node_id for node_id in dpimpr.G.nodes() 
                   if dpimpr.G.nodes[node_id].get('is_sample_variation', False)]
    
    print(f"Sample variation nodes created: {len(sample_nodes)}")
    
    # Show evidence accumulation
    high_evidence_nodes = [node for node in dpimpr.nodes.values() if node.evidence_count > 2]
    print(f"Nodes with high evidence (>2 sources): {len(high_evidence_nodes)}")
    
    for node in high_evidence_nodes:
        print(f"  - '{node.content[:40]}...': {node.evidence_count} sources, uncertainty: {node.base_uncertainty:.2f}")
    
    return dpimpr, results

def run_example_from_file(file_path: str, k: int = 2):
    with open(file_path, 'r') as file:
        all_data = json.load(file)
    tdp_data = all_data["tdp_data"]
    question = all_data["question"]
    return run_example(question, tdp_data, k)


def demo_unified_trace_processing():
    """
    Demonstrate the unified trace processing for both HotpotQA and GSM8K formats.
    """
    print("\n" + "="*60)
    print("DEMO: Unified Trace Processing for HotpotQA and GSM8K")
    print("="*60)

    # Test GSM8K trace processing
    gsm8k_trace_file = "../results/gsm8k_20250922_093044/paths_0.json"
    try:
        with open(gsm8k_trace_file, 'r') as f:
            gsm8k_data = json.load(f)

        print("\n1. Processing GSM8K Trace:")
        print("-" * 30)
        dpimpr_gsm8k = DPIMPR(similarity_threshold=0.7)
        results_gsm8k = dpimpr_gsm8k.add_reasoning_traces(gsm8k_data)

        print(f"✓ Processed {results_gsm8k['trajectories_processed']} GSM8K trajectories")
        print(f"✓ Format: {results_gsm8k['trace_format']}")
        print(f"✓ Nodes created: {results_gsm8k['dag_analysis']['node_count']}")
        print(f"✓ Edges created: {results_gsm8k['dag_analysis']['edge_count']}")

        # Visualize GSM8K DAG
        dpimpr_gsm8k.visualize_dag(save_path="../results/vis/gsm8k_unified_demo.png")
        print("✓ GSM8K DAG saved to ../results/vis/gsm8k_unified_demo.png")

    except FileNotFoundError:
        print("⚠ GSM8K trace file not found, skipping GSM8K demo")
    except Exception as e:
        print(f"⚠ Error processing GSM8K trace: {e}")

    # Test HotpotQA trace processing
    hotpotqa_trace_file = "../results/hotpotqa_20250918_094927/paths_0.json"
    try:
        with open(hotpotqa_trace_file, 'r') as f:
            hotpotqa_data = json.load(f)

        print("\n2. Processing HotpotQA Trace:")
        print("-" * 30)
        dpimpr_hotpot = DPIMPR(similarity_threshold=0.7)
        results_hotpot = dpimpr_hotpot.add_reasoning_traces(hotpotqa_data)

        print(f"✓ Processed {results_hotpot['trajectories_processed']} HotpotQA trajectories")
        print(f"✓ Format: {results_hotpot['trace_format']}")
        print(f"✓ Nodes created: {results_hotpot['dag_analysis']['node_count']}")
        print(f"✓ Edges created: {results_hotpot['dag_analysis']['edge_count']}")

        # Visualize HotpotQA DAG
        dpimpr_hotpot.visualize_dag(save_path="../results/vis/hotpotqa_unified_demo.png")
        print("✓ HotpotQA DAG saved to ../results/vis/hotpotqa_unified_demo.png")

    except FileNotFoundError:
        print("⚠ HotpotQA trace file not found, skipping HotpotQA demo")
    except Exception as e:
        print(f"⚠ Error processing HotpotQA trace: {e}")

    print("\n3. Format Detection Demo:")
    print("-" * 30)

    # Test auto-detection
    mock_gsm8k = {
        'tdp_data': [{
            'trajectory': [
                {'action': 'Calculator[2 + 2]'},
                {'action': 'Reason[The sum is 4]'},
                {'action': 'Finish[4]'}
            ]
        }]
    }

    mock_hotpotqa = {
        'tdp_data': [{
            'trajectory': [
                {'action': 'Search[Python programming]'},
                {'action': 'Lookup[syntax]'},
                {'action': 'Finish[Python uses indentation]'}
            ]
        }]
    }

    print(f"✓ Mock GSM8K detected as: {auto_detect_trace_format(mock_gsm8k)}")
    print(f"✓ Mock HotpotQA detected as: {auto_detect_trace_format(mock_hotpotqa)}")

    print("\n" + "="*60)
    print("UNIFIED TRACE PROCESSING FEATURES:")
    print("="*60)
    print("✓ Auto-detection of trace format (HotpotQA vs GSM8K)")
    print("✓ Standardized parsing for both formats")
    print("✓ Node type classification (search, reasoning, calculation, conclusion)")
    print("✓ Unified DAG construction with format-specific handling")
    print("✓ Enhanced visualization with calculation nodes")
    print("✓ Backwards compatibility with existing DPIMPR features")

def evaluate_dag_uncertainty_auroc(trace_files: List[str],
                                 trace_format: str = None,
                                 similarity_threshold: float = 0.7,
                                 k_trajectories: int = 3) -> Dict[str, Any]:
    """
    Evaluate AUROC metrics for DAG-based uncertainty quantification across multiple examples.

    Args:
        trace_files: List of paths to trace JSON files
        trace_format: 'hotpotqa', 'gsm8k', or None for auto-detection
        similarity_threshold: Threshold for node merging in DAG construction
        k_trajectories: Number of trajectories to use per example

    Returns:
        Dictionary containing AUROC scores and detailed results
    """
    print(f"\n{'='*70}")
    print("EVALUATING DAG UNCERTAINTY QUANTIFICATION AUROC")
    print(f"{'='*70}")
    print(f"Processing {len(trace_files)} examples with {k_trajectories} trajectories each")

    all_results = []
    correctness_labels = []
    uncertainty_scores = {
        'dag_confidence': [],
        'avg_node_uncertainty': [],
        'conclusion_uncertainty': [],
        'path_diversity': [],
        'evidence_strength': [],
        'uncertainty_dag_confidence': [],
        'uncertainty_evidence_strength': [],
        'uncertainty_path_diversity': []
    }

    successful_examples = 0

    for i, trace_file in enumerate(trace_files):
        try:
            print(f"\nProcessing example {i+1}/{len(trace_files)}: {trace_file}")

            # Load trace data
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)

            # Process with DAG
            dpimpr = DPIMPR(similarity_threshold=similarity_threshold)
            results = dpimpr.add_reasoning_traces(trace_data, trace_format)

            # Extract uncertainty metrics
            uncertainty_metrics = dpimpr.extract_dag_uncertainty_metrics()

            # Get correctness label
            is_correct = results.get('is_correct', False)
            correctness_labels.append(int(is_correct))

            # Store uncertainty scores
            for metric_name in uncertainty_scores.keys():
                if metric_name in uncertainty_metrics:
                    uncertainty_scores[metric_name].append(uncertainty_metrics[metric_name])

            # Store detailed results
            example_result = {
                'file': trace_file,
                'question': results.get('question', ''),
                'predicted_answer': results.get('predicted_answer'),
                'is_correct': is_correct,
                'dag_analysis': results.get('dag_analysis', {}),
                'uncertainty_metrics': uncertainty_metrics
            }
            all_results.append(example_result)
            successful_examples += 1

            if i % 10 == 0 and i > 0:
                print(f"  ✓ Processed {i+1} examples")

        except Exception as e:
            print(f"  ⚠ Error processing {trace_file}: {e}")
            continue

    print(f"\n✓ Successfully processed {successful_examples}/{len(trace_files)} examples")

    # Calculate AUROC scores
    auroc_results = {}
    correctness_array = np.array(correctness_labels)

    if len(np.unique(correctness_array)) > 1:  # Need both correct and incorrect examples
        print(f"\nCalculating AUROC scores...")
        print(f"Correct examples: {np.sum(correctness_array)}/{len(correctness_array)}")

        for metric_name, scores in uncertainty_scores.items():
            if len(scores) == len(correctness_labels):
                try:
                    # For uncertainty metrics, higher values should correlate with incorrectness
                    # So we use scores directly (higher uncertainty = more likely incorrect)
                    auroc_score = roc_auc_score(correctness_array, -np.array(scores))  # Negative because higher uncertainty should predict incorrectness
                    auroc_results[metric_name] = auroc_score
                    print(f"  {metric_name}: AUROC = {auroc_score:.4f}")
                except ValueError as e:
                    print(f"  ⚠ Could not compute AUROC for {metric_name}: {e}")
    else:
        print("⚠ Need both correct and incorrect examples to compute AUROC")

    # Summary statistics
    summary_stats = {
        'total_examples': len(trace_files),
        'successful_examples': successful_examples,
        'accuracy': np.mean(correctness_labels) if correctness_labels else 0.0,
        'mean_uncertainty_metrics': {
            metric: np.mean(scores) if scores else 0.0
            for metric, scores in uncertainty_scores.items()
        },
        'std_uncertainty_metrics': {
            metric: np.std(scores) if scores else 0.0
            for metric, scores in uncertainty_scores.items()
        }
    }

    # Compile final results
    evaluation_results = {
        'auroc_scores': auroc_results,
        'summary_stats': summary_stats,
        'detailed_results': all_results,
        'settings': {
            'similarity_threshold': similarity_threshold,
            'k_trajectories': k_trajectories,
            'trace_format': trace_format
        }
    }

    # Display summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Examples processed: {successful_examples}/{len(trace_files)}")
    print(f"Overall accuracy: {summary_stats['accuracy']:.3f}")

    if auroc_results:
        print(f"\nAUROC Scores (Uncertainty vs Correctness):")
        for metric_name, auroc in sorted(auroc_results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {metric_name:30s}: {auroc:.4f}")

        best_metric = max(auroc_results.items(), key=lambda x: x[1])
        print(f"\nBest performing metric: {best_metric[0]} (AUROC = {best_metric[1]:.4f})")

    return evaluation_results

def run_dag_auroc_evaluation(result_dir: str,
                           pattern: str = "paths_*.json",
                           max_examples: int = 50,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run AUROC evaluation on a directory of results.

    Args:
        result_dir: Directory containing trace files
        pattern: File pattern to match (e.g., "paths_*.json")
        max_examples: Maximum number of examples to process
        **kwargs: Additional arguments for evaluate_dag_uncertainty_auroc

    Returns:
        Evaluation results dictionary
    """
    import glob
    import os

    # Find trace files
    file_pattern = os.path.join(result_dir, pattern)
    trace_files = sorted(glob.glob(file_pattern))[:max_examples]

    if not trace_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(trace_files)} trace files in {result_dir}")

    return evaluate_dag_uncertainty_auroc(trace_files, **kwargs)


if __name__ == "__main__":


    # for i in range(10):
    #     dpimpr_instance, results = run_example_from_file(f"../results/hotpotqa_20250918_094927/paths_{i}.json", k=2)
    #     plt = dpimpr_instance.visualize_dag()
    #     plt.show()
    #     plt.savefig(f"../results/vis/hotpotqa_paths_{i}.png")

    # for i in range(9):
    #     dpimpr_instance, results = run_example_from_file(f"../results/gsm8k_20250922_093044/paths_{i}.json", k=3)
    #     plt = dpimpr_instance.visualize_dag()
    #     plt.show()
    #     plt.savefig(f"../results/gsm8k_20250922_093044/gsm8k_paths_{i}.png")

    run_dag_auroc_evaluation(result_dir="../results/gsm8k_20250922_093044", pattern="paths_*.json")

    #run_dag_auroc_evaluation(result_dir="../results/hotpotqa_20250918_094927", pattern="paths_*.json")
    print("\n=== ALGORITHM BENEFITS DEMONSTRATED ===")
    print("✓ Incremental DAG construction preserving acyclic structure")
    print("✓ Semantic node merging with similarity thresholds")
    print("✓ Evidence accumulation across multiple trajectories")
    print("✓ Topological constraint checking prevents cycles")
    print("✓ Uncertainty-weighted confidence computation")
    print("✓ Cross-trajectory validation and belief updates")