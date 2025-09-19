"""
Experiment script to replicate SAUP results on HotpotQA dataset.
Implements ReAct agent with SAUP uncertainty estimation.
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import wikipedia
from sklearn.metrics import roc_auc_score
import re
from dataclasses import dataclass, asdict
import logging

# Import our SAUP implementation
from saup import SAUP, AgentStep, SingleStepUncertainty
from agents import ReActAgentß

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_on_hotpotqa(
    num_samples: int = 2000,
    model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
    surrogate_type: str = "hmm"
):
    """
    Evaluate SAUP on HotpotQA dataset.
    
    Args:
        num_samples: Number of samples to evaluate
        model_name: LLM model to use
        surrogate_type: Type of SAUP surrogate
    """
    # Load dataset
    logger.info("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    
    # Sample subset for evaluation
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sampled_data = dataset.select(indices)
    
    # Initialize agent and SAUP
    agent = ReActAgent(model_name=model_name)
    
    # For CHMM training, we'd need labeled data
    # Here we'll use a subset for demonstration
    training_samples = []
    logger.info("Preparing CHMM training data...")
    for i in range(min(100, len(sampled_data))):
        sample = sampled_data[i]
        question = sample["question"]
        
        # Get agent trajectory (for training)
        _, trajectory = agent.solve_question(question)
        
        training_samples.append({
            "question": question,
            "trajectory": trajectory,
            "label": 1 if random.random() > 0.5 else 0  # Placeholder labels
        })
        
    # Initialize SAUP
    saup = SAUP(
        surrogate_type=surrogate_type,
        use_chmm_training_data=training_samples if surrogate_type == "hmm" else None
    )
    
    # Evaluation
    results = []
    uncertainties = []
    correctness = []
    
    logger.info(f"Evaluating on {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        sample = sampled_data[i]
        question = sample["question"]
        true_answer = sample["answer"]
        
        # Get agent's answer and trajectory
        predicted_answer, trajectory = agent.solve_question(question)
        
        # Compute uncertainty
        uncertainty_result = saup.process_trajectory(question, trajectory)
        
        # Check correctness (simplified - checking if true answer appears in prediction)
        is_correct = true_answer.lower() in predicted_answer.lower()
        
        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "uncertainty": uncertainty_result["agent_uncertainty"],
            "is_correct": is_correct
        })
        
        uncertainties.append(uncertainty_result["agent_uncertainty"])
        correctness.append(0 if is_correct else 1)  # 0 for correct, 1 for incorrect
        
    # Compute AUROC
    auroc = roc_auc_score(correctness, uncertainties)
    
    # Compute accuracy
    accuracy = sum([r["is_correct"] for r in results]) / len(results)
    
    logger.info(f"Results for {surrogate_type} surrogate:")
    logger.info(f"AUROC: {auroc:.3f}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    
    # Save results
    output_file = f"hotpotqa_results_{surrogate_type}.json"
    with open(output_file, "w") as f:
        json.dump({
            "auroc": auroc,
            "accuracy": accuracy,
            "num_samples": num_samples,
            "model": model_name,
            "surrogate_type": surrogate_type,
            "detailed_results": results[:100]  # Save first 100 for analysis
        }, f, indent=2)
        
    logger.info(f"Results saved to {output_file}")
    
    return auroc, accuracy


def run_baseline_comparisons():
    """Run baseline methods for comparison (from Table 1)."""
    
    baselines = {
        "predictive_entropy": 0.631,  # From paper
        "likelihood": 0.653,
        "normalized_entropy": 0.664,
        "p_true": 0.601,
        "semantic_entropy": 0.702
    }
    
    logger.info("\nBaseline results from paper (LLAMA3 8B on HotpotQA):")
    for method, auroc in baselines.items():
        logger.info(f"{method}: {auroc:.3f}")
        

def main():
    """Main experiment runner."""
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("Starting SAUP evaluation on HotpotQA")
    
    # Run with different surrogates
    surrogate_types = ["plain", "position", "hybrid"]  # "hmm" requires more setup
    
    results = {}
    for surrogate in surrogate_types:
        logger.info(f"\nEvaluating SAUP-{surrogate.upper()}...")
        try:
            auroc, accuracy = evaluate_on_hotpotqa(
                num_samples=100,  # Start with smaller sample for testing
                model_name="meta-llama/Llama-3.2-1B-Instruct",  # Smaller model for testing
                surrogate_type=surrogate
            )
            results[surrogate] = {"auroc": auroc, "accuracy": accuracy}
        except Exception as e:
            logger.error(f"Error with {surrogate}: {str(e)}")
            
    # Display final results
    logger.info("\n" + "="*50)
    logger.info("Final Results:")
    logger.info("="*50)
    
    run_baseline_comparisons()
    
    logger.info("\nSAUP Results:")
    for surrogate, metrics in results.items():
        logger.info(f"SAUP-{surrogate.upper()}: AUROC={metrics['auroc']:.3f}, Acc={metrics['accuracy']:.3f}")
        

if __name__ == "__main__":
    main()