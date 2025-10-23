"""
GSM8K Experiment Script for UProp
Math word problem uncertainty quantification using chain-of-thought reasoning
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from tqdm import tqdm
import openai
from openai import OpenAI
import re
import string
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from uprop import UProp, Decision, TDP, SemanticEntropy
from dotenv import load_dotenv

load_dotenv()

# Setup logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = f"../results/gsm8k_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    filename=f'{output_dir}/gsm8k_uprop.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# ReAct prompt template for GSM8K with tools
REACT_PROMPT_TEMPLATE = """Solve a math word problem with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Calculator[expression], which evaluates a mathematical expression and returns the result.
(2) Finish[answer], which returns the final numerical answer and finishes the task.

Here are some examples:

Question: John has 12 apples. He gives 3 apples to his sister and 2 apples to his brother. How many apples does John have left?
Thought 1: I need to find out how many apples John gives away in total, then subtract from his initial amount. John gives 3 apples to his sister and 2 apples to his brother. I need to calculate the total apples given away.
Action 1: Calculator[3 + 2]
Observation 1: 5
Thought 2: Now I can calculate how many apples John has left.
Action 2: Calculator[12 - 5]
Observation 2: 7
Thought 3: John has 7 apples left.
Action 3: Finish[7]

Question: A bakery sells 24 cupcakes in the morning and 36 cupcakes in the afternoon. If each cupcake costs $3, how much money did the bakery make from cupcakes?
Thought 1: I need to find the total number of cupcakes sold, then multiply by the price per cupcake.
Action 1: Calculator[24 + 36]
Observation 1: 60
Thought 2: The bakery sold 60 cupcakes total. Now I need to calculate the total revenue.
Action 2: Calculator[60 * 3]
Observation 2: 180
Thought 3: The bakery made $180 from cupcakes.
Action 3: Finish[180]

Now solve this problem:
Question: {question}"""

class CalculatorTool:
    """Simple calculator tool for mathematical expressions."""

    def evaluate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Clean the expression
            expression = expression.strip()

            # Only allow basic mathematical operations and numbers
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"

            # Replace any text operators with symbols if needed
            expression = expression.replace('×', '*').replace('÷', '/')

            # Evaluate safely
            result = eval(expression)

            # Return as string, handling both int and float results
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)

        except Exception as e:
            return f"Error: {str(e)}"

def extract_numerical_answer(response: str) -> Optional[float]:
    """Extract the final numerical answer from the response."""
    # Look for Finish[number] pattern first (ReAct format)
    finish_match = re.search(r'Finish\[([+-]?\d*\.?\d+)\]', response, re.IGNORECASE)
    if finish_match:
        try:
            return float(finish_match.group(1))
        except ValueError:
            pass

    # Look for "Answer: <number>" pattern
    answer_match = re.search(r'Answer:\s*([+-]?\d*\.?\d+)', response, re.IGNORECASE)
    if answer_match:
        try:
            return float(answer_match.group(1))
        except ValueError:
            pass

    # Look for dollar amounts
    dollar_matches = re.findall(r'\$([+-]?\d*\.?\d+)', response)
    if dollar_matches:
        try:
            return float(dollar_matches[-1])
        except ValueError:
            pass

    # Look for the last number in the response
    number_matches = re.findall(r'([+-]?\d*\.?\d+)', response)
    if number_matches:
        try:
            return float(number_matches[-1])
        except ValueError:
            pass

    return None

def extract_gsm8k_answer(answer_text: str) -> float:
    """Extract the numerical answer from GSM8K answer format."""
    # GSM8K answers end with #### <number>
    match = re.search(r'####\s*([+-]?\d*\.?\d+)', answer_text)
    if match:
        return float(match.group(1))

    # Fallback: look for the last number in the text
    numbers = re.findall(r'([+-]?\d*\.?\d+)', answer_text)
    if numbers:
        return float(numbers[-1])

    raise ValueError(f"Could not extract numerical answer from: {answer_text}")

def gsm8k_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """Check if the predicted answer is correct for GSM8K."""
    try:
        pred_num = extract_numerical_answer(predicted_answer)
        true_num = extract_gsm8k_answer(ground_truth)

        if pred_num is None or true_num is None:
            return False

        # Allow small floating point differences
        return abs(pred_num - true_num) < 1e-6
    except (ValueError, TypeError):
        return False

class GSM8KUProp(UProp):
    """
    UProp implementation specifically for GSM8K with ReAct reasoning and tools.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.calculator = CalculatorTool()
        super().__init__(llm_client=self.client, **kwargs)

    def _extract_thought_action(self, text: str) -> Tuple[str, str]:
        """Extract thought and action from model output."""
        thought_match = re.search(r"Thought \d+:(.*?)(?=Action|$)", text, re.DOTALL)
        action_match = re.search(r"Action \d+:(.*?)(?=Observation|Thought|$)", text, re.DOTALL)

        # Handle case where model doesn't follow exact format
        if not action_match:
            action_match = re.search(r"Action:(.*?)(?=Observation|Thought|$)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""

        # Clean up action - remove any trailing text after the action
        if action and (action.startswith("Calculator[") or action.startswith("Finish[") or action.startswith("Reason[")):
            # Find the closing bracket and truncate there
            for prefix in ["Calculator[", "Finish[", "Reason["]:
                if action.startswith(prefix):
                    bracket_match = re.search(rf'{re.escape(prefix)}([^\]]*)\]', action)
                    if bracket_match:
                        action = bracket_match.group(0)
                        break

        return thought, action

    def _execute_action(self, action: str) -> str:
        """Execute the action using available tools."""
        # Parse action type
        if action.startswith("Reason["):
            description = re.search(r'Reason\[(.*?)\]', action)
            if description:
                return f"Reasoning: {description.group(1)}"
        elif action.startswith("Calculator["):
            expression = re.search(r'Calculator\[(.*?)\]', action)
            if expression:
                result = self.calculator.evaluate(expression.group(1))
                return f"Calculation result: {result}"
        elif action.startswith("Finish["):
            answer = re.search(r'Finish\[(.*?)\]', action)
            if answer:
                return f"Final answer: {answer.group(1)}"

        return "Invalid action format."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _sample_decision(self, prompt: str, context: str, step: int) -> Optional[Decision]:
        """
        Sample a decision using the OpenAI API with ReAct format and tools.
        Allows for reasoning-only steps (no action) or action steps.
        """
        # Combine prompt with current context
        full_prompt = prompt if step == 0 else f"{prompt}\n\n{context}"

        try:
            # Generate next step - could be reasoning or action
            next_prompt = f"{full_prompt}\nThought {step + 1}:"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that solves math problems step by step. You can either continue reasoning with another thought, or take an action using Calculator[expr], Reason[description], or Finish[answer]."},
                    {"role": "user", "content": next_prompt}
                ],
                temperature=self.temperature,
                max_tokens=150,  # Allow more tokens for reasoning
                logprobs=True,
                top_logprobs=1,
                stop=["Observation", "\n\n"]  # Stop at observation or double newline
            )

            content = response.choices[0].message.content.strip()

            # Check if this contains an action or is pure reasoning
            action_match = re.search(r'Action \d+: (Calculator\[[^\]]*\]|Finish\[[^\]]*\]|Reason\[[^\]]*\])', content)

            if action_match:
                # This step contains an action
                thought_match = re.search(r'Thought \d+: (.+?)(?=Action|\Z)', content, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                action = action_match.group(1).strip()

                # Execute action to get observation
                observation = self._execute_action(action)
            else:
                # This is a reasoning-only step
                thought_match = re.search(r'Thought \d+: (.+)', content, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else content
                action = "Reason"  # Special action type for reasoning steps
                observation = "Continuing reasoning..."

            # Calculate log probability
            log_prob = 0.0
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
                    if token_info.logprob is not None:
                        log_prob += token_info.logprob

            return Decision(
                step=step,
                action=action,
                reasoning=thought,
                observation=observation,
                log_prob=log_prob
            )

        except Exception as e:
            logger.error(f"Error sampling decision: {e}")
            return None

    def _is_terminal(self, decision: Decision) -> bool:
        """Check if this decision represents a terminal state."""
        return decision.action.startswith("Finish")

    def _update_context(self, context: str, decision: Decision) -> str:
        """Update context with new decision for GSM8K ReAct format."""
        step_num = decision.step + 1
        new_context = f"Thought {step_num}: {decision.reasoning}"

        # Only add action and observation if this wasn't a reasoning-only step
        if decision.action != "Reason":
            new_context += f"\nAction {step_num}: {decision.action}"
            new_context += f"\nObservation {step_num}: {decision.observation}"

        if context:
            return f"{context}\n{new_context}"
        return new_context

def evaluate_uncertainty_methods(
    dataset_subset,
    api_key: str,
    model: str = "gpt-3.5-turbo",
    Z: int = 3,
    N: int = 3,
    n_samples: int = 25,
    output_path: str = "../results",
    read_from_file_path: str = None,
):
    """
    Evaluate UProp and baseline methods on GSM8K.
    """
    # Initialize methods
    uprop = GSM8KUProp(
        api_key=api_key,
        model=model,
        Z=Z,  # Number of TDP samples
        N=N,  # Number of per-step samples
        temperature=0.8
    )

    # Baseline: Semantic Entropy (simplified)
    se_baseline = SemanticEntropy(n_samples=5)

    # Results storage
    results = {
        'questions': [],
        'ground_truth': [],
        'predictions': [],
        'correct': [],
        'uprop_uncertainty': [],
        'se_uncertainty': [],
        'pe_uncertainty': []  # Predictive Entropy baseline
    }

    # Sample questions from dataset
    questions = random.sample(list(dataset_subset), min(n_samples, len(dataset_subset)))

    for idx, item in enumerate(tqdm(questions, desc="Evaluating questions")):
        question = item['question']
        answer = item['answer']

        logger.info(f"Processing question {idx+1}/{n_samples}: {question[:50]}...")
        semantic_entropy = SemanticEntropy(n_samples=5)

        # Format prompt
        prompt = REACT_PROMPT_TEMPLATE.format(question=question)

        try:
            # Calculate UProp uncertainty
            if read_from_file_path is not None:
                with open(f"{read_from_file_path}/paths_{idx}.json", 'r') as f:
                    uprop_scores = json.load(f)
                predicted_answer = uprop_scores['predicted_answer']
                is_correct = uprop_scores['is_correct']
            else:
                # Get prediction with greedy decoding (temperature=0)
                uprop_greedy = GSM8KUProp(
                    api_key=api_key,
                    model=model,
                    Z=1,
                    N=1,
                    temperature=0.0
                )

                # Sample one trajectory for prediction
                tdp = uprop_greedy.sample_tdp(prompt, "")

                # Extract predicted answer from the final action
                predicted_answer = None
                if tdp.trajectory:
                    # Look for the final Finish action
                    for decision in reversed(tdp.trajectory):
                        if decision.action.startswith("Finish"):
                            predicted_answer = extract_numerical_answer(decision.action)
                            break

                    # Fallback: try to extract from the last action or reasoning
                    if predicted_answer is None and tdp.trajectory:
                        last_decision = tdp.trajectory[-1]
                        predicted_answer = extract_numerical_answer(last_decision.action)
                        if predicted_answer is None:
                            predicted_answer = extract_numerical_answer(last_decision.reasoning)

                # Check correctness
                is_correct = predicted_answer is not None and gsm8k_correctness(
                    str(predicted_answer), answer
                )

                uprop_scores = uprop.estimate_uncertainty(prompt, "", str(predicted_answer) if predicted_answer else "")
                uprop_scores['question'] = question
                uprop_scores['answer'] = answer
                uprop_scores['predicted_answer'] = predicted_answer
                uprop_scores['is_correct'] = is_correct

                with open(f"{output_path}/paths_{idx}.json", 'w') as f:
                    json.dump(uprop_scores, f, indent=2)

            uprop_uncertainty = uprop_scores.get('target_uncertainty', uprop_scores['total_uncertainty'])

            # Calculate baseline uncertainties
            tdps = uprop_scores['tdp_data']
            sampled_answers = []
            sampled_answer_logprobs = []

            for tdp in tdps:
                if tdp['trajectory']:
                    # Extract answer from the final trajectory step
                    extracted_answer = None
                    trajectory_logprobs = []

                    # Look for Finish action first
                    for step in reversed(tdp['trajectory']):
                        if step['action'].startswith("Finish"):
                            extracted_answer = extract_numerical_answer(step['action'])
                            break

                    # Collect log probabilities from all steps
                    for step in tdp['trajectory']:
                        trajectory_logprobs.append(step['log_prob'])

                    # Fallback: try to extract from last step
                    if extracted_answer is None and tdp['trajectory']:
                        last_step = tdp['trajectory'][-1]
                        extracted_answer = extract_numerical_answer(last_step['action'])
                        if extracted_answer is None:
                            extracted_answer = extract_numerical_answer(last_step['reasoning'])

                    sampled_answers.append(str(extracted_answer) if extracted_answer else "")
                    # Use sum of log probabilities for the entire trajectory
                    sampled_answer_logprobs.append(sum(trajectory_logprobs) if trajectory_logprobs else 0)

            # Predictive Entropy: Use average negative log prob
            pe_uncertainty = -1 * np.mean(sampled_answer_logprobs) if sampled_answer_logprobs else 0

            # Semantic Entropy: Would need semantic clustering
            se_uncertainty = semantic_entropy.calculate(sampled_answers, sampled_answer_logprobs)

            # Store results
            results['questions'].append(question)
            results['ground_truth'].append(answer)
            results['predictions'].append(predicted_answer)
            results['correct'].append(is_correct)
            results['uprop_uncertainty'].append(uprop_uncertainty)
            results['se_uncertainty'].append(se_uncertainty)
            results['pe_uncertainty'].append(pe_uncertainty)

            # Save intermediate results
            if idx % 10 == 0:
                with open(f"{output_path}/gsm8k_{model}_results.json", 'w') as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            continue

        # Rate limiting
        time.sleep(1)

    # Calculate AUROC scores
    correct_labels = np.array(results['correct']).astype(int)

    # Invert uncertainties (higher uncertainty = lower confidence = more likely incorrect)
    uprop_scores = 0 - np.array(results['uprop_uncertainty'])
    se_scores = 0 - np.array(results['se_uncertainty'])
    pe_scores = 0 - np.array(results['pe_uncertainty'])

    metrics = {}

    if len(np.unique(correct_labels)) > 1:  # Need both classes for AUROC
        metrics['uprop_auroc'] = roc_auc_score(correct_labels, uprop_scores)
        metrics['se_auroc'] = roc_auc_score(correct_labels, se_scores)
        metrics['pe_auroc'] = roc_auc_score(correct_labels, pe_scores)

    metrics['accuracy'] = np.mean(results['correct'])
    metrics['n_samples'] = len(results['correct'])

    # Save final results
    final_results = {
        'metrics': metrics,
        'detailed_results': results
    }

    with open(f"{output_path}/gsm8k_{model}_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)

    return metrics

def main():
    """
    Main experiment script to evaluate UProp on GSM8K.
    """
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Models to evaluate
    MODELS = {
        #"gpt-3.5-turbo": "GPT-3.5-Turbo",
        "gpt-4": "gpt-4.1-nano" 
    }

    # Load GSM8K dataset
    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    # Results table
    results_table = pd.DataFrame()

    for model_id, model_name in MODELS.items():
        logger.info(f"\nEvaluating model: {model_name}")

        metrics = evaluate_uncertainty_methods(
            dataset_subset=dataset,
            api_key=API_KEY,
            model=model_id,
            Z=3,
            N=3,
            n_samples=25,  # Use more for production runs
            output_path=output_dir,
        )

        # Add to results table
        row = {
            'Model': model_name,
            'Success Rate': metrics.get('accuracy', 0),
            'UProp (ours)': metrics.get('uprop_auroc', 0),
            'SE': metrics.get('se_auroc', 0),
            'PE': metrics.get('pe_auroc', 0)
        }

        results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)

        logger.info(f"Results for {model_name}:")
        logger.info(f"  Success Rate: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"  UProp AUROC: {metrics.get('uprop_auroc', 0):.3f}")
        logger.info(f"  SE AUROC: {metrics.get('se_auroc', 0):.3f}")
        logger.info(f"  PE AUROC: {metrics.get('pe_auroc', 0):.3f}")

    # Display results table
    print("\n" + "="*60)
    print("AUROC results on GSM8K")
    print("="*60)
    print(results_table.to_string(index=False))

    # Save results table
    results_table.to_csv(f"{output_dir}/gsm8k_table_results.csv", index=False)
    logger.info(f"\nResults saved to {output_dir}/gsm8k_table_results.csv")

if __name__ == "__main__":
    main()