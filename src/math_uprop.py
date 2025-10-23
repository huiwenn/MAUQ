"""
MATH Dataset Experiment Script for UProp
Competition mathematics problem uncertainty quantification using ReAct reasoning
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
output_dir = f"../results/math_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    filename=f'{output_dir}/math_uprop.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# ReAct prompt template for MATH dataset with calculator tool
REACT_PROMPT_TEMPLATE = """Solve a competition mathematics problem with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be two types:
(1) Calculator[expression], which evaluates a mathematical expression and returns the result.
(2) Finish[answer], which returns the final answer and finishes the task.

Here are some examples:

Question: Find the largest integer value of $n$ such that $5n + 3 < 28$.
Thought 1: I need to solve the inequality $5n + 3 < 28$ for $n$. First, I'll subtract 3 from both sides.
Action 1: Calculator[28 - 3]
Observation 1: 25
Thought 2: Now I have $5n < 25$. I need to divide both sides by 5.
Action 2: Calculator[25 / 5]
Observation 2: 5.0
Thought 3: So $n < 5$. Since I need the largest integer value of $n$, the answer is 4.
Action 3: Finish[4]

Question: If $3x + 7 = 22$, what is the value of $6x + 14$?
Thought 1: I could solve for $x$ first, but I notice that $6x + 14 = 2(3x + 7)$. Since $3x + 7 = 22$, I can directly calculate $6x + 14$.
Action 1: Calculator[2 * 22]
Observation 1: 44
Thought 2: Therefore, $6x + 14 = 44$.
Action 2: Finish[44]

Question: A right triangle has legs of length 5 and 12. What is the length of the hypotenuse?
Thought 1: I can use the Pythagorean theorem: $c^2 = a^2 + b^2$ where $a = 5$ and $b = 12$. Let me calculate $5^2 + 12^2$.
Action 1: Calculator[5**2 + 12**2]
Observation 1: 169
Thought 2: Now I need to find the square root of 169.
Action 2: Calculator[169**0.5]
Observation 2: 13.0
Thought 3: The length of the hypotenuse is 13.
Action 3: Finish[13]

Now solve this problem:
Question: {question}"""


class CalculatorTool:
    """Enhanced calculator tool for mathematical expressions including powers and roots."""

    def evaluate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Clean the expression
            expression = expression.strip()

            # Allow mathematical operations including powers and common functions
            allowed_chars = set('0123456789+-*/()., **')
            # Also allow some mathematical functions
            import math
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e
            }

            # Replace common mathematical notation
            expression = expression.replace('^', '**')  # Convert ^ to **
            expression = expression.replace('×', '*')   # Convert × to *
            expression = expression.replace('÷', '/')   # Convert ÷ to /

            # Evaluate safely with restricted namespace
            result = eval(expression, safe_dict, {})

            # Return as string, handling both int and float results
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                else:
                    # Round to reasonable precision for display
                    return str(round(result, 10))
            return str(result)

        except Exception as e:
            return f"Error: {str(e)}"


def extract_numerical_answer(response: str) -> Optional[str]:
    """Extract the final answer from the MATH dataset response as a string."""
    try:
        # Look for the final answer in common formats
        # MATH dataset often uses \\boxed{answer} format
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            answer_str = boxed_match.group(1)
        else:
            # Look for Finish[answer] format from our ReAct agent
            finish_match = re.search(r'Finish\[([^\]]+)\]', response)
            if finish_match:
                answer_str = finish_match.group(1)
            else:
                return None

        # Clean the answer string
        answer_str = answer_str.strip()
        return answer_str

    except Exception:
        return None


def normalize_math_answer(answer: str) -> str:
    """Normalize a mathematical answer for comparison."""
    if not answer:
        return ""

    # Handle fractions specially - convert \frac{a}{b} to a/b
    frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    frac_match = re.search(frac_pattern, answer)
    if frac_match:
        numerator = frac_match.group(1)
        denominator = frac_match.group(2)
        return f"{numerator}/{denominator}"

    # Remove LaTeX commands like \boxed, etc.
    normalized = re.sub(r'\\[a-zA-Z]+', '', answer)
    # Remove braces, dollar signs, backslashes
    normalized = re.sub(r'[{}$\\]', '', normalized)
    # Remove any remaining formatting except basic math symbols
    normalized = re.sub(r'[^a-zA-Z0-9./-]', '', normalized)

    # Remove whitespace
    normalized = ''.join(normalized.split())

    # Convert to lowercase
    normalized = normalized.lower()

    return normalized


def evaluate_fraction(frac_str: str) -> Optional[float]:
    """Evaluate a fraction string like '14/3' to a float."""
    try:
        if '/' in frac_str:
            parts = frac_str.strip().split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(frac_str)
    except:
        return None


def math_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches the ground truth for MATH dataset."""
    pred_str = extract_numerical_answer(predicted_answer)
    gt_str = extract_numerical_answer(ground_truth) or ground_truth

    if not pred_str or not gt_str:
        return False

    # Normalize both answers
    pred_norm = normalize_math_answer(pred_str)
    gt_norm = normalize_math_answer(gt_str)

    # Direct string comparison
    if pred_norm == gt_norm:
        return True

    # Try numerical comparison for fractions and decimals
    pred_num = evaluate_fraction(pred_norm)
    gt_num = evaluate_fraction(gt_norm)

    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < 1e-6

    return False


class MATHUProp(UProp):
    """
    UProp implementation specifically for MATH dataset with ReAct-style reasoning.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.calculator = CalculatorTool()
        super().__init__(llm_client=self.client, **kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _sample_decision(self, prompt: str, context: str, step: int) -> Optional[Decision]:
        """
        Sample a decision using the OpenAI API in ReAct format for MATH problems.
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
                    {"role": "system", "content": "You are a helpful assistant solving mathematics problems step by step. You can either continue reasoning with another thought, or take an action using Calculator[expr] or Finish[answer]."},
                    {"role": "user", "content": next_prompt}
                ],
                temperature=self.temperature,
                max_tokens=300,  # Allow more tokens for reasoning
                logprobs=True,
                top_logprobs=1,
                stop=["Thought", "Observation"]  # Stop at observation or double newline
            )

            content = response.choices[0].message.content.strip()
            print(f"Model response: {content}")

            # Check if this contains an action or is pure reasoning
            action_match = re.search(r'Action \d+: (Calculator\[[^\]]*\]|Finish\[[^\]]*\])', content)

            if action_match:
                # This step contains an action
                thought_match = re.search(r'Thought \d+: (.+?)(?=Action|\Z)', content, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                action = action_match.group(1).strip()

                # Execute action
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

    def _execute_action(self, action: str) -> str:
        """Execute the action using available tools."""
        # Parse action type
        if action.startswith("Calculator["):
            expression = re.search(r'Calculator\[(.*?)\]', action)
            if expression:
                result = self.calculator.evaluate(expression.group(1))
                return result
        elif action.startswith("Finish["):
            return "Task completed."

        return "Invalid action format."

    def _is_terminal(self, decision: Decision) -> bool:
        """Check if this decision represents a terminal state."""
        return decision.action.startswith("Finish[")

    def _update_context(self, context: str, decision: Decision) -> str:
        """Update context with new decision for MATH problems."""
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
    model: str = "gpt-4o-mini",
    Z: int = 3,
    N: int = 3,
    n_samples: int = 25,
    output_path: str = "../results",
    read_from_file_path: str = None,
):
    """
    Evaluate UProp and baseline methods on MATH dataset.
    """
    # Initialize methods
    uprop = MATHUProp(
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

    for idx, item in enumerate(tqdm(questions, desc="Evaluating MATH questions")):
        question = item['problem']
        answer = item['solution']

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
                uprop_greedy = MATHUProp(
                    api_key=api_key,
                    model=model,
                    Z=1,
                    N=1,
                    temperature=0.0
                )

                # Sample one trajectory for prediction
                tdp = uprop_greedy.sample_tdp(prompt, "")

                # Extract predicted answer
                predicted_answer = None
                if tdp.trajectory:
                    last_action = tdp.trajectory[-1].action
                    match = re.search(r'Finish\[(.*?)\]', last_action)
                    if match:
                        predicted_answer = match.group(1)

                # Check correctness
                is_correct = predicted_answer and math_correctness(predicted_answer, answer)

                uprop_scores = uprop.estimate_uncertainty(prompt, "", predicted_answer)
                uprop_scores['question'] = question
                uprop_scores['answer'] = answer
                uprop_scores['predicted_answer'] = predicted_answer
                uprop_scores['is_correct'] = is_correct
                with open(f"{output_path}/paths_{idx}.json", 'w') as f:
                    json.dump(uprop_scores, f, indent=2)

            uprop_uncertainty = uprop_scores.get('target_uncertainty', uprop_scores['total_uncertainty'])

            # Calculate baseline uncertainties
            tdps = uprop_scores['tdp_data']
            sampled_answers = [tdp['trajectory'][-1]['action'] for tdp in tdps]
            sampled_answer_logprobs = [tdp['trajectory'][-1]['log_prob'] for tdp in tdps]

            # Predictive Entropy: Use average negative log prob
            pe_uncertainty = -1 * np.mean(sampled_answer_logprobs)

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
                with open(f"{output_path}/math_{model}_results.json", 'w') as f:
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

    with open(f"{output_path}/math_{model}_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)

    return metrics


def main():
    """
    Main experiment script to evaluate MATH dataset with UProp.
    """
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Set random seed
    random.seed(42)

    # Models to evaluate
    MODELS = {
        "gpt-4": "GPT-4.1-nano"
    }

    # Load MATH dataset
    logger.info("Loading MATH dataset...")
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")

    # Results table
    results_table = pd.DataFrame()

    for model_id, model_name in MODELS.items():
        logger.info(f"\nEvaluating model: {model_name}")

        metrics = evaluate_uncertainty_methods(
            dataset_subset=dataset,
            api_key=API_KEY,
            model=model_id,
            Z=1,
            N=1,
            n_samples=1,  # Start with smaller sample for testing
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
    print("AUROC results on MATH dataset")
    print("="*60)
    print(results_table.to_string(index=False))

    # Save results table
    results_table.to_csv(f"{output_dir}/math_results.csv", index=False)
    logger.info("\nResults saved to math_results.csv")


if __name__ == "__main__":
    main()