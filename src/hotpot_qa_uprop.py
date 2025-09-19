"""
HotpotQA Experiment Script for UProp
Replicates results from Table 1 of the paper
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
import wikipedia
import re
from fuzzywuzzy import fuzz
import string
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from uprop import UProp, Decision, TDP, SemanticEntropy

# Import UProp implementation
# from uprop_implementation import UProp, Decision, TDP, SemanticEntropy
import os, time
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = f"../results/hotpotqa_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    filename=f'{output_dir}/hotpotqa_uprop.log',  # Specify the log file name
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
    filemode='a'  # 'a' for append (default), 'w' for overwrite
)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ReAct prompt template for HotpotQA
REACT_PROMPT_TEMPLATE = """Solve a question answering task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions...
Thought 4: I need to search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought 5: The answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Now solve this question:
Question: {question}"""

class WikipediaEnvironment:
    """Simulated Wikipedia environment for ReAct agent."""
    
    def __init__(self):
        self.current_page = None
        self.current_text = ""
        
    def search(self, entity: str) -> str:
        """Search Wikipedia for an entity."""
        try:
            # Limit search to prevent long delays
            results = wikipedia.search(entity, results=3)
            if not results:
                return f"No results found for '{entity}'."
            
            # Try to get the page
            try:
                page = wikipedia.page(results[0])
                self.current_page = page
                # Get first paragraph (limit to 500 chars for efficiency)
                summary = wikipedia.summary(results[0], sentences=3)
                self.current_text = summary
                return summary[:500]
            except wikipedia.DisambiguationError as e:
                # Return disambiguation options
                options = e.options[:5]
                return f"Disambiguation page. Similar entities: {', '.join(options)}"
            except:
                return f"Could not retrieve page for '{results[0]}'. Similar entities: {', '.join(results[1:])}"
        except:
            return f"Search failed for '{entity}'."
    
    def lookup(self, keyword: str) -> str:
        """Look up keyword in current text."""
        if not self.current_text:
            return "No current page to search in."
        
        sentences = self.current_text.split('.')
        for i, sent in enumerate(sentences):
            if keyword.lower() in sent.lower():
                # Return this sentence and the next one if available
                result = sent.strip()
                if i + 1 < len(sentences):
                    result += ". " + sentences[i + 1].strip()
                return result
        
        return f"No sentence containing '{keyword}' found in current page."

def hotpot_correctness(predicted_answer: str, answer: str) -> bool:
    """Check if the predicted answer is correct."""

    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    normal_pred = normalize_answer(predicted_answer)
    normal_answer = normalize_answer(answer)

    exact_match = normal_pred == normal_answer
    contains = (normal_answer in normal_pred) or (normal_pred in normal_answer)
    similar = fuzz.ratio(normal_pred, normal_answer) / 100.0 > 0.85

    return exact_match or contains or similar

class HotpotQAUProp(UProp):
    """
    UProp implementation specifically for HotpotQA with ReAct-style reasoning.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.wiki_env = WikipediaEnvironment()
        super().__init__(llm_client=self.client, **kwargs)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _sample_decision(self, prompt: str, context: str, step: int) -> Optional[Decision]:
        """
        Sample a decision using the OpenAI API in ReAct format.
        """
        # Combine prompt with current context
        full_prompt = prompt if step == 0 else f"{prompt}\n\n{context}"
        
        try:
            # Call OpenAI API with logprobs
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant solving questions step by step."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=150,
                logprobs=True,
                top_logprobs=1
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Extract Thought and Action using regex
            thought_match = re.search(r'Thought \d+: (.+?)(?=Action|\Z)', content, re.DOTALL)
            action_match = re.search(r'Action \d+: (.+?)(?=Thought|\Z)', content, re.DOTALL)
            
            if not action_match:
                return None
            
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip()
            
            # Execute action in environment
            observation = self._execute_action(action)
            
            # Calculate log probability (sum of token log probs)
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
        """Execute the action in the Wikipedia environment."""
        # Parse action type
        if action.startswith("Search["):
            entity = re.search(r'Search\[(.*?)\]', action)
            if entity:
                return f"Observation: {self.wiki_env.search(entity.group(1))}"
        elif action.startswith("Lookup["):
            keyword = re.search(r'Lookup\[(.*?)\]', action)
            if keyword:
                return f"Observation: {self.wiki_env.lookup(keyword.group(1))}"
        elif action.startswith("Finish["):
            return "Task completed."
        
        return "Invalid action format."
    
    def _update_context(self, context: str, decision: Decision) -> str:
        """Update context with new decision for HotpotQA."""
        step_num = decision.step + 1
        new_context = f"Thought {step_num}: {decision.reasoning}\n"
        new_context += f"Action {step_num}: {decision.action}\n"
        new_context += f"Observation {step_num}: {decision.observation}"
        
        if context:
            return f"{context}\n{new_context}"
        return new_context


def evaluate_uncertainty_methods(
    dataset_subset,
    api_key: str,
    model: str = "gpt-3.5-turbo",
    Z: int = 3,
    N: int = 3,
    n_samples: int = 25,  # Number of questions to evaluate
    output_path: str = "../results",
    read_from_file_path: str = None,
):
    """
    Evaluate UProp and baseline methods on HotpotQA.
    """
    # Initialize methods
    uprop = HotpotQAUProp(
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
                uprop_greedy = HotpotQAUProp(
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
                
                # Check correctness (simplified - exact match)
                is_correct = predicted_answer and hotpot_correctness(predicted_answer, answer)
            
                uprop_scores = uprop.estimate_uncertainty(prompt, "", predicted_answer)
                uprop_scores['question'] = question
                uprop_scores['answer'] = answer
                uprop_scores['predicted_answer'] = predicted_answer
                uprop_scores['is_correct'] = is_correct
                with open(f"{output_path}/paths_{idx}.json", 'w') as f:
                    json.dump(uprop_scores, f, indent=2)

            uprop_uncertainty = uprop_scores.get('target_uncertainty', uprop_scores['total_uncertainty'])
            
            # Calculate baseline uncertainties (simplified for demonstration)
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
                with open(f"{output_path}/hotpotqa_{model}_results.json", 'w') as f:
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
    
    with open(f"{output_path}/hotpotqa_{model}_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return metrics


def main():
    """
    Main experiment script to replicate Table 1 results for HotpotQA.
    """
    # Configuration
    API_KEY = "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" #os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # set random seed
    random.seed(42)

    # Models to evaluate (as per Table 1)
    MODELS = {
        #"gpt-3.5-turbo": "GPT-3.5-Turbo",
        "gpt-4": "gpt-4.1-nano"  # Note: Paper uses GPT-4.1-Nano which may not be publicly available
        #"gpt-5": "gpt-5-nano"
    }
    
    # Load HotpotQA dataset
    logger.info("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    
    # Results table
    results_table = pd.DataFrame()
    read_from_file_path = "../results/hotpotqa_20250918_084946"
    for model_id, model_name in MODELS.items():
        logger.info(f"\nEvaluating model: {model_name}")
        
        metrics = evaluate_uncertainty_methods(
            dataset_subset=dataset,
            api_key=API_KEY,
            model=model_id,
            Z=2,
            N=3,
            n_samples=20,  # Use 500 for full replication as in paper
            output_path=output_dir,
            #read_from_file_path=read_from_file_path
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
    
    # Display results table (similar to Table 1 in paper)
    print("\n" + "="*60)
    print("Table 1: AUROC results on HotpotQA")
    print("="*60)
    print(results_table.to_string(index=False))
    
    # Save results table
    results_table.to_csv(f"{output_dir}/hotpotqa_table1_results.csv", index=False)
    logger.info("\nResults saved to hotpotqa_table1_results.csv")


if __name__ == "__main__":
    main()