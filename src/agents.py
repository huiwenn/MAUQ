
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct agent implementation for multi-hop QA.
    Based on Yao et al., 2022.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
        max_steps: int = 7,
        temperature: float = 0.7
    ):
        """
        Initialize ReAct agent with LLM backbone.
        
        Args:
            model_name: HuggingFace model identifier
            max_steps: Maximum reasoning steps
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.max_steps = max_steps
        self.temperature = temperature
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _extract_thought_action(self, text: str) -> Tuple[str, str]:
        """Extract thought and action from model output."""
        thought_match = re.search(r"Thought \d+:(.*?)(?=Action|$)", text, re.DOTALL)
        action_match = re.search(r"Action \d+:(.*?)(?=Observation|Thought|$)", text, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
    
    def _wikipedia_search(self, query: str) -> str:
        """Search Wikipedia for information."""
        try:
            # Clean the search query
            clean_query = re.sub(r"^(Search|search)\[|\]$", "", query).strip()
            
            # Search Wikipedia
            search_results = wikipedia.search(clean_query, results=3)
            
            if not search_results:
                return "No results found."
                
            # Get summary of first result
            try:
                page = wikipedia.page(search_results[0])
                # Return first 500 characters
                return page.summary[:500] + "..."
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation, return first option
                page = wikipedia.page(e.options[0])
                return page.summary[:500] + "..."
            except:
                return f"Found pages: {', '.join(search_results[:3])}"
                
        except Exception as e:
            return f"Search error: {str(e)}"
            
    def _execute_action(self, action: str) -> str:
        """Execute an action and return observation."""
        # Check if it's a search action
        if "search" in action.lower():
            return self._wikipedia_search(action)
        elif "finish" in action.lower():
            # Extract answer from finish action
            answer_match = re.search(r"Finish\[(.*?)\]", action, re.IGNORECASE)
            if answer_match:
                return f"Answer: {answer_match.group(1)}"
            return "Answer extracted."
        else:
            return "Invalid action format."
            
    def generate_with_logprobs(self, prompt: str, max_length: int = 150):
        """
        Generate text with log probabilities for uncertainty estimation.
        
        Returns:
            text: Generated text
            logprobs: Log probabilities for each token
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Generate with logprobs
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            # Get generated tokens
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Compute log probabilities
            logprobs = []
            if outputs.scores:
                for i, score in enumerate(outputs.scores):
                    # Get log probability of selected token
                    token_id = generated_ids[i]
                    log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                    logprobs.append(log_prob)
                    
        return generated_text, np.array(logprobs) if logprobs else None
        
    def solve_question(self, question: str) -> Tuple[str, List[AgentStep]]:
        """
        Solve a question using ReAct framework.
        
        Returns:
            answer: Final answer
            trajectory: List of reasoning steps
        """
        trajectory = []
        
        # Initial prompt
        prompt = f"""Question: {question}

        You need to answer this question by searching for information step by step.
        Use the following format:

        Thought 1: [your reasoning about what to search]
        Action 1: Search[entity or query to search]
        Observation 1: [search results will be provided here]
        ...
        Thought N: [final reasoning]
        Action N: Finish[final answer]

        Let's begin:

        """
        
        for step_num in range(1, self.max_steps + 1):
            # Generate thought and action
            generation, logprobs = self.generate_with_logprobs(prompt)
            
            # Extract thought and action
            thought, action = self._extract_thought_action(generation)
            
            if not thought or not action:
                # If extraction failed, try to continue
                thought = generation[:100] if generation else "Continuing reasoning..."
                action = "Search[relevant information]"
                
            # Execute action to get observation
            observation = self._execute_action(action)
            
            # Create step object
            step = AgentStep(
                thought=thought,
                action=action,
                observation=observation,
                thought_logprobs=logprobs[:len(thought.split())] if logprobs is not None else None,
                action_logprobs=logprobs[len(thought.split()):] if logprobs is not None else None
            )
            trajectory.append(step)
            
            # Update prompt
            prompt += f"Thought {step_num}: {thought}\n"
            prompt += f"Action {step_num}: {action}\n"
            prompt += f"Observation {step_num}: {observation}\n\n"
            
            # Check if we have a final answer
            if "finish" in action.lower():
                answer_match = re.search(r"Finish\[(.*?)\]", action, re.IGNORECASE)
                if answer_match:
                    return answer_match.group(1), trajectory
                    
        # If no answer found, extract from last observation
        final_answer = trajectory[-1].observation if trajectory else "Unable to find answer"
        return final_answer, trajectory
