#!/usr/bin/env python3
"""
Minimal 1-agent Bedrock validation script.

Creates a single BedrockAgent with Claude Haiku, asks one hardcoded MMLU-style
question, extracts the answer, and saves the result to a SQLite store.

Prerequisites:
  - AWS credentials configured (AWS_REGION, AWS_ACCESS_KEY_ID, etc.)
  - All topology_tax modules built (bedrock_client, datasets, storage, config)

Usage:
  cd /Users/sunsophi/gt-worktree && uv run python scripts/run_toy_bedrock.py
"""
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from topology_tax.bedrock_client import BedrockAgent
    from topology_tax.datasets import extract_answer
    from topology_tax.storage import ResultStore
    from topology_tax.config import CORE_MODELS, RESULTS_DIR
except ImportError as e:
    print(f"Missing module: {e}")
    print("Run this after all modules are built.")
    sys.exit(1)


QUESTION = (
    "What is the capital of France? "
    "(A) London (B) Paris (C) Berlin (D) Tokyo"
)
CORRECT_ANSWER = "B"


def main():
    model_id = CORE_MODELS["claude-haiku"]
    print(f"Model: {model_id}")
    print(f"Question: {QUESTION}")
    print()

    # Create a single agent -- uses boto3.client("bedrock-runtime") internally,
    # reading AWS_REGION from environment (no hardcoded region).
    agent = BedrockAgent(agent_id="toy-agent-0", model_id=model_id)

    # Invoke the model
    t0 = time.perf_counter()
    response = agent.invoke(QUESTION)
    wall_latency_ms = (time.perf_counter() - t0) * 1000

    # Extract the answer letter
    extracted = extract_answer(response.text)

    # Print results
    print(f"Response text: {response.text}")
    print(f"Extracted answer: {extracted}")
    print(f"Correct: {extracted == CORRECT_ANSWER}")
    print(f"Latency (API reported): {response.latency_ms} ms")
    print(f"Latency (wall clock):   {wall_latency_ms:.0f} ms")
    print(f"Input tokens:  {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print()

    # Save to SQLite store
    db_path = RESULTS_DIR / "toy_bedrock.db"
    store = ResultStore(db_path)
    store.save_result(
        dataset="toy",
        question_id="capital_of_france",
        topology="independent",
        run_id=0,
        agent_id="toy-agent-0",
        agent_position=0,
        prompt=QUESTION,
        response=response.text,
        answer_extracted=extracted,
        correct=(extracted == CORRECT_ANSWER),
        model_id=model_id,
        temperature=0.0,
        latency_ms=int(response.latency_ms),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        metadata={"wall_latency_ms": round(wall_latency_ms, 1)},
    )
    store.close()
    print(f"Result saved to {db_path}")


if __name__ == "__main__":
    main()
