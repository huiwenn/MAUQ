from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

BEDROCK_REGION = "us-east-1"

CORE_MODELS = {
    "claude-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "llama-70b": "meta.llama3-3-70b-instruct-v1:0",
    "mistral-large": "mistral.mistral-large-2402-v1:0",
    "command-r-plus": "cohere.command-r-plus-v1:0",
    "llama4-maverick": "meta.llama4-maverick-17b-instruct-v1:0",
}

EXTENDED_MODELS = {
    "claude-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "ministral-8b": "mistral.ministral-3-8b-instruct",
    "jamba-large": "ai21.jamba-1-5-large-v1:0",
    "llama-3b": "meta.llama3-2-3b-instruct-v1:0",
}

TOPOLOGY_NAMES = [
    "complete", "star", "ring", "chain", "binary_tree",
    "erdos_renyi", "small_world", "independent",
    "context_matched", "sparse_random",
]

PRIMARY_DATASETS = ["mmlu", "gsm8k", "humaneval", "hotpotqa"]

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_RUNS_PER_CONDITION = 20
CONSENSUS_THRESHOLD = 0.8
