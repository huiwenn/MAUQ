from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

BEDROCK_REGION = "us-west-2"

CORE_MODELS = {
    "claude-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "llama-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "mistral-large": "us.mistral.pixtral-large-2502-v1:0",
    "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "llama4-scout": "us.meta.llama4-scout-17b-instruct-v1:0",
}

EXTENDED_MODELS = {
    "claude-haiku-3": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "llama-3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
    "llama-3b": "us.meta.llama3-2-3b-instruct-v1:0",
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
