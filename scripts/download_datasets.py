"""Pre-download and cache all datasets for the Topology Tax experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# NOTE: src/topology_tax/datasets.py does not exist yet.
# Once another agent creates it, uncomment the working version below.
# For now, this script serves as a placeholder.

def main():
    try:
        from topology_tax.datasets import load_questions
    except ImportError:
        print("ERROR: topology_tax.datasets module not found.")
        print("This script requires src/topology_tax/datasets.py which has not been created yet.")
        print("Once datasets.py is available, re-run this script to pre-download and cache all datasets.")
        sys.exit(1)

    print("Downloading and caching datasets...")

    print("\n1. MMLU (5 subjects)...")
    qs = load_questions("mmlu", n=10, seed=42)
    print(f"   Got {len(qs)} questions. Sample: {qs[0]['question_id']}")

    print("\n2. GSM8K...")
    qs = load_questions("gsm8k", n=10, seed=42)
    print(f"   Got {len(qs)} questions. Sample: {qs[0]['question_id']}")

    print("\n3. HumanEval...")
    qs = load_questions("humaneval", n=10, seed=42)
    print(f"   Got {len(qs)} questions. Sample: {qs[0]['question_id']}")

    print("\n4. HotpotQA...")
    qs = load_questions("hotpotqa", n=10, seed=42)
    print(f"   Got {len(qs)} questions. Sample: {qs[0]['question_id']}")

    print("\nAll datasets cached successfully!")

if __name__ == "__main__":
    main()
