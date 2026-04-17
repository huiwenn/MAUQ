import re
import hashlib
from typing import Optional


def normalize_answer(answer: str) -> str:
    text = answer.strip().lower()
    # Strip common answer prefixes
    text = re.sub(
        r"^(?:the answer is|answer is|answer:)\s*", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def extract_answer(text: str, task_type: str = "mcq") -> str:
    if task_type == "mcq":
        match = re.search(
            r"(?:the answer is|answer:|)\s*\(?([A-D])\)?", text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()
        return text.strip()
    elif task_type == "numeric":
        match = re.search(r"####\s*([\d,.\-]+)", text)
        if match:
            return match.group(1).replace(",", "")
        match = re.search(r"(?:answer is|=)\s*([\d,.\-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
        numbers = re.findall(r"[\d,]+\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return text.strip()
    elif task_type == "code":
        if "```" in text:
            match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text.strip()
    else:
        return text.strip()


def _make_question_id(dataset: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{dataset}:{idx}:{text[:100]}".encode()).hexdigest()[:8]
    return f"{dataset}_{idx}_{h}"


def load_questions(
    dataset: str, n: Optional[int] = None, seed: int = 42
) -> list[dict]:
    loaders = {
        "mmlu": _load_mmlu,
        "gsm8k": _load_gsm8k,
        "humaneval": _load_humaneval,
        "hotpotqa": _load_hotpotqa,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(loaders)}")
    return loaders[dataset](n=n, seed=seed)


def _load_mmlu(n=None, seed=42):
    from datasets import load_dataset
    import random

    subjects = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
    ]
    questions = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        for i, row in enumerate(ds):
            choices = row["choices"]
            choice_str = "\n".join(
                f"({chr(65+j)}) {c}" for j, c in enumerate(choices)
            )
            q_text = f"{row['question']}\n{choice_str}"
            questions.append(
                {
                    "question": q_text,
                    "correct_answer": chr(65 + row["answer"]),
                    "question_id": _make_question_id("mmlu", len(questions), q_text),
                    "task_type": "mcq",
                    "subject": subj,
                }
            )
    random.Random(seed).shuffle(questions)
    return questions[:n] if n else questions


def _load_gsm8k(n=None, seed=42):
    from datasets import load_dataset
    import random

    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    questions = []
    for i, row in enumerate(ds):
        answer = row["answer"].split("####")[-1].strip()
        questions.append(
            {
                "question": row["question"],
                "correct_answer": answer,
                "question_id": _make_question_id("gsm8k", i, row["question"]),
                "task_type": "numeric",
            }
        )
    random.Random(seed).shuffle(questions)
    return questions[:n] if n else questions


def _load_humaneval(n=None, seed=42):
    from datasets import load_dataset
    import random

    ds = load_dataset(
        "openai/openai_humaneval", split="test", trust_remote_code=True
    )
    questions = []
    for i, row in enumerate(ds):
        questions.append(
            {
                "question": row["prompt"],
                "correct_answer": row["canonical_solution"],
                "question_id": row["task_id"],
                "task_type": "code",
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    random.Random(seed).shuffle(questions)
    return questions[:n] if n else questions


def _load_hotpotqa(n=None, seed=42):
    from datasets import load_dataset
    import random

    ds = load_dataset(
        "hotpot_qa", "fullwiki", split="validation", trust_remote_code=True
    )
    questions = []
    for i, row in enumerate(ds):
        questions.append(
            {
                "question": row["question"],
                "correct_answer": row["answer"],
                "question_id": _make_question_id("hotpotqa", i, row["question"]),
                "task_type": "freeform",
            }
        )
    random.Random(seed).shuffle(questions)
    return questions[:n] if n else questions
