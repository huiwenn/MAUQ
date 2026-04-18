import re
import hashlib
from typing import Optional


def normalize_answer(answer: str) -> str:
    text = answer.strip().lower()
    text = re.sub(
        r"^(?:the answer is|answer is|answer:)\s*", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def normalize_math_answer(answer: str) -> str:
    """Normalize a LaTeX math answer for comparison."""
    s = answer.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\ ", "").replace("\\,", "")
    s = s.replace("\\text{", "").replace("}", "")
    s = s.replace("$", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\boxed{", "").rstrip("}")
    s = re.sub(r"\s+", "", s)
    return s


def math_answers_equal(pred: str, gold: str) -> bool:
    """Compare two math answers with normalization."""
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if p == g:
        return True
    # Try numeric comparison
    try:
        pv = float(p.replace(",", ""))
        gv = float(g.replace(",", ""))
        return abs(pv - gv) < 1e-6
    except (ValueError, OverflowError):
        pass
    # Try sympy as last resort
    try:
        from sympy.parsing.latex import parse_latex
        pe = parse_latex(pred.strip())
        ge = parse_latex(gold.strip())
        return pe.equals(ge)
    except Exception:
        pass
    return False


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_answer(text: str, task_type: str = "mcq") -> str:
    if task_type == "mcq":
        match = re.search(
            r"(?:the answer is|answer:)\s*\**\(?([A-D])\)?\**", text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        match = re.search(r"\(([A-D])\)", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()
        return text.strip()
    elif task_type == "numeric":
        boxed = _extract_boxed(text)
        if boxed is not None:
            return boxed
        match = re.search(
            r"(?:the answer is|answer is|answer:)\s*(.+?)(?:\.|$)",
            text, re.IGNORECASE,
        )
        if match:
            ans = match.group(1).strip()
            if ans:
                return ans
        match = re.search(r"####\s*([\d,.\-]+)", text)
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
        "math500": _load_math500,
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
        ds = load_dataset("cais/mmlu", subj, split="test")
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
        "openai/openai_humaneval", split="test"    )
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
        "hotpot_qa", "fullwiki", split="validation"    )
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


def _load_math500(n=None, seed=42):
    from datasets import load_dataset
    import random

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = []
    for i, row in enumerate(ds):
        questions.append(
            {
                "question": row["problem"],
                "correct_answer": row["answer"],
                "question_id": _make_question_id("math500", i, row["problem"]),
                "task_type": "numeric",
                "level": row["level"],
                "subject": row["subject"],
                "solution": row["solution"],
            }
        )
    random.Random(seed).shuffle(questions)
    return questions[:n] if n else questions
