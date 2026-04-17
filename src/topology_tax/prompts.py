SYSTEM_PROMPTS = {
    "mcq": (
        "You are an expert answering multiple-choice questions. Reason step by step, "
        "then state your final answer as a single letter (A, B, C, or D) on its own "
        "line prefixed with 'The answer is'."
    ),
    "numeric": (
        "You are an expert at mathematical reasoning. Solve the problem step by step. "
        "State your final numeric answer on its own line prefixed with '#### '."
    ),
    "freeform": (
        "You are a knowledgeable assistant answering factual questions. "
        "Reason step by step, then state your final answer concisely."
    ),
    "code": (
        "You are an expert Python programmer. Complete the given function. "
        "Return only the function body, no explanation."
    ),
}


def build_system_prompt(task_type="mcq"):
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["freeform"])


def format_other_responses(responses):
    return "\n".join(
        f"[{agent_label}]: {response}"
        for agent_label, response in responses.items()
    )


def build_user_prompt(question, other_responses=None, task_type="mcq"):
    parts = [question]
    if other_responses:
        parts.append(
            "\nThe following agents have already responded:\n"
            + format_other_responses(other_responses)
            + "\n\nConsidering their responses, provide your own answer."
        )
    return "\n".join(parts)
