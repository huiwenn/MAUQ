from topology_tax.prompts import build_system_prompt, build_user_prompt, format_other_responses

def test_build_system_prompt_mcq():
    prompt = build_system_prompt(task_type="mcq")
    assert "multiple choice" in prompt.lower() or "answer" in prompt.lower()

def test_build_user_prompt_no_context():
    prompt = build_user_prompt(question="What is the capital of France?", other_responses=None, task_type="mcq")
    assert "What is the capital of France?" in prompt
    assert "Other agents" not in prompt

def test_build_user_prompt_with_context():
    prompt = build_user_prompt(question="What is 2+2?", other_responses={"Agent 1 (claude-haiku)": "I think it's 4", "Agent 3 (llama-70b)": "The answer is 4"}, task_type="numeric")
    assert "What is 2+2?" in prompt
    assert "Agent 1" in prompt

def test_format_other_responses():
    formatted = format_other_responses({"Agent 0": "Answer A", "Agent 2": "Answer B"})
    assert "Agent 0" in formatted
    assert "Answer A" in formatted
