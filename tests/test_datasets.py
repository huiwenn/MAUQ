from topology_tax.datasets import load_questions, extract_answer, normalize_answer

def test_normalize_answer():
    assert normalize_answer("  The answer is B  ") == "b"
    assert normalize_answer("42.0") == "42.0"

def test_extract_answer_mcq():
    assert extract_answer("The answer is B", task_type="mcq") == "B"
    assert extract_answer("I think (C) is correct", task_type="mcq") == "C"
    assert extract_answer("A", task_type="mcq") == "A"

def test_extract_answer_numeric():
    assert extract_answer("The answer is 42", task_type="numeric") == "42"
    assert extract_answer("#### 15", task_type="numeric") == "15"
    assert extract_answer("Therefore, 3.14", task_type="numeric") == "3.14"

def test_extract_answer_freeform():
    assert extract_answer("Paris is the capital", task_type="freeform") == "Paris is the capital"

def test_load_questions_mmlu():
    questions = load_questions("mmlu", n=5, seed=42)
    assert len(questions) == 5
    assert all("question" in q for q in questions)
    assert all("correct_answer" in q for q in questions)
    assert all("question_id" in q for q in questions)
    assert all(q["task_type"] == "mcq" for q in questions)

def test_load_questions_gsm8k():
    questions = load_questions("gsm8k", n=5, seed=42)
    assert len(questions) == 5
    assert all(q["task_type"] == "numeric" for q in questions)
