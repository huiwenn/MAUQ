import tempfile
from pathlib import Path
from topology_tax.storage import ResultStore

def test_store_and_retrieve_result():
    with tempfile.TemporaryDirectory() as td:
        store = ResultStore(Path(td) / "test.db")
        store.save_result(dataset="mmlu", question_id="q1", topology="complete", run_id=0, agent_id="claude-haiku", agent_position=0, prompt="What is 2+2?", response="4", answer_extracted="4", correct=True, model_id="anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0.0, latency_ms=150, input_tokens=50, output_tokens=10, metadata={"round": 1})
        rows = store.get_results(dataset="mmlu", question_id="q1")
        assert len(rows) == 1
        assert rows[0]["answer_extracted"] == "4"
        assert bool(rows[0]["correct"]) is True

def test_checkpoint_idempotent():
    with tempfile.TemporaryDirectory() as td:
        store = ResultStore(Path(td) / "test.db")
        for _ in range(3):
            store.save_result(dataset="gsm8k", question_id="q1", topology="star", run_id=0, agent_id="llama-70b", agent_position=1, prompt="p", response="r", answer_extracted="42", correct=True, model_id="meta.llama3-3-70b-instruct-v1:0", temperature=0.0, latency_ms=100, input_tokens=30, output_tokens=5, metadata={})
        rows = store.get_results(dataset="gsm8k", question_id="q1")
        assert len(rows) == 1

def test_is_completed():
    with tempfile.TemporaryDirectory() as td:
        store = ResultStore(Path(td) / "test.db")
        assert not store.is_completed("mmlu", "q1", "star", 0, "claude-haiku")
        store.save_result(dataset="mmlu", question_id="q1", topology="star", run_id=0, agent_id="claude-haiku", agent_position=0, prompt="p", response="r", answer_extracted="A", correct=True, model_id="test", temperature=0.0, latency_ms=100, input_tokens=10, output_tokens=5, metadata={})
        assert store.is_completed("mmlu", "q1", "star", 0, "claude-haiku")

def test_get_all_results_for_topology():
    with tempfile.TemporaryDirectory() as td:
        store = ResultStore(Path(td) / "test.db")
        for i in range(5):
            store.save_result(dataset="mmlu", question_id=f"q{i}", topology="ring", run_id=0, agent_id="a1", agent_position=0, prompt="p", response="r", answer_extracted=str(i), correct=(i % 2 == 0), model_id="test", temperature=0.0, latency_ms=100, input_tokens=10, output_tokens=5, metadata={})
        rows = store.get_results(dataset="mmlu", topology="ring")
        assert len(rows) == 5
