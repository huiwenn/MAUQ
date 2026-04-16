"""
Integration smoke test: runs 1 question through the full pipeline.
Requires AWS Bedrock credentials. Skip if not available.
"""
import pytest
import tempfile
from pathlib import Path

# Check if all required modules exist
try:
    from topology_tax.bedrock_client import BedrockAgent, BedrockResponse
    from topology_tax.orchestrator import DebateOrchestrator
    from topology_tax.topologies import build_topology
    from topology_tax.datasets import extract_answer
    from topology_tax.storage import ResultStore
    from topology_tax.metrics import decompose_topology_tax
    HAS_ALL_MODULES = True
except ImportError:
    HAS_ALL_MODULES = False

try:
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    # Quick check: list available models (lightweight call)
    client.meta.service_model
    HAS_BEDROCK = True
except Exception:
    HAS_BEDROCK = False


@pytest.mark.skipif(not HAS_ALL_MODULES, reason="Not all topology_tax modules are built yet")
@pytest.mark.skipif(not HAS_BEDROCK, reason="AWS Bedrock credentials not available")
class TestEndToEndSmoke:
    """Smoke test: 1 question, 1 model, 1 topology, through the full pipeline."""

    def test_single_question_chain_topology(self):
        """Run a single MMLU-style question through a 3-agent chain debate."""
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"

        agents = [
            BedrockAgent(agent_id=f"agent_{i}", model_id=model_id)
            for i in range(3)
        ]

        G = build_topology("chain", n_agents=3)
        orch = DebateOrchestrator(agents)

        question = (
            "What is the capital of France?\n"
            "(A) London\n(B) Berlin\n(C) Paris\n(D) Madrid"
        )

        result = orch.run_debate(
            G, question=question, task_type="mcq", n_rounds=1
        )

        assert len(result["responses"]) == 3
        assert len(result["final_answers"]) == 3
        # At least one agent should answer C (Paris)
        answers = list(result["final_answers"].values())
        assert "C" in answers, f"Expected at least one 'C', got {answers}"

    def test_storage_roundtrip_with_real_data(self):
        """Verify storage works with realistic data from a debate."""
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "smoke.db")
            store.save_result(
                dataset="mmlu", question_id="smoke_q1",
                topology="chain", run_id=0,
                agent_id="claude-haiku", agent_position=0,
                prompt="What is 2+2?", response="The answer is B",
                answer_extracted="B", correct=False,
                model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
                temperature=0.0, latency_ms=250,
                input_tokens=100, output_tokens=20,
                metadata={"round": 1, "topology": "chain"},
            )
            rows = store.get_results(dataset="mmlu", question_id="smoke_q1")
            assert len(rows) == 1
            assert rows[0]["answer_extracted"] == "B"


@pytest.mark.skipif(not HAS_ALL_MODULES, reason="Not all topology_tax modules are built yet")
class TestLocalIntegration:
    """Integration tests that don't need Bedrock (use mocks)."""

    def test_full_pipeline_with_mocks(self):
        """End-to-end: topology -> orchestrator -> metrics -> storage."""
        from unittest.mock import MagicMock
        import numpy as np

        # Build topology
        G = build_topology("star", n_agents=3)

        # Create mock agents
        agents = []
        mock_answers = ["C", "C", "A"]
        for i, ans in enumerate(mock_answers):
            agent = MagicMock(spec=BedrockAgent)
            agent.agent_id = f"agent_{i}"
            agent.model_id = "mock"
            agent.invoke.return_value = BedrockResponse(
                text=f"The answer is {ans}",
                input_tokens=50, output_tokens=10,
                latency_ms=100, model_id="mock",
            )
            agents.append(agent)

        # Run debate
        orch = DebateOrchestrator(agents)
        result = orch.run_debate(G, question="Test?", task_type="mcq", n_rounds=1)

        assert len(result["final_answers"]) == 3

        # Store results
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "test.db")
            for idx, ans in result["final_answers"].items():
                store.save_result(
                    dataset="test", question_id="q0",
                    topology="star", run_id=0,
                    agent_id=f"agent_{idx}", agent_position=idx,
                    prompt="Test?", response=result["responses"][idx],
                    answer_extracted=ans, correct=(ans == "C"),
                    model_id="mock", temperature=0.0,
                    latency_ms=100, input_tokens=50, output_tokens=10,
                    metadata={},
                )
            rows = store.get_results(dataset="test", topology="star")
            assert len(rows) == 3
