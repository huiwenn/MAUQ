"""
Integration smoke tests for the Topology Tax pipeline.

Three levels:
  1. End-to-end (requires real AWS Bedrock credentials + all modules)
  2. Local integration (mocks Bedrock, requires all modules)
  3. Simulator-to-analysis pipeline (no Bedrock needed, uses only core modules)
"""
import pytest
import tempfile
from pathlib import Path

import numpy as np

# --- Core modules (always available once the base project is set up) ---
try:
    from topology_tax.topologies import build_topology
    from topology_tax.simulator import NoisyChannelSimulator
    from topology_tax.metrics import decompose_topology_tax
    from topology_tax.storage import ResultStore
    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False

# --- Modules still being built by other agents ---
try:
    from topology_tax.bedrock_client import BedrockAgent, BedrockResponse
    from topology_tax.orchestrator import DebateOrchestrator
    from topology_tax.datasets import extract_answer
    HAS_ALL_MODULES = True
except ImportError:
    HAS_ALL_MODULES = False

try:
    from topology_tax.analysis import analyze_results
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

# --- Bedrock credential check ---
try:
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    client.meta.service_model
    HAS_BEDROCK = True
except Exception:
    HAS_BEDROCK = False


# ── Test 1: End-to-end smoke (real Bedrock) ────────────────────────────

@pytest.mark.skipif(not HAS_ALL_MODULES, reason="Not all topology_tax modules are built yet")
@pytest.mark.skipif(not HAS_BEDROCK, reason="AWS Bedrock credentials not available")
class TestEndToEndSmoke:
    """Smoke test: 1 question, 1 agent, independent topology, through the full pipeline."""

    def test_single_question_independent_topology(self):
        """Run a single MMLU-style question through 1 agent on 'independent' topology."""
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"

        agents = [BedrockAgent(agent_id="agent_0", model_id=model_id)]

        G = build_topology("independent", n_agents=1)
        orch = DebateOrchestrator(agents)

        question = (
            "What is the capital of France?\n"
            "(A) London\n(B) Paris\n(C) Berlin\n(D) Tokyo"
        )

        result = orch.run_debate(
            G, question=question, task_type="mcq", n_rounds=1
        )

        # We should get a response from the single agent
        assert len(result["responses"]) >= 1
        assert len(result["final_answers"]) >= 1
        # The agent should answer B (Paris)
        answers = list(result["final_answers"].values())
        assert "B" in answers, f"Expected 'B' (Paris), got {answers}"

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


# ── Test 2: Local integration (mocked Bedrock) ────────────────────────

@pytest.mark.skipif(not HAS_ALL_MODULES, reason="Not all topology_tax modules are built yet")
class TestLocalIntegration:
    """Integration tests that mock Bedrock but exercise the rest of the pipeline."""

    def test_full_pipeline_with_mocks(self):
        """End-to-end with mock Bedrock: 1 agent, independent topology -> orchestrator -> storage."""
        from unittest.mock import MagicMock

        # Build a 1-agent independent topology
        G = build_topology("independent", n_agents=1)

        # Create a mock agent that answers "B"
        agent = MagicMock(spec=BedrockAgent)
        agent.agent_id = "agent_0"
        agent.model_id = "mock"
        agent.invoke.return_value = BedrockResponse(
            text="The answer is (B) Paris.",
            input_tokens=50, output_tokens=10,
            latency_ms=100, model_id="mock",
        )

        # Run debate
        orch = DebateOrchestrator([agent])
        result = orch.run_debate(
            G, question="What is the capital of France?",
            task_type="mcq", n_rounds=1,
        )

        assert len(result["final_answers"]) >= 1

        # Store results
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "test.db")
            for idx, ans in result["final_answers"].items():
                store.save_result(
                    dataset="test", question_id="q0",
                    topology="independent", run_id=0,
                    agent_id=f"agent_{idx}", agent_position=idx,
                    prompt="What is the capital of France?",
                    response=result["responses"][idx],
                    answer_extracted=ans, correct=(ans == "B"),
                    model_id="mock", temperature=0.0,
                    latency_ms=100, input_tokens=50, output_tokens=10,
                    metadata={},
                )
            rows = store.get_results(dataset="test", topology="independent")
            assert len(rows) >= 1

    def test_multi_agent_star_topology_with_mocks(self):
        """3-agent star topology with mocked Bedrock, verifies full pipeline."""
        from unittest.mock import MagicMock

        G = build_topology("star", n_agents=3)

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

        orch = DebateOrchestrator(agents)
        result = orch.run_debate(G, question="Test?", task_type="mcq", n_rounds=1)

        assert len(result["final_answers"]) == 3

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


# ── Test 3: Simulator-to-analysis pipeline (no Bedrock needed) ────────

@pytest.mark.skipif(not HAS_CORE_MODULES, reason="Core topology_tax modules not available")
class TestSimulatorToAnalysisPipeline:
    """Run the synthetic simulator and feed results through analysis.

    This test uses only core modules (topologies, simulator, metrics, storage)
    and does NOT require Bedrock credentials or the bedrock_client module.
    """

    def test_simulator_decomposition_and_storage(self):
        """Run simulator on two topologies, verify decomposition, store results."""
        n_agents = 5
        accuracies = [0.7] * n_agents
        anchoring = np.full((n_agents, n_agents), 0.3)
        np.fill_diagonal(anchoring, 0.0)
        sim = NoisyChannelSimulator(n_agents, accuracies, anchoring, seed=42)

        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "sim_pipeline.db")

            for topo_name in ["independent", "star", "chain"]:
                G = build_topology(topo_name, n_agents=n_agents)
                result = sim.run(G, n_questions=50, n_runs=10)

                # Verify structure of simulation output
                assert "majority_accuracy" in result
                assert "independent_accuracy" in result
                assert "decomposition" in result

                dec = result["decomposition"]
                assert "benefit" in dec
                assert "amplification" in dec
                assert "absorption" in dec

                # Accuracies should be in valid range
                assert 0.0 <= result["majority_accuracy"] <= 1.0
                assert 0.0 <= result["independent_accuracy"] <= 1.0

                # Store synthetic results
                store.save_result(
                    dataset="synthetic", question_id="sim_batch",
                    topology=topo_name, run_id=0,
                    agent_id="simulator", agent_position=0,
                    prompt="synthetic_sim", response="n/a",
                    answer_extracted="n/a",
                    correct=(result["majority_accuracy"] > 0.5),
                    model_id="noisy_channel", temperature=0.0,
                    latency_ms=0, input_tokens=0, output_tokens=0,
                    metadata={
                        "majority_accuracy": result["majority_accuracy"],
                        "independent_accuracy": result["independent_accuracy"],
                        **dec,
                    },
                )

            # Verify all three topologies were stored
            rows = store.get_results(dataset="synthetic")
            assert len(rows) == 3
            topologies_stored = {r["topology"] for r in rows}
            assert topologies_stored == {"independent", "star", "chain"}

    def test_independent_topology_is_baseline(self):
        """Independent topology should have zero benefit (it IS the baseline)."""
        n_agents = 5
        accuracies = [0.65] * n_agents
        anchoring = np.full((n_agents, n_agents), 0.3)
        np.fill_diagonal(anchoring, 0.0)
        sim = NoisyChannelSimulator(n_agents, accuracies, anchoring, seed=99)

        G_indep = build_topology("independent", n_agents=n_agents)
        result = sim.run(G_indep, n_questions=100, n_runs=10)

        # Independent topology: majority_accuracy == independent_accuracy
        # so benefit should be ~0
        assert abs(result["decomposition"]["benefit"]) < 0.05, (
            f"Independent topology should have ~0 benefit, got {result['decomposition']['benefit']}"
        )

    @pytest.mark.skipif(not HAS_ANALYSIS, reason="analysis module not yet available")
    def test_analysis_on_simulator_output(self):
        """Feed simulator results through the analysis module (if available)."""
        n_agents = 5
        accuracies = [0.7] * n_agents
        anchoring = np.full((n_agents, n_agents), 0.3)
        np.fill_diagonal(anchoring, 0.0)
        sim = NoisyChannelSimulator(n_agents, accuracies, anchoring, seed=42)

        results_for_analysis = []
        for topo_name in ["independent", "star", "complete", "ring"]:
            G = build_topology(topo_name, n_agents=n_agents)
            result = sim.run(G, n_questions=100, n_runs=10)
            results_for_analysis.append({
                "topology": topo_name,
                **result,
            })

        # Pass to analysis module
        analysis_output = analyze_results(results_for_analysis)
        assert analysis_output is not None
