from unittest.mock import MagicMock
from topology_tax.orchestrator import DebateOrchestrator
from topology_tax.topologies import build_topology
from topology_tax.bedrock_client import BedrockAgent, BedrockResponse

def _make_mock_agent(agent_id, answer="B"):
    agent = MagicMock(spec=BedrockAgent)
    agent.agent_id = agent_id
    agent.model_id = f"mock-{agent_id}"
    agent.invoke.return_value = BedrockResponse(text=f"I think the answer is {answer}", input_tokens=50, output_tokens=10, latency_ms=100, model_id=f"mock-{agent_id}")
    return agent

def test_orchestrator_independent():
    agents = [_make_mock_agent(f"agent_{i}") for i in range(3)]
    G = build_topology("independent", n_agents=3)
    orch = DebateOrchestrator(agents)
    result = orch.run_debate(G, question="What is 2+2?", task_type="mcq", n_rounds=1)
    assert len(result["responses"]) == 3
    for agent in agents:
        agent.invoke.assert_called_once()

def test_orchestrator_chain_respects_order():
    agents = [_make_mock_agent(f"agent_{i}", answer=str(i)) for i in range(3)]
    G = build_topology("chain", n_agents=3)
    orch = DebateOrchestrator(agents)
    result = orch.run_debate(G, question="What?", task_type="mcq", n_rounds=1)
    first_call = agents[0].invoke.call_args
    assert first_call[1].get("other_responses") is None or first_call[1]["other_responses"] == {}

def test_orchestrator_complete_sees_all():
    agents = [_make_mock_agent(f"agent_{i}") for i in range(3)]
    G = build_topology("complete", n_agents=3)
    orch = DebateOrchestrator(agents)
    result = orch.run_debate(G, question="What?", task_type="mcq", n_rounds=2)
    assert result["n_rounds"] == 2
    assert len(result["round_responses"]) == 2

def test_orchestrator_returns_final_answers():
    agents = [_make_mock_agent(f"agent_{i}", answer="C") for i in range(3)]
    G = build_topology("ring", n_agents=3)
    orch = DebateOrchestrator(agents)
    result = orch.run_debate(G, question="What?", task_type="mcq", n_rounds=1)
    assert "final_answers" in result
    assert len(result["final_answers"]) == 3
