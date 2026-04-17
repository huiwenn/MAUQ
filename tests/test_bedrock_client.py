from unittest.mock import MagicMock
from topology_tax.bedrock_client import BedrockAgent, BedrockResponse

def _mock_converse_response(text, input_tokens=10, output_tokens=5):
    return {"output": {"message": {"content": [{"text": text}]}}, "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens}, "metrics": {"latencyMs": 150}}

def test_bedrock_agent_invoke():
    mock_client = MagicMock()
    mock_client.converse.return_value = _mock_converse_response("The answer is B")
    agent = BedrockAgent(agent_id="claude-haiku", model_id="anthropic.claude-3-5-haiku-20241022-v1:0", client=mock_client)
    resp = agent.invoke("What is 2+2?", system_prompt="Answer concisely.")
    assert resp.text == "The answer is B"
    assert resp.input_tokens == 10
    assert resp.latency_ms == 150
    mock_client.converse.assert_called_once()

def test_bedrock_agent_invoke_with_context():
    mock_client = MagicMock()
    mock_client.converse.return_value = _mock_converse_response("I agree, B")
    agent = BedrockAgent(agent_id="llama-70b", model_id="meta.llama3-3-70b-instruct-v1:0", client=mock_client)
    resp = agent.invoke("What is 2+2?", system_prompt="You are a helpful assistant.", other_responses={"agent_0": "I think B", "agent_2": "I think A"})
    assert resp.text == "I agree, B"
    call_args = mock_client.converse.call_args
    messages = call_args[1].get("messages", call_args[0][0] if call_args[0] else [])
    assert any("agent_0" in str(m) for m in messages)

def test_bedrock_response_fields():
    resp = BedrockResponse(text="Answer is A", input_tokens=100, output_tokens=20, latency_ms=200, model_id="test-model")
    assert resp.text == "Answer is A"
    assert resp.input_tokens == 100
