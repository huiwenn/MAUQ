import time
from dataclasses import dataclass
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


@dataclass
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str


def _is_throttle(exc):
    name = type(exc).__name__
    return "Throttling" in name or "ThrottlingException" in name or "429" in str(exc)


class BedrockAgent:
    def __init__(self, agent_id, model_id, client=None, temperature=0.0, max_tokens=1024):
        self.agent_id = agent_id
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            import os
            import boto3
            region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"
            self.client = boto3.client("bedrock-runtime", region_name=region)
        else:
            self.client = client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception(_is_throttle),
    )
    def invoke(self, question, system_prompt="Answer the question concisely.", other_responses=None):
        user_content = question
        if other_responses:
            context_parts = [f"[{aid}]: {resp}" for aid, resp in other_responses.items()]
            user_content = (
                f"{question}\n\nOther agents have responded:\n"
                + "\n".join(context_parts)
                + "\n\nNow provide your answer."
            )

        messages = [{"role": "user", "content": [{"text": user_content}]}]

        inference_config = {"maxTokens": self.max_tokens}
        if self.temperature > 0:
            inference_config["temperature"] = self.temperature

        start = time.monotonic()
        response = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig=inference_config,
        )
        latency = int((time.monotonic() - start) * 1000)

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        metrics = response.get("metrics", {})

        return BedrockResponse(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            latency_ms=metrics.get("latencyMs", latency),
            model_id=self.model_id,
        )
