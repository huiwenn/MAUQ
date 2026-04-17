import networkx as nx
from topology_tax.bedrock_client import BedrockAgent
from topology_tax.prompts import build_system_prompt, build_user_prompt
from topology_tax.datasets import extract_answer
from topology_tax.topologies import get_communication_order


class DebateOrchestrator:
    def __init__(self, agents):
        self.agents = agents

    def run_debate(self, topology, question, task_type="mcq", n_rounds=2):
        system_prompt = build_system_prompt(task_type)
        n_agents = len(self.agents)
        round_responses = []

        for round_num in range(n_rounds):
            current_round = {}
            order = get_communication_order(topology)

            for agent_idx in order:
                if agent_idx >= n_agents:
                    continue

                agent = self.agents[agent_idx]
                other_responses = {}

                if round_num > 0:
                    prev = round_responses[round_num - 1]
                    predecessors = set(topology.predecessors(agent_idx))
                    for pred_idx in predecessors:
                        if pred_idx in prev:
                            label = f"Agent {pred_idx} ({self.agents[pred_idx].agent_id})"
                            other_responses[label] = prev[pred_idx]

                if not other_responses:
                    other_responses = None

                user_prompt = build_user_prompt(
                    question=question,
                    other_responses=other_responses,
                    task_type=task_type,
                )
                response = agent.invoke(
                    user_prompt,
                    system_prompt=system_prompt,
                    other_responses=other_responses,
                )
                current_round[agent_idx] = response.text

            round_responses.append(current_round)

        final_round = round_responses[-1]
        final_answers = {
            idx: extract_answer(text, task_type) for idx, text in final_round.items()
        }
        all_responses = {idx: final_round.get(idx, "") for idx in range(n_agents)}

        return {
            "responses": all_responses,
            "final_answers": final_answers,
            "round_responses": round_responses,
            "n_rounds": n_rounds,
        }
