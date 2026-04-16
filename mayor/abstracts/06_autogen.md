# AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

**Authors:** Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, Chi Wang

**Venue:** arXiv:2308.08155, August 16, 2023 (revised October 3, 2023; 2000+ citations)

**Category:** Multi-Agent Systems / Frameworks

## Abstract

AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.

## Key Contribution

The foundational multi-agent conversation framework from Microsoft Research that established the paradigm of customizable, conversable agents with flexible interaction patterns. One of the most cited papers in the multi-agent LLM space.

## Relevance to Gas Town / MAUQ

AutoGen is the primary comparison point for Gas Town's architecture. Key differences: AutoGen uses synchronous conversation turns while Gas Town uses asynchronous mail; AutoGen's interaction patterns are conversation-based while Gas Town uses hooks (propulsion principle). Understanding AutoGen's design choices and limitations (synchronous bottlenecks, rigid turn-taking) clarifies what Gas Town's asynchronous architecture solves.
