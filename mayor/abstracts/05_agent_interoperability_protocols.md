# A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, and ANP

**Authors:** Abul Ehtesham, Aditi Singh, Gaurav Kumar Gupta, Saket Kumar

**Venue:** arXiv:2505.02279, May 4, 2025 (revised May 23, 2025)

**Category:** Multi-Agent Systems / Communication Protocols

## Abstract

The survey examines four emerging communication frameworks for LLM-powered autonomous agents. MCP provides a JSON-RPC client-server interface for secure tool invocation and typed data exchange. ACP defines a general-purpose communication protocol over RESTful HTTP, supporting MIME-typed multipart messages. A2A enables peer-to-peer task delegation using capability-based Agent Cards. ANP supports open network agent discovery and secure collaboration using W3C decentralized identifiers. The paper proposes a phased adoption roadmap: beginning with MCP for tool access, followed by ACP for structured multimodal messaging, A2A for collaborative task execution, and extending to ANP for decentralized agent marketplaces, providing a comprehensive foundation for designing secure, interoperable, and scalable ecosystems of LLM-powered agents.

## Key Contribution

First comprehensive comparative survey of the four major agent communication protocols (MCP, ACP, A2A, ANP), with a practical phased adoption roadmap from tool access to decentralized marketplaces.

## Relevance to Gas Town / MAUQ

Directly relevant to Gas Town's mail-based communication design. Gas Town's asynchronous mail system is a custom protocol; this survey provides context on how it relates to industry-standard approaches. The phased roadmap (tool access -> messaging -> task delegation -> decentralized discovery) maps to Gas Town's evolution path. A2A's peer-to-peer task delegation via "Agent Cards" is conceptually similar to Gas Town's hook-based work assignment.
