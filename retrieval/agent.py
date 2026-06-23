"""
retrieval/agent.py
ReAct (Reason + Act) agentic retrieval loop.

The agent:
  1. Receives a user query
  2. Thinks about what to search (Thought)
  3. Calls a retrieval tool (Action)
  4. Reads the result (Observation)
  5. Decides if context is sufficient (Reflect)
  6. If not — searches again with a targeted follow-up query
  7. When sufficient — generates the final answer

Tools available:
  vector_search(query)   — FAISS semantic search
  graph_search(query)    — Neo4j Cypher traversal
  finish(answer)         — stop and return final answer

Max iterations: 4 (prevents infinite loops)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from embeddings.vector_store import VectorStore
from graph.kg_retriever import KGRetriever
from generation.llm_client import LLMClient
from config.settings import settings


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    thought:     str = ""
    action:      str = ""
    action_input:str = ""
    observation: str = ""


@dataclass
class AgentResult:
    query:        str
    answer:       str
    steps:        List[AgentStep] = field(default_factory=list)
    source_pages: List[dict]      = field(default_factory=list)
    triples_used: List[str]       = field(default_factory=list)
    iterations:   int = 0


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert technical assistant with access to two retrieval tools.
You answer questions about robotics, electronics, Arduino, ROS, and embedded systems.

You have access to these tools:
  vector_search(query) — finds relevant document chunks by semantic similarity
  graph_search(query)  — finds entity relationships in the knowledge graph

REACT FORMAT — you must follow this exact format:
Thought: [reason about what information you need next]
Action: vector_search OR graph_search
Action Input: [specific search query]

After receiving an Observation, continue with another Thought/Action OR:
Thought: I now have sufficient information to answer the question.
Action: finish
Action Input: [your complete answer with [N] citations]

STRATEGY — for multi-hop chain questions:
  Step 1: graph_search for the entity relationships first (fast, finds connections)
  Step 2: vector_search for the hardware side (Arduino, ATmega328P, pins)
  Step 3: vector_search for the software side (ROS, rosserial, topics)
  Step 4: vector_search for the output side (motor, PWM, driver)
  Then: finish with the complete chain

RULES:
- Always start complex chain questions with graph_search to find connections
- Use vector_search for specific technical details and page citations
- Maximum 4 iterations — be efficient, search different topics each time
- Never search the same topic twice — each search must cover new ground
- Cite every fact with [N] where N is the chunk number from observations
- Use [KG] for facts from graph_search results
- If context is insufficient after 4 iterations, state what is missing honestly
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReactAgent:

    MAX_ITERATIONS = 4

    def __init__(
        self,
        vector_store: VectorStore,
        kg_retriever:  KGRetriever,
    ):
        self.vs  = vector_store
        self.kg  = kg_retriever
        self.llm = LLMClient()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, query: str) -> AgentResult:
        """Run the ReAct loop for a given query."""
        logger.info(f"Agent starting: {query!r}")

        result = AgentResult(query=query, answer="")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {query}\n\nBegin."},
        ]

        all_chunks  = []
        all_triples = []

        for iteration in range(self.MAX_ITERATIONS):
            result.iterations = iteration + 1
            logger.debug(f"Agent iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            # ── LLM thinks and acts ───────────────────────────────────────
            response = self.llm.chat(messages)
            step     = self._parse_response(response)
            result.steps.append(step)

            logger.debug(f"  Thought: {step.thought[:80]}")
            logger.debug(f"  Action:  {step.action}({step.action_input})")

            # ── Execute tool ─────────────────────────────────────────────
            if step.action.lower() == "finish":
                result.answer = step.action_input
                break

            elif step.action.lower() == "vector_search":
                chunks, observation = self._vector_search(
                    step.action_input, len(all_chunks)
                )
                all_chunks.extend(chunks)
                step.observation = observation

            elif step.action.lower() == "graph_search":
                triples, observation = self._graph_search(step.action_input)
                all_triples.extend(triples)
                step.observation = observation

            else:
                step.observation = f"Unknown tool '{step.action}'. Use vector_search or graph_search."

            logger.debug(f"  Observation: {step.observation[:120]}")

            # ── Feed observation back to LLM ─────────────────────────────
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {step.observation}\n\nContinue."
            })

        # If we hit max iterations without finishing
        if not result.answer:
            result.answer = self._force_finish(messages, all_chunks, all_triples)

        # Build source pages
        result.source_pages = [
            {
                "n":    i + 1,
                "file": chunk.source_file,
                "page": chunk.page_number,
                "ref":  f"[{i+1}] {chunk.source_file} — page {chunk.page_number}",
            }
            for i, chunk in enumerate(all_chunks)
        ]
        result.triples_used = [
            f"{t.source} --[{t.relation}]--> {t.target}"
            for t in all_triples
        ]

        logger.info(
            f"Agent done: {result.iterations} iterations, "
            f"{len(all_chunks)} chunks, {len(all_triples)} triples"
        )
        return result

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, offset: int):
        """Run FAISS search and format as observation."""
        try:
            results = self.vs.search(query, top_k=3)
            if not results:
                return [], "No relevant chunks found."

            chunks = [c for c, _ in results]
            lines  = []
            for i, (chunk, score) in enumerate(results, offset + 1):
                lines.append(
                    f"[{i}] {chunk.source_file} page {chunk.page_number} "
                    f"(score={score:.3f}):\n{chunk.text[:400].strip()}"
                )
            return chunks, "\n\n".join(lines)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return [], f"Search failed: {e}"

    def _graph_search(self, query: str):
        """Run Neo4j search and format as observation."""
        try:
            triples = self.kg.search(query, top_k=8)
            if not triples:
                return [], "No graph relationships found."
            lines = [f"[KG] {t.to_string()}" for t in triples]
            return triples, "\n".join(lines)
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return [], f"Graph search failed: {e}"

    # ── Parse LLM response ────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> AgentStep:
        """Parse Thought/Action/Action Input from LLM response."""
        step = AgentStep()

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
        action_match  = re.search(r"Action:\s*(\w+)", text)
        input_match   = re.search(r"Action Input:\s*(.+?)(?=Thought:|Observation:|$)", text, re.DOTALL)

        if thought_match:
            step.thought = thought_match.group(1).strip()
        if action_match:
            step.action = action_match.group(1).strip()
        if input_match:
            step.action_input = input_match.group(1).strip()

        # If LLM didn't follow format — treat entire response as finish
        if not step.action:
            step.action       = "finish"
            step.action_input = text.strip()

        return step

    def _force_finish(self, messages, chunks, triples) -> str:
        """Force a final answer after max iterations."""
        context = "\n\n".join(
            f"[{i+1}] {c.source_file} p{c.page_number}: {c.text[:300]}"
            for i, c in enumerate(chunks)
        )
        kg_context = "\n".join(
            f"[KG] {t.source} --[{t.relation}]--> {t.target}"
            for t in triples
        )
        messages.append({
            "role": "user",
            "content": (
                f"You have reached the maximum number of iterations. "
                f"Based on everything retrieved so far, provide your best answer now.\n\n"
                f"Retrieved chunks:\n{context}\n\n"
                f"Graph relationships:\n{kg_context}"
            )
        })
        return self.llm.chat(messages)