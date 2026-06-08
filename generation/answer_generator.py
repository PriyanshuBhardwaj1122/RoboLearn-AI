"""
generation/answer_generator.py
Builds the final prompt from HybridRetriever context and calls the LLM.
Supports streaming, citation extraction, and conversation history.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Generator

from loguru import logger

from config.settings import settings
from generation.llm_client import LLMClient
from retrieval.hybrid_retriever import RetrievedContext


SYSTEM_PROMPT = """You are an expert technical training assistant specialising in
robotics, electronics, embedded systems, and communication networks.

You answer questions using ONLY the context provided below — document excerpts
and knowledge graph relationships extracted from official training materials.

Guidelines:
- Be precise and technical. Prefer terminology from the source materials.
- If the answer spans multiple components (e.g. ROS → Arduino → Motor), trace
  the chain step by step.
- Cite sources using [N] notation (matching the document context numbering).
- If the context is insufficient, say so honestly rather than guessing.
- For relationships from the knowledge graph, explain the nature of each link.
"""


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    graph_triples_used: List[str] = field(default_factory=list)
    context: RetrievedContext | None = None


class AnswerGenerator:
    """
    Usage:
        gen = AnswerGenerator()
        answer = gen.generate(query, context)
        print(answer.answer)
    """

    def __init__(self):
        self.llm = LLMClient()

    # ── Public ──────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: List[dict] | None = None,
    ) -> Answer:
        messages = self._build_messages(query, context, history or [])
        logger.debug(f"Sending {len(messages)} messages to LLM…")
        raw = self.llm.chat(messages)
        return Answer(
            question=query,
            answer=raw,
            sources=[c.source_file for c, _ in context.chunks],
            graph_triples_used=[t.to_string() for t in context.triples],
            context=context,
        )

    # ── Prompt assembly ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        context: RetrievedContext,
        history: List[dict],
    ) -> List[dict]:
        context_block = context.to_prompt_block()
        user_content = (
            f"{context_block}\n\n"
            f"---\n\n"
            f"**Question:** {query}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages
