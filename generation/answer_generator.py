"""
generation/answer_generator.py
Builds the final prompt from HybridRetriever context and calls the LLM.
Every chunk is numbered [1], [2]... so the LLM can cite sources inline.
The answer always ends with a Sources section showing filename + page.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from config.settings import settings
from generation.llm_client import LLMClient
from retrieval.hybrid_retriever import RetrievedContext


SYSTEM_PROMPT = """You are an expert technical training assistant specialising in
robotics, electronics, embedded systems, and communication networks.

You answer questions using ONLY the context provided below — document excerpts
and knowledge graph relationships extracted from official training materials.

CITATION RULES (strictly follow these):
- Every factual claim must be cited using [N] where N matches the document chunk number.
- Example: "ROS uses a publish-subscribe model [1] where nodes communicate via topics [2]."
- If a fact comes from the knowledge graph relationships, write [KG] instead.
- Example: "Arduino controls the DC motor [KG] using PWM signals [3]."
- At the end of your answer, always add a ## Sources section listing each cited source.
- Format each source as: [N] filename.pdf — page X
- If the context is insufficient, say so honestly rather than guessing.
- Never cite [N] values that don't exist in the provided context.
"""


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    source_pages: List[dict] = field(default_factory=list)   # [{n, file, page}]
    graph_triples_used: List[str] = field(default_factory=list)
    context: RetrievedContext | None = None


class AnswerGenerator:

    def __init__(self):
        self.llm = LLMClient()

    def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: List[dict] | None = None,
    ) -> Answer:
        messages = self._build_messages(query, context, history or [])
        logger.debug(f"Sending {len(messages)} messages to LLM…")
        raw = self.llm.chat(messages)

        # Build structured source list
        source_pages = []
        for i, (chunk, _) in enumerate(context.chunks, 1):
            source_pages.append({
                "n":    i,
                "file": chunk.source_file,
                "page": chunk.page_number,
                "ref":  f"[{i}] {chunk.source_file} — page {chunk.page_number}",
            })

        return Answer(
            question=query,
            answer=raw,
            sources=[s["file"] for s in source_pages],
            source_pages=source_pages,
            graph_triples_used=[t.to_string() for t in context.triples],
            context=context,
        )

    def _build_messages(
        self,
        query: str,
        context: RetrievedContext,
        history: List[dict],
    ) -> List[dict]:
        context_block = self._build_context_block(context)
        user_content = (
            f"{context_block}\n\n"
            f"---\n\n"
            f"**Question:** {query}\n\n"
            f"Remember to cite every fact with [N] or [KG] and end with a ## Sources section."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_context_block(self, context: RetrievedContext) -> str:
        lines = ["## Retrieved document context\n"]
        for i, (chunk, score) in enumerate(context.chunks, 1):
            lines.append(
                f"[{i}] score={score:.3f} | {chunk.source_file} | page {chunk.page_number}\n"
                f"{chunk.text.strip()}\n"
            )
        if context.triples:
            lines.append("\n## Knowledge graph relationships [KG]\n")
            for t in context.triples:
                lines.append(f"  • {t.to_string()}")
        return "\n".join(lines)