from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source") or r["metadata"].get("doc_id", "")
            context_parts.append(f"[{i}] (source: {source})\n{r['content']}")
        context = "\n\n".join(context_parts)
        prompt = (
            f"Answer the question based ONLY on the provided context.\n"
            f"If the answer is not in the context, say: \"I don't have enough information.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return self.llm_fn(prompt)
