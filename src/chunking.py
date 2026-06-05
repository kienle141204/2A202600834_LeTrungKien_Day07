from __future__ import annotations

import math
import re


def preprocess_legal_markdown(text: str) -> str:
    """Remove PDF-to-Markdown artifacts from Vietnamese legal documents.

    Cleans up:
    - Pure separator lines (----, ****, ====)
    - Lines that are only punctuation/symbols with no legal content
    - Collapses 3+ consecutive blank lines into one blank line
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Remove pure separator lines (dashes, asterisks, equals, underscores)
        if re.fullmatch(r'[-*=_]{2,}', stripped):
            continue
        # Remove lines that are only symbols/punctuation artifacts (no CJK/Latin letters)
        if stripped and not re.search(r'[A-Za-zÀ-ỹ\d]', stripped):
            continue
        cleaned.append(line)

    # Collapse 3+ consecutive blank lines into a single blank line
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split on sentence boundaries, keeping punctuation with the sentence
        sentences = re.split(r'(?<=\.) |(?<=!) |(?<=\?) |(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            joined = " ".join(group)
            if joined:
                chunks.append(joined)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            # Last resort: return as-is (allow oversized chunk)
            return [current_text]

        sep = remaining_separators[0]

        # Empty string separator is not usable for splitting; skip it
        if sep == "":
            return self._split(current_text, remaining_separators[1:])

        parts = current_text.split(sep)

        # Separator not found — try the next one
        if len(parts) == 1:
            return self._split(current_text, remaining_separators[1:])

        result: list[str] = []
        for part in parts:
            if part:
                result.extend(self._split(part, remaining_separators))
        return result if result else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
