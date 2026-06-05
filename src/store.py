from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    When a chunker is provided, add_documents automatically splits each document
    into chunks before embedding, storing each chunk as a separate record.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        chunker: Any | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._chunker = chunker
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": {**doc.metadata, "doc_id": doc.id},
        }

    def _make_chunk_records(self, doc: Document) -> list[dict[str, Any]]:
        """Split doc into chunks and build a record for each chunk."""
        chunks = self._chunker.chunk(doc.content)
        records = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.id}__chunk_{i}"
            chunk_meta = {**doc.metadata, "doc_id": doc.id, "chunk_index": i, "total_chunks": len(chunks)}
            records.append({
                "id": chunk_id,
                "content": chunk_text,
                "embedding": self._embedding_fn(chunk_text),
                "metadata": chunk_meta,
            })
        return records

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            r = dict(record)
            r["score"] = score
            scored.append(r)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document and store it.

        When a chunker was provided at construction time, each document is split
        into chunks first; each chunk is stored as a separate record with metadata
        keys doc_id, chunk_index, and total_chunks.
        """
        records_to_add: list[dict[str, Any]] = []
        for doc in docs:
            if self._chunker is not None:
                records_to_add.extend(self._make_chunk_records(doc))
            else:
                records_to_add.append(self._make_record(doc))

        if self._use_chroma:
            for r in records_to_add:
                self._collection.add(
                    ids=[r["id"]],
                    documents=[r["content"]],
                    embeddings=[r["embedding"]],
                    metadatas=[r["metadata"]],
                )
        else:
            self._store.extend(records_to_add)
            self._next_index += len(records_to_add)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents/chunks to query."""
        if self._use_chroma:
            count = self._collection.count()
            if count == 0:
                return []
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
            )
            output: list[dict[str, Any]] = []
            if results and results["documents"]:
                for content, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    output.append({"content": content, "metadata": meta, "score": 1.0 - dist})
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored records (chunks or documents)."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if metadata_filter is None:
            return self.search(query, top_k)
        if self._use_chroma:
            count = self._collection.count()
            if count == 0:
                return []
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=metadata_filter,
            )
            output: list[dict] = []
            if results and results["documents"]:
                for content, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    output.append({"content": content, "metadata": meta, "score": 1.0 - dist})
            return output
        else:
            filtered = [
                r for r in self._store
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all records belonging to a document (works for both whole-doc
        and chunk-level storage).

        Returns True if any records were removed, False otherwise.
        """
        if self._use_chroma:
            try:
                existing = self._collection.get(ids=[doc_id])
                if not existing["ids"]:
                    return False
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False
        else:
            initial_size = len(self._store)
            # Filter by metadata doc_id so chunk-level records are also removed
            self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_size
