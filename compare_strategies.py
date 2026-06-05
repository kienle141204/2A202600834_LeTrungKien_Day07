"""
compare_strategies.py
---------------------
Các hàm so sánh chunking strategy cho bộ tài liệu pháp luật Việt Nam.

Sử dụng:
    python compare_strategies.py                        # chạy benchmark đầy đủ
    python compare_strategies.py --dir data/new_data    # chỉ định thư mục data
    python compare_strategies.py --chunk-size 800       # thay đổi chunk_size
    python compare_strategies.py --no-preprocess        # bỏ qua tiền xử lý
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(__file__))

from src.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    preprocess_legal_markdown,
)
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


# ---------------------------------------------------------------------------
# 1. Thống kê chunk
# ---------------------------------------------------------------------------

def chunk_stats(chunks: list[str]) -> dict:
    """Tính các chỉ số thống kê cho một danh sách chunk.

    Returns:
        dict với các key: count, avg_length, min_length, max_length,
        std_length, short_chunks (< 50 chars), very_short (< 20 chars).
    """
    if not chunks:
        return {
            "count": 0, "avg_length": 0.0, "min_length": 0,
            "max_length": 0, "std_length": 0.0,
            "short_chunks": 0, "very_short": 0,
        }
    lengths = [len(c) for c in chunks]
    n = len(lengths)
    avg = sum(lengths) / n
    std = math.sqrt(sum((l - avg) ** 2 for l in lengths) / n)
    return {
        "count": n,
        "avg_length": round(avg, 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "std_length": round(std, 1),
        "short_chunks": sum(1 for l in lengths if l < 50),
        "very_short": sum(1 for l in lengths if l < 20),
    }


# ---------------------------------------------------------------------------
# 2. So sánh strategy trên một văn bản
# ---------------------------------------------------------------------------

def compare_on_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    max_sentences: int = 3,
    preprocess: bool = True,
) -> dict[str, dict]:
    """So sánh 3 strategy (fixed_size, by_sentences, recursive) trên một văn bản.

    Args:
        text: Nội dung văn bản cần so sánh.
        chunk_size: Kích thước chunk tối đa (chars).
        overlap: Số ký tự overlap cho FixedSizeChunker.
        max_sentences: Số câu tối đa mỗi chunk cho SentenceChunker.
        preprocess: Nếu True, làm sạch artifact trước khi chunking.

    Returns:
        {strategy_name: {stats, sample}} — stats từ chunk_stats(),
        sample là 2 chunk đầu tiên.
    """
    if preprocess:
        text = preprocess_legal_markdown(text)

    chunkers: dict[str, object] = {
        "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=max_sentences),
        "recursive": RecursiveChunker(chunk_size=chunk_size),
    }
    results: dict[str, dict] = {}
    for name, chunker in chunkers.items():
        chunks = chunker.chunk(text)
        stats = chunk_stats(chunks)
        stats["sample"] = [c[:120].replace("\n", " ") for c in chunks[:2]]
        results[name] = stats
    return results


# ---------------------------------------------------------------------------
# 3. So sánh strategy trên toàn bộ corpus (nhiều file)
# ---------------------------------------------------------------------------

def compare_on_corpus(
    file_paths: list[str],
    chunk_size: int = 500,
    overlap: int = 50,
    max_sentences: int = 3,
    preprocess: bool = True,
) -> dict[str, dict]:
    """So sánh 3 strategy trên nhiều tài liệu, tổng hợp thống kê.

    Args:
        file_paths: Danh sách đường dẫn tuyệt đối hoặc tương đối tới file .md/.txt.
        chunk_size, overlap, max_sentences, preprocess: như compare_on_text().

    Returns:
        {strategy_name: {per_file: [...], total_chunks, avg_length, short_chunks}}
    """
    aggregated: dict[str, dict] = {
        "fixed_size": {"per_file": [], "total_chunks": 0, "total_chars": 0, "short_chunks": 0},
        "by_sentences": {"per_file": [], "total_chunks": 0, "total_chars": 0, "short_chunks": 0},
        "recursive": {"per_file": [], "total_chunks": 0, "total_chars": 0, "short_chunks": 0},
    }

    for path in file_paths:
        text = open(path, encoding="utf-8").read()
        file_results = compare_on_text(text, chunk_size, overlap, max_sentences, preprocess)
        for strategy, stats in file_results.items():
            agg = aggregated[strategy]
            agg["per_file"].append({"file": os.path.basename(path), **stats})
            agg["total_chunks"] += stats["count"]
            agg["total_chars"] += stats["avg_length"] * stats["count"]
            agg["short_chunks"] += stats["short_chunks"]

    for strategy, agg in aggregated.items():
        n = agg["total_chunks"]
        agg["corpus_avg_length"] = round(agg["total_chars"] / n, 1) if n > 0 else 0.0

    return aggregated


# ---------------------------------------------------------------------------
# 4. So sánh retrieval qua các strategy
# ---------------------------------------------------------------------------

def build_store(
    docs: list[Document],
    chunker,
    embedding_fn: Callable = _mock_embed,
) -> EmbeddingStore:
    """Tạo EmbeddingStore từ danh sách Document với một chunker cụ thể."""
    store = EmbeddingStore(embedding_fn=embedding_fn, chunker=chunker)
    store.add_documents(docs)
    return store


def evaluate_retrieval(
    store: EmbeddingStore,
    queries: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Chạy danh sách query trên store, trả về kết quả retrieval.

    Returns:
        list of {query, hits: [{content, metadata, score}]}
    """
    return [
        {"query": q, "hits": store.search(q, top_k=top_k)}
        for q in queries
    ]


def compare_retrieval_across_strategies(
    docs: list[Document],
    queries: list[str],
    chunk_size: int = 500,
    overlap: int = 50,
    max_sentences: int = 3,
    embedding_fn: Callable = _mock_embed,
    top_k: int = 3,
    preprocess: bool = True,
) -> dict[str, dict]:
    """So sánh retrieval quality của 3 strategy trên cùng một bộ doc và queries.

    Args:
        docs: Danh sách Document đã load.
        queries: Danh sách câu query để đánh giá.
        chunk_size, overlap, max_sentences: Tham số chunking.
        embedding_fn: Hàm embedding (mặc định MockEmbedder).
        top_k: Số kết quả trả về mỗi query.
        preprocess: Có tiền xử lý Markdown artifact trước khi chunking không.

    Returns:
        {strategy_name: {store_size, results: [{query, hits}]}}
    """
    if preprocess:
        docs = [
            Document(
                id=doc.id,
                content=preprocess_legal_markdown(doc.content),
                metadata=doc.metadata,
            )
            for doc in docs
        ]

    chunkers: dict[str, object] = {
        "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=max_sentences),
        "recursive": RecursiveChunker(chunk_size=chunk_size),
    }

    comparison: dict[str, dict] = {}
    for name, chunker in chunkers.items():
        store = build_store(docs, chunker, embedding_fn)
        results = evaluate_retrieval(store, queries, top_k)
        comparison[name] = {
            "store_size": store.get_collection_size(),
            "results": results,
        }
    return comparison


# ---------------------------------------------------------------------------
# 5. In báo cáo so sánh
# ---------------------------------------------------------------------------

def print_chunking_report(
    corpus_comparison: dict[str, dict],
    chunk_size: int,
    preprocess: bool,
) -> None:
    """In bảng thống kê chunking tổng hợp trên toàn corpus."""
    print("\n" + "=" * 70)
    print(f"  CHUNKING COMPARISON  |  chunk_size={chunk_size}  |  preprocess={preprocess}")
    print("=" * 70)
    header = f"{'Strategy':<16} {'Total Chunks':>13} {'Avg Length':>11} {'Short (<50)':>12} {'Very Short (<20)':>16}"
    print(header)
    print("-" * 70)
    for strategy, agg in corpus_comparison.items():
        print(
            f"{strategy:<16} {agg['total_chunks']:>13,} "
            f"{agg['corpus_avg_length']:>11.1f} "
            f"{agg['short_chunks']:>12,} "
            f"{'(see per-file)':>16}"
        )
    print()

    # Per-file breakdown
    strategies = list(corpus_comparison.keys())
    files = [r["file"] for r in corpus_comparison[strategies[0]]["per_file"]]
    print(f"{'File':<30} " + "  ".join(f"{s[:12]:<14}" for s in strategies))
    print("-" * 70)
    for i, fname in enumerate(files):
        row = f"{fname[:28]:<30} "
        for strategy in strategies:
            st = corpus_comparison[strategy]["per_file"][i]
            row += f"  {st['count']:>4}chk/{st['avg_length']:>5.0f}avg "
        print(row)
    print()


def print_retrieval_report(comparison: dict[str, dict], top_k: int) -> None:
    """In bảng kết quả retrieval so sánh giữa các strategy."""
    print("=" * 70)
    print(f"  RETRIEVAL COMPARISON  |  top_k={top_k}")
    print("=" * 70)

    strategies = list(comparison.keys())
    queries = [r["query"] for r in comparison[strategies[0]]["results"]]

    for qi, query in enumerate(queries):
        print(f"\nQ{qi + 1}: {query[:80]}")
        print("-" * 70)
        for strategy in strategies:
            result = comparison[strategy]["results"][qi]
            store_size = comparison[strategy]["store_size"]
            hits = result["hits"]
            print(f"  [{strategy}]  store={store_size:,} chunks")
            for rank, hit in enumerate(hits, 1):
                content = hit["content"][:90].replace("\n", " ")
                score = hit["score"]
                src = hit["metadata"].get("source", hit["metadata"].get("doc_id", "?"))
                src = os.path.basename(src)
                print(f"    #{rank}  score={score:.4f}  src={src:<30}  \"{content}...\"")
        print()


# ---------------------------------------------------------------------------
# 6. Load tài liệu pháp luật từ thư mục
# ---------------------------------------------------------------------------

_DOC_TYPE_MAP = {
    "luat": "luat",
    "nghi_dinh": "nghi_dinh",
    "thong_tu": "thong_tu",
}

def load_legal_docs(data_dir: str) -> list[Document]:
    """Load tất cả file .md/.txt trong thư mục thành danh sách Document.

    Tự động suy ra document_type từ tên file (luat/nghi_dinh/thong_tu).
    """
    docs: list[Document] = []
    for fname in sorted(os.listdir(data_dir)):
        if not (fname.endswith(".md") or fname.endswith(".txt")):
            continue
        path = os.path.join(data_dir, fname)
        content = open(path, encoding="utf-8").read()
        doc_id = fname.rsplit(".", 1)[0]

        doc_type = "unknown"
        for key in _DOC_TYPE_MAP:
            if key in doc_id:
                doc_type = _DOC_TYPE_MAP[key]
                break

        docs.append(Document(
            id=doc_id,
            content=content,
            metadata={
                "source": path,
                "extension": os.path.splitext(fname)[1],
                "doc_id": doc_id,
                "document_type": doc_type,
                "language": "vi",
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# 7. Benchmark đầy đủ (entry point)
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "An ninh quốc gia là gì và bao gồm những nội dung nào?",
    "Quyền và nghĩa vụ của công dân trong bảo vệ an ninh quốc gia?",
    "Các hành vi nào bị nghiêm cấm theo quy định về an ninh quốc gia?",
    "Cơ quan nào có thẩm quyền áp dụng biện pháp pháp luật bảo vệ an ninh?",
    "Điều kiện và trình tự áp dụng biện pháp ngăn chặn trong bảo vệ an ninh?",
]


def run_benchmark(
    data_dir: str = "data/new_data",
    chunk_size: int = 500,
    top_k: int = 3,
    preprocess: bool = True,
    queries: list[str] | None = None,
) -> None:
    """Chạy benchmark đầy đủ: load docs → so sánh chunking → so sánh retrieval."""
    if queries is None:
        queries = DEFAULT_QUERIES

    print(f"\nLoading documents from: {data_dir}")
    docs = load_legal_docs(data_dir)
    print(f"  → {len(docs)} documents loaded")

    file_paths = [doc.metadata["source"] for doc in docs]

    # --- Chunking comparison ---
    print("\nRunning chunking comparison...")
    corpus_cmp = compare_on_corpus(
        file_paths, chunk_size=chunk_size, preprocess=preprocess
    )
    print_chunking_report(corpus_cmp, chunk_size=chunk_size, preprocess=preprocess)

    # --- Retrieval comparison ---
    print("Running retrieval comparison...")
    retrieval_cmp = compare_retrieval_across_strategies(
        docs, queries, chunk_size=chunk_size, top_k=top_k, preprocess=preprocess
    )
    print_retrieval_report(retrieval_cmp, top_k=top_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh chunking strategy trên tài liệu pháp luật")
    parser.add_argument("--dir", default="data/new_data", help="Thư mục chứa tài liệu (default: data/new_data)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Kích thước chunk (default: 500)")
    parser.add_argument("--top-k", type=int, default=3, help="Số kết quả retrieval (default: 3)")
    parser.add_argument("--no-preprocess", action="store_true", help="Bỏ qua bước tiền xử lý Markdown")
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.dir,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        preprocess=not args.no_preprocess,
    )
