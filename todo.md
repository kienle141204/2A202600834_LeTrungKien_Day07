# TODO Implementation Log & Improvement Suggestions

## Những gì đã implement

### `src/chunking.py`

| Method | Mô tả |
|--------|-------|
| `compute_similarity(vec_a, vec_b)` | Cosine similarity dùng `_dot()` helper có sẵn. Guard trả về `0.0` khi magnitude bằng 0. |
| `SentenceChunker.chunk(text)` | Dùng `re.split` với lookbehind để tách câu tại `. `, `! `, `? `, `.\n`. Group từng batch `max_sentences_per_chunk` câu rồi join bằng space. |
| `RecursiveChunker.chunk(text)` | Wrapper gọi `_split(text, self.separators)`. |
| `RecursiveChunker._split(text, separators)` | Đệ quy thử từng separator theo thứ tự ưu tiên. Base case: text ngắn hơn `chunk_size` hoặc hết separator → trả nguyên. Empty separator `""` bị bỏ qua (Python không split được). |
| `ChunkingStrategyComparator.compare(text, chunk_size)` | Khởi tạo 3 chunker, gọi `.chunk()`, tính `count` và `avg_length`, trả dict với keys `fixed_size`, `by_sentences`, `recursive`. |

### `src/store.py`

| Method | Mô tả |
|--------|-------|
| `__init__` (ChromaDB block) | Dùng `chromadb.EphemeralClient()` để tạo client in-memory, `get_or_create_collection` với tên collection. |
| `_make_record(doc)` | Build dict `{id, content, embedding, metadata}`. Nhúng `doc_id` vào metadata để `delete_document` và `search_with_filter` lọc được. |
| `_search_records(query, records, top_k)` | Embed query → tính `compute_similarity` với mỗi record → sort descending → slice `[:top_k]`. |
| `add_documents(docs)` | In-memory: append record vào `self._store`. ChromaDB: gọi `collection.add()`. |
| `search(query, top_k)` | In-memory: delegate sang `_search_records`. ChromaDB: `collection.query()` rồi map `distance` → `score = 1 - dist`. |
| `get_collection_size()` | `len(self._store)` hoặc `collection.count()`. |
| `search_with_filter(query, top_k, metadata_filter)` | Lọc `self._store` theo tất cả key-value trong `metadata_filter`, sau đó chạy `_search_records` trên subset đó. ChromaDB: dùng tham số `where`. |
| `delete_document(doc_id)` | In-memory: list comprehension loại bỏ record có `id == doc_id`, trả `True` nếu size giảm. ChromaDB: `collection.get()` kiểm tra tồn tại → `collection.delete()`. |

### `src/agent.py`

| Method | Mô tả |
|--------|-------|
| `__init__` | Lưu `self.store` và `self.llm_fn`. |
| `answer(question, top_k)` | RAG pattern: retrieve top-k chunks → build prompt `Context:\n...\n\nQuestion: ...\n\nAnswer:` → gọi `llm_fn(prompt)`. |

**Kết quả:** 42/42 tests pass, demo `python main.py` chạy end-to-end với mock embedder.

---

## Gợi ý cải thiện

### 1. Chunking

- **SentenceChunker có overlap**: Thêm tham số `overlap_sentences` để các chunk liền kề chia sẻ 1-2 câu cuối/đầu, tránh mất ngữ cảnh tại ranh giới.
- **Sentence detection tốt hơn**: Pattern regex hiện tại bỏ sót viết tắt (e.g., "Mr. Smith", "e.g. this"). Nên dùng `nltk.sent_tokenize` hoặc `spacy` để tách câu chính xác hơn.
- **Recursive merging**: Sau khi split, gom các chunk quá nhỏ (< `chunk_size / 2`) lại với chunk kế tiếp để tránh chunk rác 1-2 từ.
- **Header-aware chunker cho Markdown**: Chunk theo structure `## Section` thay vì theo ký tự, giữ nguyên tiêu đề trong mỗi chunk để retrieval có context rõ hơn.
- **Q&A chunker**: Với FAQ/playbook, chunk theo cặp "Q: ... A: ..." thay vì theo ký tự — cải thiện precision rõ rệt.

### 2. EmbeddingStore

- **Persistent ChromaDB**: Thêm hỗ trợ `CHROMA_PERSIST_DIR` để dữ liệu còn lại giữa các lần chạy (không cần re-embed mỗi lần).
- **Chunk-level storage**: Hiện tại mỗi `Document` là 1 record. Nên integrate chunker vào `add_documents` để tự động chunk và lưu nhiều record từ 1 doc (thêm `chunker` parameter vào constructor).
- **Metadata validation**: Validate metadata keys khi `add_documents` để phát hiện typo sớm thay vì fail silent lúc filter.
- **Batch embedding**: Thay vì embed từng doc một, gọi batch API của embedder (OpenAI hỗ trợ `input=[list_of_texts]`) để giảm latency và chi phí.
- **Duplicate ID handling**: Hiện tại thêm doc trùng ID sẽ tạo duplicate record trong in-memory store. Nên upsert thay vì append.

### 3. KnowledgeBaseAgent

- **Cited sources**: Trả về không chỉ câu trả lời mà còn danh sách `sources` (doc_id, score) để user trace được chunk nào hỗ trợ câu trả lời — quan trọng cho grounding quality.
- **Prompt engineering**: Thêm instruction "Answer only based on the provided context. If the answer is not in the context, say so." để giảm hallucination.
- **Re-ranking**: Sau retrieval bằng vector similarity, dùng cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) để re-rank top-k trước khi đưa vào LLM.
- **Hybrid search (BM25 + vector)**: Kết hợp sparse retrieval (BM25 via `rank_bm25`) với dense retrieval hiện tại bằng Reciprocal Rank Fusion — cải thiện recall cho exact-match queries.

### 4. Kiến trúc tổng thể

- **Streaming**: `llm_fn` hiện trả string đồng bộ. Nếu dùng LLM thật, nên hỗ trợ streaming callback để UX tốt hơn.
- **Async support**: Với nhiều concurrent queries, `async def` cho `search` và `answer` giúp scale tốt hơn.
- **Evaluation pipeline**: Tự động hóa benchmark với RAGAS hay LangSmith — tính `context_precision`, `faithfulness`, `answer_relevancy` thay vì chỉ dùng cosine score.
