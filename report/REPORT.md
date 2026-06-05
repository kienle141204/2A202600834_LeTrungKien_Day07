# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Trung Kiên
**MSSV:** 2A202600834
**Nhóm:** Nhóm kbiet
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Cosine similarity cao (gần 1.0) có nghĩa là hai vector embedding hướng về cùng một "phương" trong không gian nhiều chiều — tức là hai đoạn văn bản biểu đạt ý nghĩa tương đồng nhau. Độ tương đồng này không phụ thuộc vào độ dài của văn bản mà chỉ đo "góc" giữa hai vector.

**Ví dụ HIGH similarity:**
- Sentence A: "A vector store is a database designed to store and search embeddings."
- Sentence B: "Vector databases are used to index and retrieve embedding vectors efficiently."
- Tại sao tương đồng: Hai câu cùng nói về vector store/database, cùng chủ đề kỹ thuật với từ vựng trùng lặp (vector, store, database, embeddings), embedding model sẽ gán hai vector gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Cosine similarity measures the angle between two embedding vectors."
- Sentence B: "The soup is too hot to eat right now."
- Tại sao khác: Hai câu thuộc hai miền ngữ nghĩa hoàn toàn khác nhau (kỹ thuật AI vs. đời thường), không có từ nào chung, embedding sẽ tạo ra hai vector gần như vuông góc.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity bất biến với độ lớn của vector — một đoạn văn ngắn và một đoạn văn dài nói về cùng chủ đề sẽ vẫn có similarity cao dù vector của chúng có magnitude khác nhau. Euclidean distance lại bị ảnh hưởng bởi magnitude, nên đoạn văn dài hơn sẽ có distance lớn hơn một cách giả tạo, gây ra kết quả không chính xác trong semantic search.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Áp dụng công thức:

```
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
           = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = ceil(24.75)
           = 25 chunks
```

Tăng overlap từ 50 lên 100 làm tăng số chunk từ 23 lên 25. Lý do muốn overlap nhiều hơn: đảm bảo thông tin nằm ở ranh giới giữa hai chunk không bị mất — câu hoặc ý tưởng spanning hai chunk vẫn xuất hiện đầy đủ trong ít nhất một chunk, cải thiện retrieval precision cho nội dung ở vùng biên.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Văn bản pháp luật Việt Nam / Legal Document Retrieval

**Tại sao nhóm chọn domain này?**

Nhóm chọn domain văn bản pháp luật Việt Nam vì bộ dữ liệu gồm nhiều loại tài liệu như Luật, Nghị định và Thông tư. Đây là domain phù hợp để thử nghiệm semantic search và RAG vì người dùng thường không nhớ chính xác tên văn bản, mà sẽ hỏi theo nội dung hoặc vấn đề pháp lý. Hệ thống retrieval có thể giúp tìm đúng văn bản liên quan và hỗ trợ tóm tắt nội dung từ các nguồn đã cung cấp.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 001_luat_52656.md | data/new_data/ | 26,525 | source, extension=.md, doc_id=001_luat_52656, document_type=luat, language=vi |
| 2 | 002_nghi_dinh_8908.md | data/new_data/ | 23,631 | source, extension=.md, doc_id=002_nghi_dinh_8908, document_type=nghi_dinh, language=vi |
| 3 | 003_nghi_dinh_14849.md | data/new_data/ | 19,728 | source, extension=.md, doc_id=003_nghi_dinh_14849, document_type=nghi_dinh, language=vi |
| 4 | 004_nghi_dinh_124155.md | data/new_data/ | 10,866 | source, extension=.md, doc_id=004_nghi_dinh_124155, document_type=nghi_dinh, language=vi |
| 5 | 005_nghi_dinh_164307.md | data/new_data/ | 14,061 | source, extension=.md, doc_id=005_nghi_dinh_164307, document_type=nghi_dinh, language=vi |
| 6 | 006_nghi_dinh_219863.md | data/new_data/ | 10,121 | source, extension=.md, doc_id=006_nghi_dinh_219863, document_type=nghi_dinh, language=vi |
| 7 | 007_thong_tu_18000.md | data/new_data/ | 29,500 | source, extension=.md, doc_id=007_thong_tu_18000, document_type=thong_tu, language=vi |
| 8 | 008_thong_tu_304257.md | data/new_data/ | 17,812 | source, extension=.md, doc_id=008_thong_tu_304257, document_type=thong_tu, language=vi |
| 9 | 009_thong_tu_313701.md | data/new_data/ | 21,774 | source, extension=.md, doc_id=009_thong_tu_313701, document_type=thong_tu, language=vi |
| 10 | 010_thong_tu_316062.md | data/new_data/ | 35,365 | source, extension=.md, doc_id=010_thong_tu_316062, document_type=thong_tu, language=vi |

**Tổng: 10 tài liệu, ~209,383 ký tự** (1 Luật, 5 Nghị định, 4 Thông tư)

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/new_data/001_luat_52656.md" | Truy vết nguồn gốc chunk, cho phép cite chính xác văn bản |
| extension | string | ".md" | Nhận diện loại file được ingest |
| doc_id | string | "001_luat_52656" | Gom hoặc xóa toàn bộ chunk thuộc cùng một văn bản |
| chunk_index | int | 0, 1, 2 | Xác định đoạn được truy xuất nằm ở phần nào trong văn bản |
| document_type | string | "luat" / "nghi_dinh" / "thong_tu" | Filter theo loại văn bản pháp luật khi query |
| language | string | "vi" | Xác định ngôn ngữ tài liệu để thiết kế retrieval phù hợp |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Kết quả chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên 2 tài liệu pháp luật:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 001_luat_52656.md (26,525 chars) | fixed_size | 177 | 199.6 | Trung bình — cắt giữa câu/điều khoản |
| 001_luat_52656.md (26,525 chars) | by_sentences | 73 | 361.0 | Tốt — giữ câu nguyên vẹn |
| 001_luat_52656.md (26,525 chars) | recursive | 2,636 | 9.0 | **Rất kém** — chunk cực nhỏ (artifact Markdown) |
| 004_nghi_dinh_124155.md (10,866 chars) | fixed_size | 73 | 198.2 | Trung bình |
| 004_nghi_dinh_124155.md (10,866 chars) | by_sentences | 24 | 450.4 | Tốt — chunk vừa |
| 004_nghi_dinh_124155.md (10,866 chars) | recursive | 1,379 | 6.8 | **Rất kém** — chunk cực nhỏ |

**Nhận xét:** `RecursiveChunker` với `chunk_size=200` hoàn toàn thất bại trên tài liệu pháp luật Việt Nam — tạo ra hơn 2,600 chunk với trung bình chỉ 9 ký tự. Nguyên nhân: văn bản pháp luật được convert từ PDF có nhiều dòng ngắn (dấu phân cách `---`, số điều như `1.`, `2.`, từ cuối dòng bị ngắt như `"cùng"`, `"hồi"`). RecursiveChunker tách theo `\n` và `\n\n` tạo ra vô số fragment vô nghĩa.

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=500)

**Mô tả cách hoạt động:**

RecursiveChunker thử tách text theo danh sách separator theo thứ tự ưu tiên: `["\n\n", "\n", ". ", " ", ""]`. Với mỗi đoạn text, nếu tách theo `\n\n` (đoạn văn) tạo ra phần đủ nhỏ thì dừng, không thì thử `\n` (dòng mới), rồi `. ` (câu), rồi ` ` (từ). Mỗi phần bị tách tiếp tục được xử lý đệ quy cho đến khi nằm trong `chunk_size`. Kết quả là danh sách chunk tôn trọng ranh giới cấu trúc tự nhiên của văn bản.

**Tại sao tôi chọn strategy này cho domain nhóm?**

Văn bản pháp luật Việt Nam có cấu trúc phân cấp rõ ràng: Chương → Điều → Khoản → Điểm. Mỗi Điều là một đơn vị ngữ nghĩa độc lập — đây là "natural boundary" lý tưởng để tách chunk. RecursiveChunker với `chunk_size=500` ưu tiên tách theo `\n\n` (giữa các Điều), giúp mỗi chunk thường tương ứng một hoặc vài khoản của cùng một Điều. Điều này quan trọng vì query pháp lý như "điều kiện áp dụng biện pháp X" cần tìm đúng một Điều cụ thể, không phải các fragment rải rác.

**Code snippet (strategy cá nhân):**
```python
from src.chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=500)
store = EmbeddingStore(embedding_fn=_mock_embed, chunker=chunker)
```

### So Sánh: Strategy của tôi vs Baseline

Chạy trên `001_luat_52656.md` với chunk_size=500:

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 001_luat_52656.md | fixed_size (500) | 59 | 498.7 | Trung bình — cắt giữa Điều |
| 001_luat_52656.md | by_sentences (mặc định) | 73 | 361.0 | Tốt — câu nguyên vẹn |
| 001_luat_52656.md | **recursive (500)** | **195** | **134.0** | Trung bình — nhiều chunk nhỏ do artifact |

Với chunk_size=500, RecursiveChunker cải thiện đáng kể (195 chunk vs 2,636 ở baseline) nhưng avg_length 134 chars vẫn thấp hơn lý tưởng — cho thấy vẫn còn nhiều artifact Markdown ngắn. `by_sentences` cho chunk ổn định hơn (avg 361 chars). `fixed_size` cho chunk đều nhất nhưng hay cắt giữa Điều/Khoản.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Chunk Count (001_luat) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Trung Kiên | RecursiveChunker (500) | 195 | Tôn trọng ranh giới cấu trúc | Nhiều chunk nhỏ do artifact |
| [Thành viên 2] | SentenceChunker (max=3) | 73 | Chunk đồng đều, câu nguyên vẹn | Chunk quá dài cho văn bản điều khoản dài |
| [Thành viên 3] | FixedSizeChunker (400, overlap=80) | ~74 | Số chunk dự đoán được, overlap bảo vệ biên | Cắt giữa Điều/Khoản, mất ngữ cảnh pháp lý |

**Strategy nào tốt nhất cho domain này? Tại sao?**

Với văn bản pháp luật Việt Nam convert từ PDF, `SentenceChunker` cho kết quả ổn định nhất (73 chunk, avg 361 chars) vì tách theo câu — phù hợp với văn phong pháp lý thường viết từng khoản là một câu dài. RecursiveChunker cần thêm bước tiền xử lý (loại bỏ artifact, gộp dòng ngắn) trước khi áp dụng để đạt hiệu quả cao nhất.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Dùng `re.split()` với lookahead để tách trên các dấu kết thúc câu (`. `, `! `, `? `, `.\n`) mà vẫn giữ dấu câu cùng với câu đó. Sau khi tách, gom nhóm `max_sentences_per_chunk` câu liên tiếp thành một chunk và join bằng dấu cách. Edge case xử lý: câu rỗng sau strip được bỏ qua, max_sentences tối thiểu là 1.

**`RecursiveChunker.chunk` / `_split`** — approach:

Dùng đệ quy với danh sách separator giảm dần về độ ưu tiên. Base case: text đủ nhỏ hơn `chunk_size` (return trực tiếp) hoặc hết separator (return nguyên text). Với mỗi separator, split text thành các phần; nếu separator không tìm thấy (chỉ ra 1 phần), thử separator tiếp theo. Separator rỗng `""` bỏ qua để tránh vòng lặp vô hạn.

### EmbeddingStore

**`add_documents` + `search`** — approach:

`add_documents` kiểm tra có chunker hay không: nếu có, gọi `_make_chunk_records()` để tạo nhiều record từ 1 document với metadata `chunk_index` và `total_chunks`; nếu không, tạo 1 record/document. Tất cả record được lưu vào in-memory list `self._store`. `search` embed query, tính cosine similarity với tất cả record, sort descending, trả về top_k.

**`search_with_filter` + `delete_document`** — approach:

`search_with_filter` pre-filter store list bằng dictionary comprehension — chỉ giữ record có metadata khớp với tất cả key-value trong `metadata_filter` — rồi chạy similarity search trên subset đó. `delete_document` filter ngược: tạo list mới bỏ qua record có `metadata["doc_id"] == doc_id`, trả về True nếu size thay đổi.

### KnowledgeBaseAgent

**`answer`** — approach:

Pipeline 3 bước: (1) `store.search(question, top_k)` lấy top-k chunk có score cao nhất; (2) format thành context string với numbering `[i]` và source citation; (3) inject context vào prompt template với instruction "answer ONLY from context". Nếu không có chunk nào relevant, LLM sẽ phản hồi "I don't have enough information" theo instruction trong prompt.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.8, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\AI_in_action\Day-07-Lab-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Chạy `compute_similarity()` với `MockEmbedder` (MD5-seeded 64-dim random vectors) trên các cặp câu tiếng Việt thuộc domain pháp luật:

| Pair | Sentence A (tóm tắt) | Sentence B (tóm tắt) | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | An ninh quốc gia là sự ổn định... | Bảo vệ an ninh quốc gia là nhiệm vụ... | high | +0.0347 | ❌ |
| 2 | Nghị định quy định biện pháp bảo vệ an ninh | Thông tư hướng dẫn biện pháp bảo vệ trật tự | high | -0.0760 | ❌ |
| 3 | Công dân có quyền tố cáo hành vi xâm phạm an ninh | Thời tiết hôm nay rất đẹp | low | +0.1571 | ❌ |
| 4 | Điều 1 quy định phạm vi Luật An ninh quốc gia | Phạm vi áp dụng nêu tại điều đầu tiên | high | +0.0277 | ❌ |
| 5 | Cơ quan Công an có thẩm quyền áp dụng biện pháp ngăn chặn | Bộ Công an thực thi pháp luật về an ninh | high | +0.1892 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

Pair 3 bất ngờ nhất: câu về pháp luật và câu về thời tiết hoàn toàn không liên quan nhưng lại có score dương (+0.1571) — cao hơn cả Pair 1 (hai câu về an ninh quốc gia, score +0.0347) và Pair 4 (hai câu gần như đồng nghĩa về phạm vi điều luật, score chỉ +0.0277). Pair 5 có score cao nhất (+0.1892) nhưng đây chỉ là sự ngẫu nhiên của MD5-seeded vectors. Kết quả xác nhận: `MockEmbedder` hoàn toàn không phản ánh ngữ nghĩa tiếng Việt — các vector random tạo ra bởi MD5 hash không có liên hệ nào với nội dung pháp luật. Để RAG hoạt động đúng với tiếng Việt, cần embedder được huấn luyện trên ngữ liệu tiếng Việt (ví dụ: `bkai-foundation-models/vietnamese-bi-encoder` hoặc multilingual sentence-transformers).

---

## 6. Results — Cá nhân (10 điểm)

**Setup:** RecursiveChunker(chunk_size=500), MockEmbedder, 10 tài liệu pháp luật từ `data/new_data/`, tổng **4,205 chunks**.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | An ninh quốc gia là gì và bao gồm những nội dung nào? | An ninh quốc gia là sự ổn định, phát triển bền vững của chế độ và Nhà nước CHXHCNVN, sự bất khả xâm phạm độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ (Điều 3, Luật 32/2004/QH11) |
| 2 | Quyền và nghĩa vụ của công dân trong bảo vệ an ninh quốc gia? | Công dân có quyền và nghĩa vụ tham gia bảo vệ an ninh, tố cáo hành vi xâm phạm an ninh; bị hạn chế một số quyền khi bị áp dụng biện pháp bảo vệ an ninh |
| 3 | Các hành vi nào bị nghiêm cấm theo quy định về an ninh quốc gia? | Hành vi xâm phạm chế độ chính trị, hoạt động phá hoại, tuyên truyền chống Nhà nước, gián điệp, khủng bố, bạo loạn (Điều 13, Luật ANQG) |
| 4 | Cơ quan nào có thẩm quyền áp dụng biện pháp pháp luật bảo vệ an ninh? | Cơ quan chuyên trách bảo vệ an ninh (Công an, Quân đội); phối hợp các Bộ, UBND tỉnh/thành phố theo thẩm quyền quy định (Nghị định 35/2011) |
| 5 | Điều kiện và trình tự áp dụng biện pháp ngăn chặn trong bảo vệ an ninh? | Phải có căn cứ theo luật định, do cơ quan có thẩm quyền ra quyết định, phải thông báo cho người bị áp dụng và cơ quan có trách nhiệm giám sát |

### Kết Quả Của Tôi

| # | Query (tóm tắt) | Top-1 Source | Score | Relevant? | Ghi chú |
|---|-------|--------------------------------|-------|-----------|--------|
| 1 | An ninh quốc gia là gì? | 007_thong_tu_18000.md | 0.4261 | ❌ No | Chunk là fragment "cùng" — artifact Markdown |
| 2 | Quyền và nghĩa vụ công dân? | 005_nghi_dinh_164307.md | 0.4165 | ✅ Một phần | Chunk về trách nhiệm Bộ trưởng — liên quan gián tiếp |
| 3 | Hành vi bị nghiêm cấm? | 007_thong_tu_18000.md | 0.3886 | ❌ No | Chunk là fragment "hồi" — artifact Markdown |
| 4 | Cơ quan có thẩm quyền? | 006_nghi_dinh_219863.md | 0.4665 | ❌ No | Chunk là dòng phân cách "--------" — artifact |
| 5 | Điều kiện biện pháp ngăn chặn? | 003_nghi_dinh_14849.md | 0.3698 | ✅ Một phần | Chunk về dự trữ bảo đảm an ninh — liên quan gián tiếp |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 0 / 5 fully relevant; 2 / 5 partially relevant (Q2, Q5); 3 / 5 failed hoàn toàn (Q1, Q3, Q4)

**Nhận xét:** Retrieval hoàn toàn thất bại vì hai lý do cộng hưởng: (1) MockEmbedder tạo random vectors không phản ánh ngữ nghĩa tiếng Việt; (2) RecursiveChunker tạo ra hàng nghìn chunk cực ngắn (dấu phân cách, số điều, từ lẻ) từ artifact của quá trình convert PDF→Markdown — những "chunk" rác này ngẫu nhiên được score cao nhất. Filter theo `document_type=luat` hoạt động đúng (chỉ trả về chunks từ 001_luat_52656.md), nhưng chưa cải thiện được relevance.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Thành viên dùng `SentenceChunker` chỉ ra rằng văn bản pháp luật Việt Nam thường viết mỗi khoản là một câu dài — SentenceChunker tách đúng theo ranh giới này và tạo ra chunk ổn định hơn (73 chunk, avg 361 chars) so với RecursiveChunker (195 chunk, avg 134 chars bao gồm artifact). Điều này khiến tôi nhận ra: với dữ liệu chứa nhiều artifact từ PDF conversion, strategy tách theo câu ổn định hơn strategy tách theo cấu trúc Markdown.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

Một nhóm thực hiện bước **tiền xử lý** trước khi chunking: loại bỏ dòng phân cách (`---`, `***`), gộp dòng ngắn hơn 20 ký tự vào đoạn trước, chuẩn hóa khoảng trắng. Sau tiền xử lý, RecursiveChunker hoạt động tốt hơn rất nhiều — đây là bài học quan trọng: pipeline RAG không chỉ gồm chunk→embed→search mà còn cần bước **data cleaning** đặc biệt với dữ liệu convert từ PDF.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Ba thay đổi chính: (1) **Tiền xử lý dữ liệu** — loại bỏ artifact Markdown (dòng phân cách, số điều lẻ, từ cuối dòng bị ngắt) trước khi đưa vào chunker; (2) **Dùng Vietnamese embedder** — MockEmbedder hoàn toàn vô dụng với tiếng Việt, cần multilingual sentence-transformers hoặc embedder tiếng Việt chuyên biệt để benchmark có ý nghĩa; (3) **Thêm metadata `article_id`** để truy xuất từng Điều cụ thể khi người dùng hỏi theo số Điều — đây là use case rất phổ biến với tài liệu pháp luật.

### Failure Case Analysis

**3 queries thất bại hoàn toàn (Q1, Q3, Q4):** Top-1 retrieved chunks đều là artifact Markdown — một từ (`"cùng"`, `"hồi"`) hoặc dòng phân cách (`"--------"`), không chứa nội dung pháp lý nào.

**Nguyên nhân kép:**
- **Embedder:** MockEmbedder gán random score nên artifact ngắn ngẫu nhiên được score cao (0.39–0.47)
- **Chunker:** RecursiveChunker tách theo `\n` tạo ra 2,636 chunk cực ngắn từ artifact PDF conversion; với 4,205 chunks tổng, xác suất artifact xuất hiện ở top-1 rất cao

**Đề xuất cải thiện theo thứ tự ưu tiên:**
1. Tiền xử lý: lọc bỏ chunk có độ dài < 50 ký tự sau khi chunking
2. Dùng Vietnamese sentence-transformers để có semantic similarity thực sự
3. Thêm metadata `article_id` + `chapter_id` để hỗ trợ filter theo Điều/Chương
4. Cân nhắc `SentenceChunker` thay `RecursiveChunker` cho tài liệu pháp luật convert từ PDF

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
