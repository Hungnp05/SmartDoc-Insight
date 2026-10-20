"""
SmartDoc-Insight Test Suite
────────────────────────────
Tests for all major components.
Run: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Minimal config mock for unit tests."""
    from src.config import config
    return config


@pytest.fixture
def mock_ollama():
    """Mock Ollama client that returns dummy responses."""
    client = MagicMock()
    client.query.return_value = "Doanh thu quý 3 tăng 20% so với quý 2."
    client.embed.return_value = [[0.1] * 768]
    client.vision_query.return_value = "Biểu đồ cột cho thấy doanh thu tăng đều qua các quý."
    client.list_models.return_value = ["llama3:8b", "llava:7b", "nomic-embed-text"]
    client.is_model_available.return_value = True
    return client


@pytest.fixture
def sample_table_html():
    return """
    <table>
        <tr><th>Quý</th><th>Doanh thu (tỷ VND)</th><th>Tăng trưởng</th></tr>
        <tr><td>Q1 2024</td><td>1,200</td><td>—</td></tr>
        <tr><td>Q2 2024</td><td>1,450</td><td>+20.8%</td></tr>
        <tr><td>Q3 2024</td><td>1,740</td><td>+20.0%</td></tr>
        <tr><td>Q4 2024</td><td>2,100</td><td>+20.7%</td></tr>
    </table>
    """


@pytest.fixture
def sample_image():
    """Create a simple test image (white background with black text)."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White
    # Add some black region to simulate text
    img[50:150, 50:350] = 50
    return img


# ── Table Extractor Tests ─────────────────────────────────────────────────────

class TestTableExtractor:

    def test_html_to_markdown_basic(self, sample_table_html):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown(sample_table_html)
        assert "Quý" in result
        assert "Doanh thu" in result
        assert "---" in result  # Separator row
        assert "Q3 2024" in result
        assert "|" in result

    def test_html_to_markdown_preserves_numbers(self, sample_table_html):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown(sample_table_html)
        assert "1,740" in result
        assert "+20.0%" in result

    def test_html_to_markdown_empty(self):
        from src.utils.table_extractor import html_table_to_markdown
        result = html_table_to_markdown("<table></table>")
        # Should not crash, returns something
        assert isinstance(result, str)

    def test_html_to_markdown_single_row(self):
        from src.utils.table_extractor import html_table_to_markdown
        html = "<table><tr><th>A</th><th>B</th></tr></table>"
        result = html_table_to_markdown(html)
        assert "A" in result
        assert "B" in result


# ── Smart Chunker Tests ───────────────────────────────────────────────────────

class TestSmartChunker:

    def test_text_chunking_basic(self, mock_config):
        from src.layers.knowledge_base import SmartChunker
        chunker = SmartChunker(mock_config)
        long_text = "Đây là một câu văn. " * 100
        chunks = chunker._split_text(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= mock_config.chunk.text_chunk_size * 1.2  # Some tolerance

    def test_short_text_not_split(self, mock_config):
        from src.layers.knowledge_base import SmartChunker
        chunker = SmartChunker(mock_config)
        short_text = "Doanh thu tăng 20%."
        chunks = chunker._split_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_table_stored_as_single_chunk(self, mock_config, mock_ollama):
        """Tables should never be split — stored as one chunk."""
        from src.layers.knowledge_base import SmartChunker
        from src.layers.vision_processing import ProcessedDocument, ProcessedPage, DocumentRegion, BoundingBox

        chunker = SmartChunker(mock_config)

        # Create a mock document with one table region
        region = DocumentRegion(
            region_type="table",
            bbox=BoundingBox(0, 0, 100, 100),
            confidence=0.9,
            page_num=0,
            content="| Q1 | Q2 | Q3 |\n|---|---|---|\n| 100 | 120 | 144 |"
        )
        page = ProcessedPage(page_num=0, page_summary="Quarterly data", regions=[region])
        doc = ProcessedDocument(source_file="test.pdf", total_pages=1, pages=[page])

        chunks = chunker.chunk_document(doc)
        # Table should result in exactly 1 chunk (not split)
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
        assert len(table_chunks) == 1
        assert "Q1" in table_chunks[0].text
        assert "Q3" in table_chunks[0].text

    def test_chunk_metadata_populated(self, mock_config, mock_ollama):
        """Every chunk must have required metadata fields."""
        from src.layers.knowledge_base import SmartChunker
        from src.layers.vision_processing import ProcessedDocument, ProcessedPage, DocumentRegion, BoundingBox

        chunker = SmartChunker(mock_config)
        region = DocumentRegion(
            region_type="text", bbox=BoundingBox(0, 0, 100, 50),
            confidence=0.95, page_num=2,
            content="Báo cáo tài chính năm 2024."
        )
        page = ProcessedPage(page_num=2, page_summary="Financial report", regions=[region])
        doc = ProcessedDocument(source_file="annual.pdf", total_pages=10, pages=[page])

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "page" in chunk.metadata
            assert "content_type" in chunk.metadata
            assert chunk.metadata["source_file"] == "annual.pdf"
            assert chunk.metadata["page"] == 3  # 1-indexed


# ── Retrieval & Reasoning Tests ───────────────────────────────────────────────

class TestRetrievalReasoning:

    def test_context_builder_basic(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="1", text="Doanh thu Q3 là 1,740 tỷ đồng.",
                    metadata={"page": 2, "content_type": "table", "source_file": "report.pdf", "page_summary": "Revenue data"}
                ),
                score=0.92
            )
        ]
        context = ContextBuilder.build(chunks, "Doanh thu Q3?")
        assert "1,740" in context
        assert "table" in context.lower()
        assert "Page 2" in context

    def test_context_builder_empty(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        context = ContextBuilder.build([], "test query")
        assert "không tìm thấy" in context.lower() or len(context) < 100

    def test_source_conversion(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="2", text="Biểu đồ tăng trưởng.",
                    metadata={"page": 3, "content_type": "figure", "source_file": "report.pdf", "page_summary": "Charts"}
                ),
                score=0.85
            )
        ]
        sources = ContextBuilder.chunks_to_sources(chunks)
        assert len(sources) == 1
        assert sources[0].content_type == "figure"
        assert sources[0].page == 3
        assert sources[0].score == 0.85

    def test_source_type_emoji(self):
        from src.layers.retrieval_reasoning import Source
        s_table = Source("content", "table", 1, "file.pdf", 0.9)
        s_figure = Source("content", "figure", 1, "file.pdf", 0.9)
        s_text = Source("content", "text", 1, "file.pdf", 0.9)
        assert s_table.type_emoji == "📊"
        assert s_figure.type_emoji == "📈"
        assert s_text.type_emoji == "📝"


# ── Cross-Encoder Re-ranker Tests ─────────────────────────────────────────────

class TestReranker:

    def test_rerank_fewer_than_topk(self, mock_config):
        """If fewer chunks than top_k, return all without re-ranking."""
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        reranker = CrossEncoderReranker(mock_config)
        chunks = [
            RetrievedChunk(chunk=Chunk("1", "text", {}), score=0.8),
            RetrievedChunk(chunk=Chunk("2", "text", {}), score=0.7),
        ]
        result = reranker.rerank("query", chunks, top_k=5)
        assert len(result) == 2  # All returned since fewer than top_k

    def test_score_based_rerank_fallback(self, mock_config):
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        from src.layers.knowledge_base import RetrievedChunk, Chunk

        reranker = CrossEncoderReranker(mock_config)
        chunks = [
            RetrievedChunk(chunk=Chunk("1", "a", {}), score=0.5),
            RetrievedChunk(chunk=Chunk("2", "b", {}), score=0.9),
            RetrievedChunk(chunk=Chunk("3", "c", {}), score=0.7),
            RetrievedChunk(chunk=Chunk("4", "d", {}), score=0.3),
            RetrievedChunk(chunk=Chunk("5", "e", {}), score=0.8),
        ]
        result = reranker._score_based_rerank(chunks, top_k=3)
        assert len(result) == 3
        assert result[0].score == 0.9  # Highest first
        assert result[1].score == 0.8


# ── Integration-style test (no Ollama needed) ─────────────────────────────────

class TestIntegration:

    def test_pipeline_import(self):
        """Ensure pipeline can be imported without errors."""
        from src.pipeline import SmartDocPipeline
        assert SmartDocPipeline is not None

    def test_config_import(self):
        # Values come from .env when present, so only check invariants.
        from src.config import config
        assert config.ollama.llm_model
        assert config.ollama.vision_model
        assert 0 < config.chroma.rerank_top_k <= config.chroma.top_k
        assert config.ollama.max_tokens < config.ollama.num_ctx

    def test_chunk_size_config(self):
        from src.config import config
        assert config.chunk.text_chunk_size > 0
        assert config.chunk.text_chunk_overlap < config.chunk.text_chunk_size

    def test_data_dirs_created(self):
        from src.config import CHROMA_DIR, UPLOAD_DIR, CACHE_DIR
        assert CHROMA_DIR.exists()
        assert UPLOAD_DIR.exists()
        assert CACHE_DIR.exists()


# ── Vision Layer Tests (sample PDF, Ollama mocked) ────────────────────────────

SAMPLE_PDF = Path(__file__).parent.parent / "test_data" / "sample_report.pdf"


@pytest.fixture(scope="module")
def processed_sample():
    from src.config import config
    from src.layers.vision_processing import VisionProcessingLayer
    ollama = MagicMock()
    ollama.query.return_value = "Page summary."
    ollama.vision_query.return_value = "Line chart showing revenue increasing."
    doc = VisionProcessingLayer(config, ollama).process_document(SAMPLE_PDF)
    return doc, ollama


class TestVisionProcessing:

    def test_tables_extracted_whole_with_header(self, processed_sample):
        doc, _ = processed_sample
        tables = [r for r in doc.all_regions() if r.region_type == "table"]
        assert len(tables) == 5
        q_table = next(t for t in tables if "Quy 3/2024" in t.content)
        # Header and the matching row are in the same region, so values keep their columns.
        assert "|Doanh thu|1.820|2.050|2.340|2.550|8.760|" in q_table.content
        assert "Bang 1" in q_table.caption

    def test_vector_charts_detected_and_described(self, processed_sample):
        doc, ollama = processed_sample
        figures = [r for r in doc.all_regions() if r.region_type == "figure"]
        assert len(figures) == 2
        assert ollama.vision_query.call_count == 2
        assert all(f.raw_image is not None and f.raw_image.ndim == 3 for f in figures)
        chart1 = next(f for f in figures if f.page_num == 0)
        assert "Bieu do 1" in chart1.content
        assert "Quy 4" in chart1.labels

    def test_table_text_not_duplicated_as_text(self, processed_sample):
        doc, _ = processed_sample
        texts = "\n".join(r.content for r in doc.all_regions() if r.region_type == "text")
        assert "Giai phap phan mem\n2.450" not in texts
        assert "Rui ro canh tranh" in texts

    def test_model_calls_batched_by_phase(self, processed_sample):
        """All vision calls happen before any page summary (one VRAM model swap)."""
        _, ollama = processed_sample
        names = [c[0] for c in ollama.mock_calls if c[0] in ("vision_query", "query")]
        assert names == sorted(names, key=lambda n: n != "vision_query")

    def test_image_upload_without_ocr_goes_to_vision_model(self, tmp_path, sample_image):
        from PIL import Image
        from src.config import config
        from src.layers.vision_processing import VisionProcessingLayer
        path = tmp_path / "chart.png"
        Image.fromarray(sample_image).save(path)
        ollama = MagicMock()
        ollama.vision_query.return_value = "A bar chart of quarterly revenue."
        ollama.query.return_value = "Summary."
        layer = VisionProcessingLayer(config, ollama)
        layer.ocr.enabled = False
        doc = layer.process_document(path)
        regions = doc.all_regions()
        assert [r.region_type for r in regions] == ["figure"]
        assert "bar chart" in regions[0].content

    def test_large_table_split_repeats_header(self, mock_config):
        from src.layers.vision_processing import VisionProcessingLayer
        layer = VisionProcessingLayer(mock_config, MagicMock())
        header = "|Name|Value|\n|---|---|"
        rows = "\n".join(f"|row{i}|{i}|" for i in range(2000))
        parts = layer._split_table(f"{header}\n{rows}")
        assert len(parts) > 1
        assert all(p.startswith(header) for p in parts)
        assert all(len(p) <= mock_config.chunk.table_max_chars for p in parts)


# ── Ollama Client Tests ───────────────────────────────────────────────────────

class TestOllamaClient:

    def _client(self, mock_config):
        from src.models.ollama_client import OllamaClient
        with patch("src.models.ollama_client.requests.get"):
            return OllamaClient(mock_config)

    def test_embed_batches_and_raises_on_error(self, mock_config):
        from src.models.ollama_client import OllamaError
        client = self._client(mock_config)
        ok = MagicMock(status_code=200, ok=True)
        ok.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        with patch("src.models.ollama_client.requests.post", return_value=ok) as post:
            assert client.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
            assert post.call_count == 1

        bad = MagicMock(status_code=500, ok=False, text="boom")
        bad.json.return_value = {"error": "boom"}
        with patch("src.models.ollama_client.requests.post", return_value=bad):
            with pytest.raises(OllamaError):
                client.embed(["a"])

    def test_temperature_zero_is_respected(self, mock_config):
        client = self._client(mock_config)
        payload = client._build_payload("m", "p", None, temperature=0.0, max_tokens=None, stream=False)
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_ctx"] == mock_config.ollama.num_ctx

    def test_model_name_matching(self, mock_config):
        client = self._client(mock_config)
        available = ["llama3:8b", "nomic-embed-text:latest"]
        assert client.is_model_available("nomic-embed-text", available)
        assert client.is_model_available("llama3:8b", available)
        assert not client.is_model_available("llama3", available)
        assert not client.is_model_available("llava:7b", available)


# ── Knowledge Base Tests (temporary ChromaDB) ─────────────────────────────────

class TestKnowledgeBase:

    @pytest.fixture
    def kb(self, tmp_path):
        from dataclasses import replace
        from src.config import Config
        from src.layers.knowledge_base import KnowledgeBaseLayer
        cfg = Config()
        cfg.chroma = replace(Config.chroma, persist_dir=str(tmp_path / "chroma"))
        ollama = MagicMock()
        ollama.embed.side_effect = lambda texts, model=None: [[1.0, 0.0, 0.0] for _ in texts]
        return KnowledgeBaseLayer(cfg, ollama), ollama

    def _doc(self, content="Doanh thu nam 2024 dat 8.760 ty dong, tang truong 24% so voi 2023."):
        from src.layers.vision_processing import ProcessedDocument, ProcessedPage, DocumentRegion, BoundingBox
        region = DocumentRegion("text", BoundingBox(0, 0, 10, 10), 1.0, 0, content)
        return ProcessedDocument("a.pdf", 1, [ProcessedPage(0, "summary", [region])])

    def test_failed_embedding_keeps_previous_version(self, kb):
        kb, ollama = kb
        assert kb.ingest_document(self._doc()) == 1
        ollama.embed.side_effect = RuntimeError("ollama down")
        with pytest.raises(RuntimeError):
            kb.ingest_document(self._doc())
        assert kb.get_stats()["total_chunks"] == 1

    def test_reingest_replaces_and_clear_all(self, kb):
        kb, _ = kb
        kb.ingest_document(self._doc())
        kb.ingest_document(self._doc())
        assert kb.get_stats()["total_chunks"] == 1
        assert kb.has_document("a.pdf")
        kb.clear_all()
        assert kb.get_stats()["total_chunks"] == 0

    def test_nomic_prefixes(self, kb):
        kb, ollama = kb
        kb.ingest_document(self._doc())
        assert ollama.embed.call_args[0][0][0].startswith("search_document: ")
        kb.retrieve("doanh thu")
        assert ollama.embed.call_args[0][0][0].startswith("search_query: ")

    def test_retrieve_similarity_is_cosine(self, kb):
        kb, _ = kb
        kb.ingest_document(self._doc())
        result = kb.retrieve("doanh thu")
        assert result[0].score == pytest.approx(1.0, abs=1e-3)

    def test_long_unpunctuated_text_is_hard_split(self, mock_config):
        from src.layers.knowledge_base import SmartChunker
        chunks = SmartChunker(mock_config)._split_text("x" * (mock_config.chunk.text_chunk_size * 3))
        assert all(len(c) <= mock_config.chunk.text_chunk_size for c in chunks)


# ── Retrieval Tests ───────────────────────────────────────────────────────────

class TestRetrievalExtras:

    def _rc(self, text, page=1, ctype="text", score=0.5):
        from src.layers.knowledge_base import RetrievedChunk, Chunk
        return RetrievedChunk(Chunk(text, text, {"page": page, "content_type": ctype, "source_file": "a.pdf"}), score)

    def test_context_keeps_rerank_order_and_budget(self):
        from src.layers.retrieval_reasoning import ContextBuilder
        chunks = [self._rc("first " * 50, page=3), self._rc("second " * 50, page=1, ctype="table"),
                  self._rc("third " * 500, page=2)]
        context = ContextBuilder.build(chunks, "q", max_chars=1000)
        assert context.index("first") < context.index("second")
        assert "third" not in context
        assert "Snippet 2" in context

    @pytest.mark.parametrize("text,expected", [
        ("Doanh thu quý 3 là bao nhiêu?", "vietnamese"),
        ("Doanh thu quy 3 nam 2024 la bao nhieu?", "vietnamese"),
        ("What is the theory behind the revenue growth?", "english"),
        ("Echo the anchor value for Sao Paulo", "english"),
    ])
    def test_language_detection(self, text, expected):
        from src.layers.retrieval_reasoning import detect_language
        assert detect_language(text) == expected

    def test_rerank_scores_on_unit_scale(self, mock_config):
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        reranker = CrossEncoderReranker(mock_config)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [9.0, -9.0, 0.0]
        chunks = [self._rc("a", score=0.2), self._rc("b", score=0.9), self._rc("c", score=0.5)]
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
            result = reranker.rerank("q", chunks, top_k=2)
        assert result[0].chunk.text == "a"
        assert all(0.0 <= c.final_score <= 1.0 for c in result)

    def test_reranker_falls_back_when_model_unavailable(self, mock_config):
        from src.layers.retrieval_reasoning import CrossEncoderReranker
        reranker = CrossEncoderReranker(mock_config)
        chunks = [self._rc(str(i), score=i / 10) for i in range(6)]
        with patch.object(reranker, "_cross_encoder_rerank", side_effect=OSError("offline")):
            result = reranker.rerank("q", chunks, top_k=3)
        assert [c.score for c in result] == [0.5, 0.4, 0.3]
        assert reranker._unavailable


# ── Benchmark Scoring Tests ───────────────────────────────────────────────────

class TestBenchmarkScoring:

    def test_numbers_normalized(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from benchmark import evaluate_answer
        assert evaluate_answer("Doanh thu quý 3 đạt 2.340 tỷ đồng.", [["2340"]]) == 1.0
        assert evaluate_answer("Doanh thu quy 3 la 2.550 ty.", [["2340"]]) == 0.0

    def test_restating_question_scores_zero(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from benchmark import SAMPLE_QA_PAIRS, evaluate_answer, validate_qa_pairs
        validate_qa_pairs(SAMPLE_QA_PAIRS)
        for qa in SAMPLE_QA_PAIRS:
            assert evaluate_answer(qa["question"], qa["facts"]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
