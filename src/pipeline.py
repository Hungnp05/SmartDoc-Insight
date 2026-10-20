"""
SmartDoc-Insight Pipeline Orchestrator
────────────────────────────────────────
Top-level API that wires all layers together.
Used by the Streamlit app, the benchmark script and debug.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartDocPipeline:
    """
    Unified pipeline orchestrator for SmartDoc-Insight.
    Wraps all layers into a single clean interface.

    Usage:
        pipeline = SmartDocPipeline()
        pipeline.ingest("report.pdf", progress_callback=my_callback)
        response = pipeline.ask("What was Q3 revenue?")
    """

    def __init__(self, config=None):
        from src.config import config as default_config
        from src.models.ollama_client import OllamaClient
        from src.layers.vision_processing import VisionProcessingLayer
        from src.layers.knowledge_base import KnowledgeBaseLayer
        from src.layers.retrieval_reasoning import RetrievalReasoningLayer

        self.config = config or default_config

        logger.info("Initializing SmartDoc-Insight pipeline...")
        self.ollama = OllamaClient(self.config)
        self.vision = VisionProcessingLayer(self.config, self.ollama)
        self.kb = KnowledgeBaseLayer(self.config, self.ollama)
        self.rag = RetrievalReasoningLayer(self.config, self.kb, self.ollama)
        logger.info("Pipeline ready.")

    def ingest(self, file_path: str | Path, progress_callback=None) -> dict:
        """
        Full ingestion pipeline: vision → chunk → embed → store.

        Args:
            file_path: Path to PDF or image file
            progress_callback: Optional callable(stage: str, fraction: float | None)

        Returns:
            dict with ingestion stats
        """
        file_path = Path(file_path)

        def report(msg, fraction=None):
            if progress_callback:
                progress_callback(msg, fraction)

        def _vision_progress(idx, total, msg):
            # Vision covers the first 80% of the progress bar.
            report(f"[Vision] {msg} ({idx + 1}/{total})", 0.8 * idx / max(total, 1))

        def _kb_progress(msg):
            report(f"[KB] {msg}", 0.9)

        report(f"Starting vision processing for {file_path.name}...", 0.0)
        processed_doc = self.vision.process_document(file_path, _vision_progress)
        chunk_count = self.kb.ingest_document(processed_doc, _kb_progress)
        report("Done", 1.0)

        region_summary = {}
        for region in processed_doc.all_regions():
            region_summary[region.region_type] = region_summary.get(region.region_type, 0) + 1

        return {
            "source_file": file_path.name,
            "total_pages": processed_doc.total_pages,
            "total_chunks": chunk_count,
            "regions": region_summary,
        }

    def ask(self, question: str, source_filter: str = None, top_k: int = None, rerank_top_k: int = None):
        """Query the knowledge base and generate an answer (RAGResponse)."""
        return self.rag.query(question, filter_source=source_filter, top_k=top_k, rerank_top_k=rerank_top_k)

    def ask_stream(self, question: str, source_filter: str = None, top_k: int = None, rerank_top_k: int = None):
        """Streaming version of ask(): StreamingRAGResponse with `.sources` and a `.tokens` generator."""
        return self.rag.query_stream(question, filter_source=source_filter, top_k=top_k, rerank_top_k=rerank_top_k)

    def stats(self) -> dict:
        """Return knowledge base stats."""
        return self.kb.get_stats()

    def list_documents(self) -> list:
        """List all indexed documents."""
        return self.kb.list_documents()

    def clear(self):
        """Delete every indexed chunk."""
        self.kb.clear_all()

    def check_models(self) -> dict:
        """Check which Ollama models are available."""
        available = self.ollama.list_models()
        return {
            "llm": self.ollama.is_model_available(self.config.ollama.llm_model, available),
            "vision": self.ollama.is_model_available(self.config.ollama.vision_model, available),
            "embed": self.ollama.is_model_available(self.config.ollama.embed_model, available),
            "available_models": available,
        }
