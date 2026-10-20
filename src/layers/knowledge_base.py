import re
import uuid
import logging
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict
    embedding: Optional[list] = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        return self.score if self.rerank_score is None else self.rerank_score


class SmartChunker:
    def __init__(self, config):
        self.cfg = config.chunk

    def chunk_document(self, processed_doc) -> list:
        chunks = []

        for page in processed_doc.pages:
            page_meta_base = {
                "source_file": processed_doc.source_file,
                "page": page.page_num + 1,
                "page_summary": page.page_summary or "",
                "total_pages": processed_doc.total_pages,
            }

            for region in page.regions:
                if not self._keep(region.content):
                    continue

                region_meta = {
                    **page_meta_base,
                    "content_type": region.region_type,
                    "confidence": region.confidence,
                    "bbox": f"{region.bbox.x1},{region.bbox.y1},{region.bbox.x2},{region.bbox.y2}",
                }

                # Tables and figures are never split here: the vision layer already produced
                # self-contained parts (oversized tables are split by rows with the header repeated).
                if region.region_type in ("table", "figure"):
                    chunks.append(self._make_chunk(
                        text=self._format_structured(region),
                        metadata=region_meta,
                    ))
                else:
                    for i, text in enumerate(self._split_text(region.content)):
                        chunks.append(self._make_chunk(
                            text=text,
                            metadata={**region_meta, "chunk_index": i},
                        ))

        logger.info(f"Chunked '{processed_doc.source_file}' -> {len(chunks)} chunks")
        return chunks

    def _keep(self, text: str) -> bool:
        """Drop fragments, but keep short snippets that carry a number (e.g. a one-line KPI)."""
        text = (text or "").strip()
        if len(text) >= self.cfg.min_chunk_size:
            return True
        return len(text) >= 10 and bool(re.search(r"\d", text))

    def _format_structured(self, region) -> str:
        labels = {"table": "BANG DU LIEU / TABLE", "figure": "MO TA BIEU DO / FIGURE"}
        label = labels.get(region.region_type, region.region_type.upper())
        return f"[{label} — Trang {region.page_num + 1}]\n\n{region.content}"

    def _split_text(self, text: str) -> list:
        size = self.cfg.text_chunk_size
        overlap = self.cfg.text_chunk_overlap

        if len(text) <= size:
            return [text]

        sentences = []
        for s in re.split(r'(?<=[.!?\n])\s+', text):
            s = s.strip()
            # A single "sentence" longer than the chunk size (e.g. text without punctuation)
            # is hard-split so no chunk exceeds the limit.
            while len(s) > size:
                sentences.append(s[:size])
                s = s[size:]
            if s:
                sentences.append(s)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > size and current:
                chunks.append(" ".join(current))
                overlap_buf, overlap_len = [], 0
                for s in reversed(current):
                    if overlap_len + len(s) <= overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len

            current.append(sentence)
            current_len += s_len

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if self._keep(c)]

    @staticmethod
    def _make_chunk(text: str, metadata: dict) -> Chunk:
        return Chunk(chunk_id=str(uuid.uuid4()), text=text, metadata=metadata)


class EmbeddingHandler:
    # nomic-embed-text is trained with task prefixes; without them retrieval quality drops.
    NOMIC_DOC_PREFIX = "search_document: "
    NOMIC_QUERY_PREFIX = "search_query: "

    def __init__(self, ollama_client, config):
        self.ollama = ollama_client
        self.model = config.ollama.embed_model
        self._use_prefix = "nomic" in self.model.lower()

    def embed_texts(self, texts: list, batch_size: int = 32) -> list:
        prefix = self.NOMIC_DOC_PREFIX if self._use_prefix else ""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            all_embeddings.extend(self.ollama.embed(batch, model=self.model))
        return all_embeddings

    def embed_query(self, query: str) -> list:
        prefix = self.NOMIC_QUERY_PREFIX if self._use_prefix else ""
        return self.ollama.embed([prefix + query], model=self.model)[0]


class KnowledgeBaseLayer:
    def __init__(self, config, ollama_client):
        self.cfg = config
        self.chunker = SmartChunker(config)
        self.embedder = EmbeddingHandler(ollama_client, config)
        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        self._client = chromadb.PersistentClient(
            path=self.cfg.chroma.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_collection()
        logger.info(f"ChromaDB ready: {self._collection.count()} chunks")

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self.cfg.chroma.collection_name,
            metadata={"hnsw:space": self.cfg.chroma.distance_metric},
        )

    def ingest_document(self, processed_doc, progress_callback=None) -> int:
        if progress_callback:
            progress_callback("Creating chunks...")
        chunks = self.chunker.chunk_document(processed_doc)
        if not chunks:
            logger.warning(f"No chunks extracted from '{processed_doc.source_file}'")
            return 0

        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...")

        # Embed before touching the store: if embedding fails, the previous version stays intact.
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        if progress_callback:
            progress_callback("Storing in ChromaDB...")
        self._remove_existing(processed_doc.source_file)
        self._store_chunks(chunks)

        type_counts = {}
        for c in chunks:
            t = c.metadata.get("content_type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"Ingested {len(chunks)} chunks — {type_counts}")
        return len(chunks)

    def has_document(self, source_file: str) -> bool:
        results = self._collection.get(where={"source_file": source_file}, limit=1)
        return bool(results["ids"])

    def _remove_existing(self, source_file: str):
        results = self._collection.get(where={"source_file": source_file})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(f"Removed {len(results['ids'])} old chunks for '{source_file}'")

    def _store_chunks(self, chunks: list, batch_size: int = 100):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[c.embedding for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 filter_source: Optional[str] = None) -> list:
        k = top_k or self.cfg.chroma.top_k
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_source:
            query_kwargs["where"] = {"source_file": filter_source}

        results = self._collection.query(**query_kwargs)

        retrieved = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # Chroma's cosine distance is 1 - cos_sim.
            similarity = max(0.0, 1.0 - dist)
            chunk = Chunk(
                chunk_id=results["ids"][0][i],
                text=doc,
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

        return retrieved

    def list_documents(self) -> list:
        results = self._collection.get(include=["metadatas"])
        docs = {}
        for meta in results["metadatas"]:
            src = meta.get("source_file", "unknown")
            if src not in docs:
                docs[src] = {
                    "source_file": src,
                    "total_pages": meta.get("total_pages", "?"),
                    "chunk_count": 0,
                }
            docs[src]["chunk_count"] += 1
        return list(docs.values())

    def get_stats(self) -> dict:
        count = self._collection.count()
        docs = self.list_documents()
        return {"total_chunks": count, "total_documents": len(docs), "documents": docs}

    def clear_all(self, batch_size: int = 1000):
        # Delete by id rather than dropping the collection: other open handles to the same
        # store (another pipeline instance or process) stay valid.
        while True:
            ids = self._collection.get(limit=batch_size, include=[])["ids"]
            if not ids:
                break
            self._collection.delete(ids=ids)
        logger.info("Knowledge base cleared")
