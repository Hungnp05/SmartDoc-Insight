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


class SmartChunker:
    def __init__(self, config):
        self.cfg = config.chunk

    def chunk_document(self, processed_doc) -> list:
        chunks = []

        for page in processed_doc.pages:
            page_meta_base = {
                "source_file": processed_doc.source_file,
                "page": page.page_num + 1,
                "page_summary": page.page_summary,
                "total_pages": processed_doc.total_pages,
            }

            for region in page.regions:
                if not region.content or len(region.content.strip()) < self.cfg.min_chunk_size:
                    continue

                region_meta = {
                    **page_meta_base,
                    "content_type": region.region_type,
                    "confidence": region.confidence,
                    "bbox": f"{region.bbox.x1},{region.bbox.y1},{region.bbox.x2},{region.bbox.y2}",
                }

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

    def _format_structured(self, region) -> str:
        labels = {"table": "BANG DU LIEU / TABLE", "figure": "MO TA BIEU DO / FIGURE"}
        label = labels.get(region.region_type, region.region_type.upper())
        return f"[{label} — Trang {region.page_num + 1}]\n\n{region.content}"

    def _split_text(self, text: str) -> list:
        size = self.cfg.text_chunk_size
        overlap = self.cfg.text_chunk_overlap

        if len(text) <= size:
            return [text]

        import re
        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

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

        return [c for c in chunks if len(c.strip()) >= self.cfg.min_chunk_size]

    @staticmethod
    def _make_chunk(text: str, metadata: dict) -> Chunk:
        return Chunk(chunk_id=str(uuid.uuid4()), text=text, metadata=metadata)


class EmbeddingHandler:
    def __init__(self, ollama_client, config):
        self.ollama = ollama_client
        self.model = config.ollama.embed_model

    def embed_texts(self, texts: list, batch_size: int = 32) -> list:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.ollama.embed(batch, model=self.model)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list:
        return self.ollama.embed([query], model=self.model)[0]


class KnowledgeBaseLayer:
    def __init__(self, config, ollama_client):
        self.cfg = config
        self.chunker = SmartChunker(config)
        self.embedder = EmbeddingHandler(ollama_client, config)
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        client = chromadb.PersistentClient(
            path=self.cfg.chroma.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=self.cfg.chroma.collection_name,
            metadata={"hnsw:space": self.cfg.chroma.distance_metric},
        )
        logger.info(f"ChromaDB ready: {self._collection.count()} chunks")

    def ingest_document(self, processed_doc, progress_callback=None) -> int:
        if progress_callback:
            progress_callback("Creating chunks...")
        chunks = self.chunker.chunk_document(processed_doc)

        self._remove_existing(processed_doc.source_file)

        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...")

        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        if progress_callback:
            progress_callback("Storing in ChromaDB...")
        self._store_chunks(chunks)

        type_counts = {}
        for c in chunks:
            t = c.metadata.get("content_type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"Ingested {len(chunks)} chunks — {type_counts}")
        return len(chunks)

    def _remove_existing(self, source_file: str):
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} old chunks for '{source_file}'")
        except Exception as e:
            logger.warning(f"Could not remove existing chunks: {e}")

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
            similarity = 1 - (dist / 2)
            chunk = Chunk(
                chunk_id=results["ids"][0][i],
                text=doc,
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

        return retrieved

    def list_documents(self) -> list:
        try:
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
        except Exception:
            return []

    def get_stats(self) -> dict:
        count = self._collection.count()
        docs = self.list_documents()
        return {"total_chunks": count, "total_documents": len(docs), "documents": docs}

    def clear_all(self):
        try:
            self._collection.delete(where={"source_file": {"$ne": ""}})
        except Exception:
            pass
        logger.info("Knowledge base cleared")