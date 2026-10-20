import math
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

# Rough characters-per-token for budgeting the prompt (Vietnamese/English mix, Llama 3 tokenizer).
CHARS_PER_TOKEN = 2.5
# Tokens reserved for the system prompt, template and question.
PROMPT_OVERHEAD_TOKENS = 700

VIETNAMESE_DIACRITICS = set(
    "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
)
# Unaccented Vietnamese words that are not English words, matched as whole tokens.
VIETNAMESE_WORDS = {
    "la", "va", "cua", "co", "duoc", "khong", "nay", "cho", "voi", "trong", "mot", "cac",
    "toi", "ban", "hay", "theo", "nhu", "khi", "ve", "tu", "tai", "bao", "nhieu", "gi",
    "nao", "sao", "doanh", "thu", "bang", "bieu", "quy", "nam", "tang", "giam", "tong",
    "chi", "phi", "truong", "nhan", "vien", "loi", "nhuan", "ke", "hoach", "rui", "ro",
    "xu", "huong", "phan", "tram", "chiem", "mang", "kinh",
}
ENGLISH_WORDS = {
    "the", "what", "which", "how", "is", "are", "was", "were", "of", "in", "and", "to",
    "for", "does", "did", "a", "an", "revenue", "growth", "from", "by", "with", "much", "many",
}


@dataclass
class Source:
    content: str
    content_type: str
    page: int
    source_file: str
    score: float

    @property
    def type_label(self):
        return {"table": "TABLE", "figure": "FIGURE", "title": "TITLE", "text": "TEXT"}.get(
            self.content_type, "UNKNOWN"
        )

    @property
    def type_emoji(self):
        return {"table": "📊", "figure": "📈"}.get(self.content_type, "📝")

    @property
    def display_label(self):
        return f"[{self.type_label}] Page {self.page}"


@dataclass
class RAGResponse:
    answer: str
    sources: list
    query: str
    retrieved_count: int
    reranked_count: int


@dataclass
class StreamingRAGResponse:
    """Sources are known before generation starts; `tokens` yields the answer."""
    tokens: Iterator[str]
    sources: list
    query: str
    retrieved_count: int
    reranked_count: int
    debug: dict = field(default_factory=dict)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, config):
        self.model_name = config.reranker.model_name
        self.use_gpu = config.reranker.use_gpu
        self.ce_weight = config.reranker.ce_weight
        self._model = None
        self._unavailable = False

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        if not chunks:
            return []
        if len(chunks) <= top_k or self._unavailable:
            return self._score_based_rerank(chunks, top_k)
        try:
            return self._cross_encoder_rerank(query, chunks, top_k)
        except Exception as e:
            # Missing package or model download failure (offline): fall back once, don't retry each query.
            logger.warning(f"Cross-encoder unavailable ({e}); using vector similarity ranking")
            self._unavailable = True
            return self._score_based_rerank(chunks, top_k)

    def _cross_encoder_rerank(self, query: str, chunks: list, top_k: int) -> list:
        from sentence_transformers import CrossEncoder
        if self._model is None:
            device = "cuda" if self.use_gpu else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Cross-encoder loaded: {self.model_name}")

        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        # Cross-encoder outputs unbounded logits; map them to 0..1 so they can be mixed
        # with the cosine similarity on the same scale.
        for chunk, ce_score in zip(chunks, scores):
            chunk.rerank_score = (
                self.ce_weight * _sigmoid(float(ce_score)) + (1 - self.ce_weight) * chunk.score
            )

        return sorted(chunks, key=lambda c: c.final_score, reverse=True)[:top_k]

    @staticmethod
    def _score_based_rerank(chunks: list, top_k: int) -> list:
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


class ContextBuilder:
    @staticmethod
    def build(chunks: list, query: str, max_chars: Optional[int] = None) -> str:
        """Build the prompt context in rerank order (most relevant first), skipping duplicates
        and stopping before `max_chars` so the prompt fits the model's context window."""
        if not chunks:
            return "No relevant information found in the document."

        parts = []
        seen = set()
        total = 0

        for rc in chunks:
            meta = rc.chunk.metadata
            page = meta.get("page", "?")
            content_type = meta.get("content_type", "text")
            page_summary = meta.get("page_summary", "")
            source = meta.get("source_file", "")

            dedup_key = f"{page}:{rc.chunk.text[:80]}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            header = (
                f"--- Snippet {len(parts) + 1} | Page {page} | Type: {content_type.upper()} "
                f"| Source: {source} ---"
            )
            if page_summary:
                header += f"\n[Page summary: {page_summary}]"

            if content_type == "table":
                header += "\n[READING INSTRUCTION: Read this table ROW BY ROW. Each row is one record. Do not mix values across rows or columns.]"

            part = f"{header}\n{rc.chunk.text}"
            if max_chars is not None and parts and total + len(part) > max_chars:
                logger.info(f"Context budget reached; dropped snippets after #{len(parts)}")
                break
            if max_chars is not None and not parts and len(part) > max_chars:
                part = part[:max_chars]
            parts.append(part)
            total += len(part) + 2

        return "\n\n".join(parts)

    @staticmethod
    def chunks_to_sources(chunks: list) -> list:
        sources = []
        for rc in chunks:
            meta = rc.chunk.metadata
            sources.append(Source(
                content=rc.chunk.text[:500] + ("..." if len(rc.chunk.text) > 500 else ""),
                content_type=meta.get("content_type", "text"),
                page=meta.get("page", 0),
                source_file=meta.get("source_file", "unknown"),
                score=rc.score,
            ))
        return sources


def detect_language(text: str) -> str:
    if sum(1 for c in text.lower() if c in VIETNAMESE_DIACRITICS) >= 2:
        return "vietnamese"
    tokens = re.findall(r"[a-z]+", text.lower())
    vi_hits = sum(1 for t in tokens if t in VIETNAMESE_WORDS)
    en_hits = sum(1 for t in tokens if t in ENGLISH_WORDS)
    return "vietnamese" if vi_hits >= 2 and vi_hits > en_hits else "english"


class RetrievalReasoningLayer:
    def __init__(self, config, knowledge_base, ollama_client):
        self.cfg = config
        self.kb = knowledge_base
        self.ollama = ollama_client
        self.reranker = CrossEncoderReranker(config)
        self.context_builder = ContextBuilder()

    @property
    def context_char_budget(self) -> int:
        tokens = self.cfg.ollama.num_ctx - self.cfg.ollama.max_tokens - PROMPT_OVERHEAD_TOKENS
        return max(1000, int(tokens * CHARS_PER_TOKEN))

    def _retrieve(self, question, filter_source, top_k, rerank_top_k):
        raw_chunks = self.kb.retrieve(
            query=question,
            top_k=top_k or self.cfg.chroma.top_k,
            filter_source=filter_source,
        )
        reranked_chunks = self.reranker.rerank(
            query=question,
            chunks=raw_chunks,
            top_k=rerank_top_k or self.cfg.chroma.rerank_top_k,
        )
        logger.info(f"Retrieved {len(raw_chunks)} chunks, reranked to {len(reranked_chunks)}")
        return raw_chunks, reranked_chunks

    def _build_prompt(self, question: str, chunks: list) -> str:
        from src.config import RAG_PROMPT_TEMPLATE
        context = self.context_builder.build(chunks, question, max_chars=self.context_char_budget)
        return RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            language_instruction=self._get_language_instruction(detect_language(question)),
        )

    def query(self, question: str, filter_source: Optional[str] = None,
              top_k: Optional[int] = None, rerank_top_k: Optional[int] = None) -> RAGResponse:
        from src.config import SYSTEM_PROMPT
        raw_chunks, reranked_chunks = self._retrieve(question, filter_source, top_k, rerank_top_k)
        answer = self.ollama.query(prompt=self._build_prompt(question, reranked_chunks), system=SYSTEM_PROMPT)
        return RAGResponse(
            answer=answer,
            sources=self.context_builder.chunks_to_sources(reranked_chunks),
            query=question,
            retrieved_count=len(raw_chunks),
            reranked_count=len(reranked_chunks),
        )

    def query_stream(self, question: str, filter_source: Optional[str] = None,
                     top_k: Optional[int] = None, rerank_top_k: Optional[int] = None) -> StreamingRAGResponse:
        from src.config import SYSTEM_PROMPT
        raw_chunks, reranked_chunks = self._retrieve(question, filter_source, top_k, rerank_top_k)
        prompt = self._build_prompt(question, reranked_chunks)
        return StreamingRAGResponse(
            tokens=self.ollama.stream(prompt=prompt, system=SYSTEM_PROMPT),
            sources=self.context_builder.chunks_to_sources(reranked_chunks),
            query=question,
            retrieved_count=len(raw_chunks),
            reranked_count=len(reranked_chunks),
        )

    def _detect_language(self, text: str) -> str:
        return detect_language(text)

    def _get_language_instruction(self, language: str) -> str:
        if language == "vietnamese":
            return (
                "Câu hỏi được viết bằng TIẾNG VIỆT. "
                "Bạn BẮT BUỘC phải trả lời HOÀN TOÀN bằng tiếng Việt. "
                "Không được dùng câu tiếng Anh trong phần trả lời và không thêm bản dịch. "
                "Tên riêng và số liệu giữ nguyên, nhưng tất cả văn bản giải thích phải bằng tiếng Việt."
            )
        return (
            "The question is in ENGLISH. "
            "You MUST answer entirely in English."
        )

    def get_retrieval_debug(self, question: str, filter_source: Optional[str] = None,
                            top_k: Optional[int] = None, rerank_top_k: Optional[int] = None) -> dict:
        raw, reranked = self._retrieve(question, filter_source, top_k, rerank_top_k)
        return {
            "query": question,
            "initial_retrieval": [
                {
                    "text": c.chunk.text[:200],
                    "score": round(c.score, 4),
                    "type": c.chunk.metadata.get("content_type"),
                    "page": c.chunk.metadata.get("page"),
                }
                for c in raw
            ],
            "after_reranking": [
                {
                    "text": c.chunk.text[:200],
                    "original_score": round(c.score, 4),
                    "rerank_score": round(c.final_score, 4),
                    "type": c.chunk.metadata.get("content_type"),
                    "page": c.chunk.metadata.get("page"),
                }
                for c in reranked
            ],
        }
