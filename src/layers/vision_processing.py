"""
Layer A: Vision Processing
──────────────────────────
Turns a PDF or image into typed regions (text / table / figure).

PDF pages:
  - Tables: PyMuPDF `find_tables()` → Markdown with the header row and the caption above it.
  - Figures: vector drawing clusters (`cluster_drawings()`) and raster images are rendered
    and described by the vision model, together with the text labels inside the chart.
  - Text: remaining text blocks, grouped in reading order.
  - Pages without a text layer (scans) are OCR'd.
Image files: OCR for the text, plus a vision-model description of the whole image.

Model calls are batched by phase (layout → all figure descriptions → all page summaries)
so Ollama swaps the vision and text models in VRAM once instead of once per page.
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import fitz

logger = logging.getLogger(__name__)

TEXT_REGION_MAX_CHARS = 700


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1:self.y2, self.x1:self.x2]

    @classmethod
    def from_rect(cls, rect) -> "BoundingBox":
        return cls(round(rect[0]), round(rect[1]), round(rect[2]), round(rect[3]))


@dataclass
class DocumentRegion:
    region_type: str
    bbox: BoundingBox
    confidence: float
    page_num: int
    content: str = ""
    raw_image: Optional[np.ndarray] = None
    caption: str = ""
    labels: str = ""


@dataclass
class ProcessedPage:
    page_num: int
    page_summary: str = ""
    regions: list = field(default_factory=list)
    full_text: str = ""


@dataclass
class ProcessedDocument:
    source_file: str
    total_pages: int
    pages: list = field(default_factory=list)

    def all_regions(self):
        return [r for page in self.pages for r in page.regions]


class OCREngine:
    """Lazy PaddleOCR wrapper supporting both the 2.x (`ocr`) and 3.x (`predict`) APIs.
    Disables itself after the first failure so a missing/broken install costs nothing."""

    def __init__(self, lang: str, enabled: bool = True):
        self.lang = lang
        self.enabled = enabled
        self._engine = None

    def _load(self):
        from paddleocr import PaddleOCR
        try:
            return PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (TypeError, ValueError):
            return PaddleOCR(lang=self.lang, use_angle_cls=True, show_log=False)

    def read_lines(self, image: np.ndarray) -> Optional[list]:
        """Return recognized text lines in reading order, or None if OCR is unavailable."""
        if not self.enabled:
            return None
        try:
            if self._engine is None:
                self._engine = self._load()
            if hasattr(self._engine, "predict"):
                lines = []
                for res in self._engine.predict(image):
                    lines.extend(res["rec_texts"])
                return [line for line in lines if line.strip()]
            result = self._engine.ocr(image, cls=True) or []
            return [line[1][0] for page in result if page for line in page if line[1][0].strip()]
        except Exception as e:
            logger.warning(f"OCR unavailable, falling back to the vision model: {e}")
            self.enabled = False
            return None


class VisionProcessingLayer:
    def __init__(self, config, ollama_client):
        self.cfg = config.vision
        self.chunk_cfg = config.chunk
        self.ollama = ollama_client
        self.ocr = OCREngine(self.cfg.ocr_lang, self.cfg.ocr_enabled)

    def process_document(self, file_path, progress_callback=None) -> ProcessedDocument:
        file_path = Path(file_path)
        logger.info(f"Processing: {file_path.name}")

        def progress(idx, total, msg):
            if progress_callback:
                progress_callback(idx, total, msg)

        # Phase 1: layout (no model calls except OCR)
        if file_path.suffix.lower() == ".pdf":
            with fitz.open(str(file_path)) as pdf:
                total = len(pdf)
                pages = []
                for page in pdf:
                    progress(page.number, total, f"Analyzing layout of page {page.number + 1}...")
                    pages.append(self._analyze_pdf_page(page))
        else:
            progress(0, 1, "Analyzing image...")
            pages = [self._analyze_image(self._load_image(file_path))]

        doc = ProcessedDocument(source_file=file_path.name, total_pages=len(pages), pages=pages)

        # Phase 2: vision model for every figure
        figures = [r for r in doc.all_regions() if r.region_type == "figure" and not r.content]
        for i, region in enumerate(figures):
            progress(i, len(figures), f"Describing figure on page {region.page_num + 1} with the vision model...")
            region.content = self._describe_figure(region)

        # Phase 3: text model for page summaries
        for i, page in enumerate(doc.pages):
            page.regions = [r for r in page.regions if r.content and len(r.content.strip()) > 10]
            page.full_text = self._build_page_text(page.regions)
            if self.cfg.page_summary_enabled:
                progress(i, len(doc.pages), f"Summarizing page {page.page_num + 1}...")
                page.page_summary = self._generate_page_summary(page.full_text)

        region_types = {}
        for r in doc.all_regions():
            region_types[r.region_type] = region_types.get(r.region_type, 0) + 1
        logger.info(f"Done: {len(doc.pages)} pages, {len(doc.all_regions())} regions — {region_types}")
        return doc

    # PDF pages

    def _analyze_pdf_page(self, page) -> ProcessedPage:
        page_num = page.number
        processed = ProcessedPage(page_num=page_num)

        text_blocks = []
        image_rects = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type", 0) == 1:
                image_rects.append(fitz.Rect(block["bbox"]))
                continue
            lines = []
            for line in block.get("lines", []):
                line_text = " ".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if line_text:
                    lines.append(line_text)
            if lines:
                text_blocks.append({"text": "\n".join(lines), "rect": fitz.Rect(block["bbox"])})

        tables = self._find_tables(page)
        table_rects = [rect for rect, _ in tables]

        if not text_blocks and not tables:
            processed.regions = self._analyze_scanned_page(page)
            return processed

        figure_cores = self._find_figure_rects(page, image_rects, table_rects)
        # Axis labels and legends sit just outside the drawing cluster; pad the figure area
        # so they are rendered with the chart and attached to it as labels.
        pad = self.cfg.figure_padding
        figure_rects = [(r + (-pad, -pad, pad, pad)) & page.rect for r in figure_cores]

        body_blocks = []
        figure_labels = {i: [] for i in range(len(figure_rects))}
        for tb in text_blocks:
            center = (tb["rect"].tl + tb["rect"].br) / 2
            if any(center in r for r in table_rects):
                continue
            fig_idx = next((i for i, r in enumerate(figure_rects) if center in r), None)
            if fig_idx is not None and self._is_label_block(tb["text"]):
                figure_labels[fig_idx].append(tb["text"].replace("\n", " "))
                continue
            body_blocks.append(tb)

        regions = []
        for rect, markdown in tables:
            caption = self._find_caption(rect, body_blocks, below=False)
            for part in self._split_table(markdown):
                content = f"{caption}\n\n{part}" if caption else part
                regions.append(DocumentRegion(
                    region_type="table",
                    bbox=BoundingBox.from_rect(rect),
                    confidence=0.95,
                    page_num=page_num,
                    content=content,
                    caption=caption,
                ))

        for i, rect in enumerate(figure_rects):
            regions.append(DocumentRegion(
                region_type="figure",
                bbox=BoundingBox.from_rect(rect),
                confidence=0.9,
                page_num=page_num,
                raw_image=self._render_clip(page, rect),
                caption=self._find_caption(figure_cores[i], body_blocks, below=None),
                labels=" | ".join(figure_labels[i]),
            ))

        regions.extend(self._group_text_blocks(body_blocks, page_num))
        regions.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))
        processed.regions = regions
        return processed

    def _find_tables(self, page) -> list:
        """Return [(rect, markdown)] for every table PyMuPDF detects on the page."""
        try:
            found = page.find_tables()
        except Exception as e:
            logger.warning(f"Table detection failed on page {page.number + 1}: {e}")
            return []
        tables = []
        for table in found.tables:
            if table.row_count < 2 or table.col_count < 2:
                continue
            markdown = self._clean_table_markdown(table.to_markdown())
            if markdown:
                tables.append((fitz.Rect(table.bbox), markdown))
        return tables

    @staticmethod
    def _clean_table_markdown(markdown: str) -> str:
        markdown = markdown.replace("**", "").replace("<br>", " ")
        return "\n".join(line.strip() for line in markdown.strip().splitlines() if line.strip())

    def _split_table(self, markdown: str) -> list:
        """Split an oversized table by rows, repeating the header (first 2 lines) in every part."""
        max_chars = self.chunk_cfg.table_max_chars
        if len(markdown) <= max_chars:
            return [markdown]
        lines = markdown.splitlines()
        header, rows = lines[:2], lines[2:]
        header_len = sum(len(line) + 1 for line in header)
        parts, current, current_len = [], [], header_len
        for row in rows:
            if current and current_len + len(row) + 1 > max_chars:
                parts.append("\n".join(header + current))
                current, current_len = [], header_len
            current.append(row)
            current_len += len(row) + 1
        if current:
            parts.append("\n".join(header + current))
        return parts

    def _find_figure_rects(self, page, image_rects, table_rects) -> list:
        candidates = list(image_rects)
        try:
            candidates.extend(fitz.Rect(r) for r in page.cluster_drawings())
        except Exception as e:
            logger.debug(f"cluster_drawings unavailable on page {page.number + 1}: {e}")

        rects = []
        for rect in candidates:
            rect = rect & page.rect
            if rect.width < self.cfg.min_figure_width or rect.height < self.cfg.min_figure_height:
                continue
            # Table borders are drawings too; skip clusters that are mostly a table.
            if any(abs(rect & t) > 0.5 * abs(rect) for t in table_rects):
                continue
            rects.append(rect)

        merged = []
        for rect in sorted(rects, key=lambda r: abs(r), reverse=True):
            for i, existing in enumerate(merged):
                if abs(rect & existing) > 0.3 * min(abs(rect), abs(existing)):
                    merged[i] = existing | rect
                    break
            else:
                merged.append(rect)
        return sorted(merged, key=lambda r: (r.y0, r.x0))

    def _is_label_block(self, text: str) -> bool:
        """Short lines (axis ticks, legend entries) rather than a paragraph."""
        return all(len(line) <= self.cfg.max_label_chars for line in text.splitlines())

    def _find_caption(self, rect, blocks, below: Optional[bool]) -> str:
        """Nearest text block just above (below=False), just below (True) or both (None) of rect."""
        margin = self.cfg.caption_margin
        nearest = {"above": None, "below": None}
        for tb in blocks:
            r = tb["rect"]
            if r.x1 < rect.x0 or r.x0 > rect.x1:
                continue
            for side, gap in (("above", rect.y0 - r.y1), ("below", r.y0 - rect.y1)):
                if 0 <= gap <= margin and (nearest[side] is None or gap < nearest[side][0]):
                    nearest[side] = (gap, tb["text"])
        sides = {False: ["above"], True: ["below"], None: ["above", "below"]}[below]
        return "\n".join(nearest[s][1].replace("\n", " ") for s in sides if nearest[s])

    def _render_clip(self, page, rect) -> np.ndarray:
        pix = page.get_pixmap(clip=rect, dpi=self.cfg.pdf_dpi, alpha=False)
        return self._pixmap_to_array(pix)

    def _analyze_scanned_page(self, page) -> list:
        pix = page.get_pixmap(dpi=self.cfg.pdf_dpi, alpha=False)
        return self._analyze_image(self._pixmap_to_array(pix), page.number, describe_whole=False).regions

    @staticmethod
    def _pixmap_to_array(pix) -> np.ndarray:
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        return img[:, : pix.width * pix.n].reshape(pix.height, pix.width, pix.n).copy()

    # Images

    def _load_image(self, path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def _analyze_image(self, img: np.ndarray, page_num: int = 0, describe_whole: bool = True) -> ProcessedPage:
        """OCR the text; describe the whole image with the vision model when it is an image
        upload (likely a chart/figure) or when OCR is unavailable."""
        processed = ProcessedPage(page_num=page_num)
        h, w = img.shape[:2]
        lines = self.ocr.read_lines(img)
        if lines:
            processed.regions.extend(self._split_text_into_regions("\n".join(lines), page_num, w, h))
        if describe_whole or lines is None:
            processed.regions.append(DocumentRegion(
                region_type="figure",
                bbox=BoundingBox(0, 0, w, h),
                confidence=0.8,
                page_num=page_num,
                raw_image=img,
                labels=" | ".join(lines or [])[:1500],
            ))
        return processed

    # Text grouping

    def _group_text_blocks(self, blocks, page_num) -> list:
        regions = []
        current, current_len = [], 0

        def flush():
            if current:
                rect = fitz.Rect(current[0]["rect"])
                for tb in current[1:]:
                    rect |= tb["rect"]
                regions.append(DocumentRegion(
                    region_type="text",
                    bbox=BoundingBox.from_rect(rect),
                    confidence=1.0,
                    page_num=page_num,
                    content="\n".join(tb["text"] for tb in current),
                ))
                current.clear()

        for tb in sorted(blocks, key=lambda b: (b["rect"].y0, b["rect"].x0)):
            if current_len + len(tb["text"]) > TEXT_REGION_MAX_CHARS and current:
                flush()
                current_len = 0
            current.append(tb)
            current_len += len(tb["text"])
        flush()
        return regions

    def _split_text_into_regions(self, text, page_num, w, h) -> list:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        chunks, current, current_len = [], [], 0
        for line in lines:
            current.append(line)
            current_len += len(line)
            if current_len > TEXT_REGION_MAX_CHARS:
                chunks.append("\n".join(current))
                current, current_len = [], 0
        if current:
            chunks.append("\n".join(current))
        chunk_h = h // max(len(chunks), 1)
        return [
            DocumentRegion(
                region_type="text",
                bbox=BoundingBox(0, i * chunk_h, w, min((i + 1) * chunk_h, h)),
                confidence=0.85,
                page_num=page_num,
                content=chunk,
            )
            for i, chunk in enumerate(chunks)
        ]

    # Model calls

    def _describe_figure(self, region: DocumentRegion) -> str:
        from src.config import CHART_DESCRIPTION_PROMPT, CHART_LABELS_HINT
        prompt = CHART_DESCRIPTION_PROMPT
        if region.caption or region.labels:
            prompt += CHART_LABELS_HINT.format(labels="\n".join(filter(None, [region.caption, region.labels])))

        description = ""
        try:
            description = self.ollama.vision_query(
                prompt=prompt,
                image_base64=self._image_to_base64(region.raw_image),
            )
        except Exception as e:
            logger.error(f"Figure description failed on page {region.page_num + 1}: {e}")

        # Keep caption and labels even when the vision model fails, so the chart stays searchable.
        parts = [p for p in (region.caption, description) if p]
        if region.labels:
            parts.append(f"Labels: {region.labels}")
        return "\n".join(parts)

    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_page_text(self, regions) -> str:
        parts = []
        for r in regions:
            if r.content:
                prefix = "# " if r.region_type == "title" else ""
                parts.append(f"{prefix}{r.content}")
        return "\n\n".join(parts)

    def _generate_page_summary(self, page_text: str) -> str:
        text_sample = page_text[:1500]
        if not text_sample.strip():
            return "Empty page"
        try:
            summary = self.ollama.query(
                prompt=f"Summarize the following document page in 1-2 concise sentences.\n\nContent:\n{text_sample}\n\nSummary:",
                system="You are a document summarization assistant. Reply in 1-2 concise sentences only.",
            )
            return summary.strip()
        except Exception as e:
            logger.warning(f"Page summary failed: {e}")
            return ""
