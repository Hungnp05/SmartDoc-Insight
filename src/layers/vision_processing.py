import io
import re
import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import fitz

logger = logging.getLogger(__name__)


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


@dataclass
class DocumentRegion:
    region_type: str
    bbox: BoundingBox
    confidence: float
    page_num: int
    content: str = ""
    raw_image: Optional[np.ndarray] = None


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


class VisionProcessingLayer:
    def __init__(self, config, ollama_client):
        self.cfg = config.vision
        self.ollama = ollama_client

    def process_document(self, file_path, progress_callback=None) -> ProcessedDocument:
        file_path = Path(file_path)
        logger.info(f"Processing: {file_path.name}")

        if file_path.suffix.lower() == ".pdf":
            pages_data = self._pdf_to_pages(file_path)
        else:
            img = self._load_image(file_path)
            pages_data = [(0, img, "", [])]

        doc = ProcessedDocument(source_file=file_path.name, total_pages=len(pages_data))

        for idx, page_tuple in enumerate(pages_data):
            page_num, page_img, page_text, page_blocks = page_tuple
            if progress_callback:
                progress_callback(idx, len(pages_data), f"Analyzing page {page_num + 1}...")
            processed_page = self._process_page(page_img, page_num, page_text, page_blocks)
            doc.pages.append(processed_page)

        total = len(doc.all_regions())
        region_types = {}
        for r in doc.all_regions():
            region_types[r.region_type] = region_types.get(r.region_type, 0) + 1
        logger.info(f"Done: {len(doc.pages)} pages, {total} regions — {region_types}")
        return doc

    def _pdf_to_pages(self, pdf_path: Path):
        dpi = self.cfg.pdf_dpi
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        result = []
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img_array.reshape(pix.height, pix.width, 3)
                text = page.get_text("text")
                blocks = page.get_text("dict")["blocks"]
                result.append((page_num, img, text, blocks))
        return result

    def _load_image(self, path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def _process_page(self, page_img, page_num, page_text="", page_blocks=None):
        processed = ProcessedPage(page_num=page_num)
        regions = self._extract_regions(page_blocks, page_img, page_num, page_text)
        for region in regions:
            if region.content and len(region.content.strip()) > 10:
                processed.regions.append(region)
        processed.full_text = self._build_page_text(processed.regions)
        processed.page_summary = self._generate_page_summary(processed.regions)
        return processed

    def _extract_regions(self, blocks, page_img, page_num, page_text):
        h, w = page_img.shape[:2]

        if not blocks and not page_text:
            return []

        if not blocks:
            return self._split_text_into_regions(page_text, page_img, page_num)

        text_blocks = []
        image_blocks = []

        for block in blocks:
            btype = block.get("type", 0)
            if btype == 1:
                image_blocks.append(block)
            elif btype == 0:
                lines_text = []
                for line in block.get("lines", []):
                    line_text = " ".join(
                        span.get("text", "") for span in line.get("spans", [])
                    ).strip()
                    if line_text:
                        lines_text.append(line_text)
                if lines_text:
                    text_blocks.append({
                        "text": "\n".join(lines_text),
                        "bbox": block.get("bbox", [0, 0, w, h]),
                    })

        regions = []
        regions.extend(self._group_text_blocks(text_blocks, page_num, w, h))
        regions.extend(self._process_image_blocks(image_blocks, page_img, page_num, w, h))
        regions.sort(key=lambda r: (r.bbox.y1, r.bbox.x1))
        return regions

    def _group_text_blocks(self, text_blocks, page_num, w, h):
        regions = []
        current_text = []
        current_len = 0

        def flush_text():
            if current_text:
                regions.append(DocumentRegion(
                    region_type="text",
                    bbox=BoundingBox(0, 0, w, h),
                    confidence=1.0,
                    page_num=page_num,
                    content="\n".join(current_text),
                ))
                current_text.clear()

        for tb in text_blocks:
            block_text = tb["text"].strip()
            if not block_text:
                continue

            if self._is_table_block(block_text):
                flush_text()
                current_len = 0
                last_table = next(
                    (r for r in reversed(regions) if r.region_type == "table" and r.page_num == page_num),
                    None
                )
                if last_table and self._is_continuation(last_table.content, block_text):
                    last_table.content += "\n" + block_text
                else:
                    markdown = self._rows_to_markdown(block_text)
                    regions.append(DocumentRegion(
                        region_type="table",
                        bbox=BoundingBox(0, 0, w, h),
                        confidence=0.9,
                        page_num=page_num,
                        content=markdown,
                    ))
                continue

            if current_len + len(block_text) > 700 and current_text:
                flush_text()
                current_len = 0

            current_text.append(block_text)
            current_len += len(block_text)

        flush_text()
        return regions

    def _process_image_blocks(self, image_blocks, page_img, page_num, w, h):
        regions = []
        for block in image_blocks:
            bbox = block.get("bbox", [0, 0, w, h])
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            img_x0 = max(0, min(int(x0 * w / 595), w))
            img_y0 = max(0, min(int(y0 * h / 842), h))
            img_x1 = max(0, min(int(x1 * w / 595), w))
            img_y1 = max(0, min(int(y1 * h / 842), h))
            if img_x1 - img_x0 < 20 or img_y1 - img_y0 < 20:
                continue
            bounding = BoundingBox(img_x0, img_y0, img_x1, img_y1)
            crop = bounding.crop(page_img)
            description = self._describe_figure_crop(crop, page_num)
            if description:
                regions.append(DocumentRegion(
                    region_type="figure",
                    bbox=bounding,
                    confidence=0.9,
                    page_num=page_num,
                    content=description,
                    raw_image=crop,
                ))
        return regions

    def _is_table_block(self, text: str) -> bool:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            return False

        if sum(1 for l in lines if l.count("|") >= 2) >= 2:
            return True

        if sum(1 for l in lines if "\t" in l and l.count("\t") >= 2) >= 2:
            return True

        numeric_lines = sum(
            1 for l in lines
            if sum(1 for t in re.split(r"[\s\t]+", l) if re.search(r"\d", t)) >= 3
        )
        if numeric_lines >= max(2, len(lines) * 0.5):
            return True

        if sum(1 for l in lines if "%" in l) >= 2 and len(lines) >= 3:
            return True

        return False

    def _is_continuation(self, existing: str, new_block: str) -> bool:
        existing_lines = existing.strip().split("\n")
        new_lines = new_block.strip().split("\n")
        if not existing_lines or not new_lines:
            return False
        last = existing_lines[-1]
        first = new_lines[0]
        last_cols = last.count("|") if "|" in last else last.count("\t")
        new_cols = first.count("|") if "|" in first else first.count("\t")
        return abs(last_cols - new_cols) <= 1

    def _rows_to_markdown(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return text

        if any("|" in l for l in lines):
            return text

        result = []
        for i, line in enumerate(lines):
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                cells = [c.strip() for c in re.split(r"  {2,}", line) if c.strip()]
            if not cells:
                cells = [line]
            result.append("| " + " | ".join(cells) + " |")
            if i == 0:
                result.append("|" + "|".join([" --- " for _ in cells]) + "|")

        return "\n".join(result)

    def _split_text_into_regions(self, page_text, page_img, page_num):
        h, w = page_img.shape[:2]
        if not page_text or not page_text.strip():
            return []
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        chunks, current, current_len = [], [], 0
        for line in lines:
            current.append(line)
            current_len += len(line)
            if current_len > 700:
                chunks.append("\n".join(current))
                current, current_len = [], 0
        if current:
            chunks.append("\n".join(current))
        chunk_h = h // max(len(chunks), 1)
        return [
            DocumentRegion(
                region_type="text",
                bbox=BoundingBox(0, i * chunk_h, w, min((i + 1) * chunk_h, h)),
                confidence=1.0,
                page_num=page_num,
                content=chunk,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _describe_figure_crop(self, figure_img: np.ndarray, page_num: int) -> str:
        try:
            from src.config import CHART_DESCRIPTION_PROMPT
            img_b64 = self._image_to_base64(figure_img)
            description = self.ollama.vision_query(
                prompt=CHART_DESCRIPTION_PROMPT,
                image_base64=img_b64,
            )
            return f"[FIGURE - Page {page_num + 1}]\n{description}"
        except Exception as e:
            logger.error(f"Figure description failed: {e}")
            return ""

    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_page_text(self, regions) -> str:
        parts = []
        for r in regions:
            if r.content:
                prefix = "# " if r.region_type == "title" else ""
                parts.append(f"{prefix}{r.content}")
        return "\n\n".join(parts)

    def _generate_page_summary(self, regions) -> str:
        text_sample = self._build_page_text(regions)[:1500]
        if not text_sample.strip():
            return "Empty page"
        try:
            summary = self.ollama.query(
                prompt=f"Summarize the following document page in 1-2 concise sentences.\n\nContent:\n{text_sample}\n\nSummary:",
                system="You are a document summarization assistant. Reply in 1-2 concise sentences only.",
            )
            return summary.strip()
        except Exception:
            return text_sample[:200].replace("\n", " ") + "..."