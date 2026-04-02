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

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages_data = self._pdf_to_pages(file_path)
        else:
            img = self._load_image(file_path)
            pages_data = [(0, img, "", [])]

        doc = ProcessedDocument(source_file=file_path.name, total_pages=len(pages_data))

        for idx, page_data in enumerate(pages_data):
            page_num, page_img, page_text, page_blocks = page_data
            if progress_callback:
                progress_callback(idx, len(pages_data), f"Analyzing page {page_num + 1}...")
            processed_page = self._process_page(page_img, page_num, page_text, page_blocks)
            doc.pages.append(processed_page)

        logger.info(f"Done: {len(doc.pages)} pages, {len(doc.all_regions())} regions")
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
                blocks = page.get_text("blocks")
                result.append((page_num, img, text, blocks))

        return result

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _process_page(self, page_img, page_num, page_text="", page_blocks=None):
        processed = ProcessedPage(page_num=page_num)
        regions = self._extract_regions_from_blocks(page_blocks, page_img, page_num, page_text)
        for region in regions:
            if region.content and len(region.content.strip()) > 10:
                processed.regions.append(region)
        processed.full_text = self._build_page_text(processed.regions)
        processed.page_summary = self._generate_page_summary(processed.regions)
        return processed

    def _extract_regions_from_blocks(self, blocks, page_img, page_num, page_text):
        h, w = page_img.shape[:2]
        regions = []

        if not blocks and not page_text:
            return regions

        if not blocks:
            return self._split_text_into_regions(page_text, page_img, page_num)

        text_chunks = []
        current_chunk = []
        current_len = 0

        for block in blocks:
            if len(block) < 5:
                continue
            block_text = block[4] if len(block) > 4 else ""
            if not isinstance(block_text, str):
                continue
            block_text = block_text.strip()
            if not block_text:
                continue

            is_table_block = self._looks_like_table(block_text)

            if is_table_block:
                if current_chunk:
                    regions.append(DocumentRegion(
                        region_type="text",
                        bbox=BoundingBox(0, 0, w, h),
                        confidence=1.0,
                        page_num=page_num,
                        content="\n".join(current_chunk),
                    ))
                    current_chunk = []
                    current_len = 0

                regions.append(DocumentRegion(
                    region_type="table",
                    bbox=BoundingBox(0, 0, w, h),
                    confidence=0.85,
                    page_num=page_num,
                    content=block_text,
                ))
                continue

            if current_len + len(block_text) > 600 and current_chunk:
                regions.append(DocumentRegion(
                    region_type="text",
                    bbox=BoundingBox(0, 0, w, h),
                    confidence=1.0,
                    page_num=page_num,
                    content="\n".join(current_chunk),
                ))
                current_chunk = []
                current_len = 0

            current_chunk.append(block_text)
            current_len += len(block_text)

        if current_chunk:
            regions.append(DocumentRegion(
                region_type="text",
                bbox=BoundingBox(0, 0, w, h),
                confidence=1.0,
                page_num=page_num,
                content="\n".join(current_chunk),
            ))

        return self._merge_adjacent_table_blocks(regions)

    def _looks_like_table(self, text: str) -> bool:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            return False
        pipe_lines = sum(1 for l in lines if l.count("|") >= 2)
        if pipe_lines >= 2:
            return True
        tab_lines = sum(1 for l in lines if l.count("\t") >= 2)
        if tab_lines >= len(lines) * 0.6:
            return True
        if len(lines) >= 3:
            num_pattern = 0
            for line in lines:
                parts = line.split()
                nums = sum(1 for p in parts if any(c.isdigit() for c in p))
                if nums >= 3:
                    num_pattern += 1
            if num_pattern >= len(lines) * 0.5:
                return True
        return False

    def _merge_adjacent_table_blocks(self, regions: list) -> list:
        if not regions:
            return regions
        merged = []
        i = 0
        while i < len(regions):
            if regions[i].region_type != "table":
                merged.append(regions[i])
                i += 1
                continue
            combined = regions[i].content
            j = i + 1
            while j < len(regions) and regions[j].region_type == "table":
                combined += "\n" + regions[j].content
                j += 1
            merged.append(DocumentRegion(
                region_type="table",
                bbox=regions[i].bbox,
                confidence=regions[i].confidence,
                page_num=regions[i].page_num,
                content=combined,
            ))
            i = j
        return merged
    
    
    def _split_text_into_regions(self, page_text, page_img, page_num):
        h, w = page_img.shape[:2]
        regions = []

        if not page_text or not page_text.strip():
            return regions

        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        if not lines:
            return regions

        chunks = []
        current = []
        current_len = 0
        for line in lines:
            current.append(line)
            current_len += len(line)
            if current_len > 600:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append("\n".join(current))

        chunk_h = h // max(len(chunks), 1)
        for i, chunk_text in enumerate(chunks):
            regions.append(DocumentRegion(
                region_type="text",
                bbox=BoundingBox(0, i * chunk_h, w, min((i + 1) * chunk_h, h)),
                confidence=1.0,
                page_num=page_num,
                content=chunk_text,
            ))

        return regions

    def _describe_figure(self, figure_img, region):
        try:
            from src.config import CHART_DESCRIPTION_PROMPT
            img_b64 = self._image_to_base64(figure_img)
            description = self.ollama.vision_query(
                prompt=CHART_DESCRIPTION_PROMPT,
                image_base64=img_b64
            )
            return f"[FIGURE - Page {region.page_num + 1}]\n{description}"
        except Exception as e:
            logger.error(f"Figure description failed: {e}")
            return f"[FIGURE - Page {region.page_num + 1}]"

    @staticmethod
    def _image_to_base64(img):
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_page_text(self, regions):
        parts = []
        for r in regions:
            if r.content:
                prefix = "# " if r.region_type == "title" else ""
                parts.append(f"{prefix}{r.content}")
        return "\n\n".join(parts)

    def _generate_page_summary(self, regions):
        text_sample = self._build_page_text(regions)[:1500]
        if not text_sample.strip():
            return "Empty page"
        try:
            summary = self.ollama.query(
                prompt=f"Tom tat noi dung trang tai lieu sau trong 1-2 cau ngan gon.\n\nNoi dung:\n{text_sample}\n\nTom tat:",
                system="Ban la tro ly tom tat tai lieu. Chi tra loi bang 1-2 cau suc tich.",
            )
            return summary.strip()
        except Exception:
            return text_sample[:200].replace("\n", " ") + "..."