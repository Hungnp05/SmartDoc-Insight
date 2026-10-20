import os
import sys
from dataclasses import replace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

DEBUG_COLLECTION = "smartdoc_debug"

print("=== Thong tin co ban ===")
print(f"Thu muc lam viec: {BASE_DIR}")
print(f"File .env ton tai: {os.path.exists(os.path.join(BASE_DIR, '.env'))}")

print("\n=== Kiem tra .env ===")
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                print(f"  {line}")
else:
    print("  KHONG TIM THAY .env (dung gia tri mac dinh)")

print("\n=== Step 1: Import ===")
try:
    import fitz
    print(f"PyMuPDF: OK ({fitz.VersionBind})")
except Exception as e:
    print(f"PyMuPDF: FAIL - {e}")
    sys.exit(1)

try:
    import paddleocr  # noqa: F401
    print("PaddleOCR: OK (OCR cho anh/PDF scan)")
except Exception:
    print("PaddleOCR: khong cai (tuy chon) - anh/PDF scan se duoc mo ta bang LLaVA")

try:
    from src.config import Config, config
    print("Config: OK")
    print(f"  OLLAMA_BASE_URL = {config.ollama.base_url}")
    print(f"  LLM_MODEL       = {config.ollama.llm_model}")
    print(f"  VISION_MODEL    = {config.ollama.vision_model}")
    print(f"  EMBED_MODEL     = {config.ollama.embed_model}")
    print(f"  CHROMA_DIR      = {config.paths['chroma']}")
except Exception as e:
    print(f"Config: FAIL - {e}")
    sys.exit(1)

print("\n=== Step 2: Ollama ===")
from src.models.ollama_client import OllamaClient

ollama = OllamaClient(config)
models = ollama.list_models()
if not models:
    print("Ollama: FAIL - khong ket noi duoc hoac chua co model nao")
    print(f"  -> Kiem tra: curl {config.ollama.base_url}/api/tags")
    sys.exit(1)
print(f"Models hien co: {models}")
for m in [config.ollama.llm_model, config.ollama.vision_model, config.ollama.embed_model]:
    found = ollama.is_model_available(m, models)
    print(f"  {m}: {'OK' if found else 'THIEU - chay: ollama pull ' + m}")

print("\n=== Step 3: Tim PDF de test ===")
search_paths = [
    os.path.join(BASE_DIR, "test_data", "sample_report.pdf"),
    os.path.join(BASE_DIR, "data", "uploads", "sample_report.pdf"),
]
pdf_to_use = next((p for p in search_paths if os.path.exists(p)), None)
if not pdf_to_use:
    print("KHONG TIM THAY PDF mau. Dat mot file PDF vao test_data/sample_report.pdf.")
    sys.exit(0)
print(f"Su dung: {pdf_to_use}")

print("\n=== Step 4: Vision processing ===")
from src.layers.vision_processing import VisionProcessingLayer

vision = VisionProcessingLayer(config, ollama)
result = vision.process_document(pdf_to_use, progress_callback=lambda i, n, msg: print(f"  [{i + 1}/{n}] {msg}"))
region_count = {}
for r in result.all_regions():
    region_count[r.region_type] = region_count.get(r.region_type, 0) + 1
print(f"  Tong trang: {result.total_pages}")
print(f"  Phan loai regions: {region_count}")
for r in result.all_regions()[:5]:
    print(f"  [{r.region_type}] trang={r.page_num + 1} chars={len(r.content)} | {r.content[:80].strip()!r}")

print("\n=== Step 5: Embedding ===")
try:
    emb = ollama.embed(["day la van ban thu nghiem"])
    print(f"Embedding: OK - kich thuoc vector = {len(emb[0])}")
except Exception as e:
    print(f"Embedding FAIL: {e}")
    sys.exit(1)

print(f"\n=== Step 6: Ingest + truy van (collection rieng '{DEBUG_COLLECTION}') ===")
from src.layers.knowledge_base import KnowledgeBaseLayer

debug_config = Config()
debug_config.chroma = replace(Config.chroma, collection_name=DEBUG_COLLECTION)
kb = KnowledgeBaseLayer(debug_config, ollama)
try:
    print(f"Ingest: OK - {kb.ingest_document(result)} chunks")
    for r in kb.retrieve("doanh thu 2024", top_k=3):
        print(f"\n[score={r.score:.3f}] page={r.chunk.metadata.get('page')} type={r.chunk.metadata.get('content_type')}")
        print(r.chunk.text[:300])
finally:
    kb.clear_all()

print("\n=== XONG ===")
print("Neu tat ca Step deu OK, chay: streamlit run app/streamlit_app.py")
print("Neu co Step FAIL, paste ket qua nay de duoc ho tro.")
