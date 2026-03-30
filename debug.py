import sys
import os

sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "uploads", "sample_report.pdf")

print("=== Thong tin co ban ===")
print(f"Thu muc lam viec: {BASE_DIR}")
print(f"PDF se tim o: {PDF_PATH}")
print(f"File .env ton tai: {os.path.exists(os.path.join(BASE_DIR, '.env'))}")

print("\n=== Kiem tra .env ===")
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                print(f"  {line}")
else:
    print("  KHONG TIM THAY .env")

print("\n=== Step 1: Import ===")
try:
    import fitz
    print("PyMuPDF: OK")
except Exception as e:
    print(f"PyMuPDF: FAIL - {e}")

try:
    from src.config import config
    print(f"Config: OK")
    print(f"  OLLAMA_BASE_URL = {config.ollama.base_url}")
    print(f"  LLM_MODEL       = {config.ollama.llm_model}")
    print(f"  VISION_MODEL    = {config.ollama.vision_model}")
    print(f"  EMBED_MODEL     = {config.ollama.embed_model}")
    print(f"  UPLOAD_DIR      = {config.paths['uploads']}")
except Exception as e:
    print(f"Config: FAIL - {e}")
    sys.exit(1)

print("\n=== Step 2: Ollama ===")
try:
    from src.models.ollama_client import OllamaClient
    ollama = OllamaClient(config)
    models = ollama.list_models()
    print(f"Ket noi: OK")
    print(f"Models hien co: {models}")
    missing = []
    for m in [config.ollama.llm_model, config.ollama.vision_model, config.ollama.embed_model]:
        found = any(m in x for x in models)
        status = "OK" if found else "THIEU - chay: ollama pull " + m
        print(f"  {m}: {status}")
        if not found:
            missing.append(m)
except Exception as e:
    print(f"Ollama: FAIL - {e}")
    print("  -> Kiem tra: curl http://localhost:11434/api/tags")
    sys.exit(1)

print("\n=== Step 3: Tim PDF de test ===")
search_paths = [
    PDF_PATH,
    os.path.join(BASE_DIR, "sample_report.pdf"),
    os.path.join(BASE_DIR, "data", "sample_report.pdf"),
]

pdf_to_use = None
for p in search_paths:
    if os.path.exists(p):
        print(f"Tim thay PDF: {p}")
        pdf_to_use = p
        break

if not pdf_to_use:
    print("KHONG TIM THAY PDF nao.")
    print("Tao PDF mau bang lenh:")
    print("  pip install reportlab")
    print("  py scripts/create_sample_pdf.py")
    print("")
    print("Hoac dat bat ky file PDF nao vao thu muc:")
    print(f"  {BASE_DIR}")

    pdf_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git']]
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))

    if pdf_files:
        print(f"\nTim thay cac file PDF trong project:")
        for f in pdf_files:
            print(f"  {f}")
        pdf_to_use = pdf_files[0]
        print(f"\nSe dung: {pdf_to_use}")

if not pdf_to_use:
    print("\nKhong co PDF de test. Dung lai.")
    sys.exit(0)

print("\n=== Step 4: Doc PDF ===")
try:
    doc = fitz.open(pdf_to_use)
    print(f"So trang: {len(doc)}")
    for i in range(min(3, len(doc))):
        text = doc[i].get_text()
        print(f"  Trang {i+1}: {len(text)} ky tu | preview: {text[:100].strip()!r}")
    doc.close()
except Exception as e:
    print(f"Doc PDF FAIL: {e}")
    sys.exit(1)

print("\n=== Step 5: Vision processing ===")
try:
    from src.layers.vision_processing import VisionProcessingLayer
    vision = VisionProcessingLayer(config, ollama)

    def progress(idx, total, msg):
        print(f"  [{idx+1}/{total}] {msg}")

    result = vision.process_document(pdf_to_use, progress_callback=progress)
    print(f"\nKet qua:")
    print(f"  Tong trang: {result.total_pages}")
    print(f"  Tong regions: {len(result.all_regions())}")

    region_count = {}
    for r in result.all_regions():
        region_count[r.region_type] = region_count.get(r.region_type, 0) + 1
    print(f"  Phan loai: {region_count}")

    for r in result.all_regions()[:5]:
        print(f"  [{r.region_type}] trang={r.page_num+1} chars={len(r.content)} | {r.content[:80].strip()!r}")

except Exception as e:
    import traceback
    print(f"Vision FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 6: Embedding ===")
try:
    emb = ollama.embed(["day la van ban thu nghiem"])
    print(f"Embedding: OK - kich thuoc vector = {len(emb[0])}")
except Exception as e:
    print(f"Embedding FAIL: {e}")

print("\n=== Step 7: Ingest vao ChromaDB ===")
try:
    from src.layers.knowledge_base import KnowledgeBaseLayer
    kb = KnowledgeBaseLayer(config, ollama)
    stats_before = kb.get_stats()
    print(f"ChromaDB truoc: {stats_before['total_chunks']} chunks")

    chunk_count = kb.ingest_document(result)
    stats_after = kb.get_stats()
    print(f"Ingest: OK - them {chunk_count} chunks")
    print(f"ChromaDB sau: {stats_after['total_chunks']} chunks")

except Exception as e:
    import traceback
    print(f"Ingest FAIL: {e}")
    traceback.print_exc()

print("\n=== XONG ===")
print("Neu tat ca Step deu OK, chay lai: streamlit run app/streamlit_app.py")
print("Neu co Step FAIL, paste ket qua nay de duoc ho tro.")