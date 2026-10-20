# SmartDoc-Insight
### Multi-Modal RAG for Complex Documents

> **Local-First** · **Table-Aware** · **Vision-Enhanced** · **Privacy-Preserving**

A Retrieval-Augmented Generation system that understands PDFs containing mixed content: raw text, financial tables, and growth charts — all running entirely on local hardware (RTX 4050 6GB VRAM).

Video Demo: https://www.youtube.com/watch?v=JRN3Oe5GRZA&t=157s

---

## Key Features

- **Table-Aware Chunking**: Each table is extracted whole (header row + caption) and stored as one chunk, so values never lose their column
- **Multi-Modal Understanding**: Vector and raster charts are rendered and described by LLaVA, together with the axis labels extracted from the PDF
- **OCR for scans and images** (optional PaddleOCR), with a LLaVA fallback
- **Advanced Retrieval**: Metadata-enriched chunks, cross-encoder reranking, context trimmed to the model's window
- **Fully Local & Private**: No data leaves your machine — ideal for sensitive enterprise documents

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SmartDoc-Insight Pipeline                │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Layer A     │  Layer B     │  Layer C     │  Layer D      │
│  Vision      │  Knowledge   │  Retrieval   │  Presentation │
│  Processing  │  Base        │  & Reasoning │  (Streamlit)  │
├──────────────┼──────────────┼──────────────┼───────────────┤
│ PyMuPDF      │ ChromaDB     │ Vector Search│ Upload UI     │
│ find_tables  │ Hybrid Meta  │ Cross-encoder│ Streaming Chat│
│ LLaVA-7B     │ nomic-embed  │ Llama3-8B    │ Source Viewer │
│ PaddleOCR*   │ Embeddings   │ Inference    │ Debug Tab     │
└──────────────┴──────────────┴──────────────┴───────────────┘
                                          * optional
```

All layers are wired by `src/pipeline.py` (`SmartDocPipeline`), which the app, the benchmark and `debug.py` share.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (RTX 4050 6GB+ recommended)
- [Ollama](https://ollama.ai/) installed locally (or use the Ollama container from docker-compose)

### 1. Clone & Setup
```bash
git clone <repo>
cd smartdoc-insight
cp .env.example .env
```

### 2. Pull AI Models via Ollama
```bash
ollama pull llama3:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

### 3. Run with Docker
```bash
docker-compose -f docker/docker-compose.yml up --build
```

### 4. Access the App
Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
SmartDoc-Insight/
├── .env.example                    # Environment configuration template
├── Makefile                        # Common commands (run, test, lint, benchmark)
├── requirements.txt                # Python dependencies
├── requirements-ocr.txt            # Optional OCR dependencies (PaddleOCR)
├── ruff.toml                       # Lint configuration
├── debug.py                        # Environment / Ollama diagnostics
│
├── app/
│   └── streamlit_app.py            # Streamlit web application
│
├── src/
│   ├── config.py                   # Configuration and prompts (reads .env)
│   ├── pipeline.py                 # SmartDocPipeline facade
│   ├── layers/
│   │   ├── vision_processing.py    # Layer A: tables, charts, text, OCR
│   │   ├── knowledge_base.py       # Layer B: chunking, embeddings, ChromaDB
│   │   └── retrieval_reasoning.py  # Layer C: retrieval, reranking, generation
│   ├── models/
│   │   └── ollama_client.py        # Ollama REST client
│   └── utils/
│       └── table_extractor.py      # HTML/grid table → Markdown helpers
│
├── docker/                         # Dockerfile + docker-compose.yml
├── scripts/
│   ├── setup.sh                    # One-click setup (bash)
│   └── benchmark.py                # Accuracy / latency benchmark
├── test_data/
│   └── sample_report.pdf           # Sample financial report used by tests and benchmark
└── tests/
    └── test_pipeline.py            # Test suite (no Ollama required)
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Layout** | PyMuPDF (`find_tables`, `cluster_drawings`) | Detect tables, charts and text blocks |
| **OCR** (optional) | PaddleOCR | Text from image uploads and scanned pages |
| **Vision LLM** | LLaVA-7B (via Ollama) | Describe charts & figures |
| **Main LLM** | Llama 3 8B (via Ollama) | Answer questions |
| **Embeddings** | `nomic-embed-text` | Semantic encoding |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-rank retrieved chunks |
| **Vector DB** | ChromaDB | Similarity search |
| **UI** | Streamlit | Web interface |
| **Runtime** | Docker + Ollama | Deployment |

---

## Key Technical Choices

### 1. Table-Aware RAG
Instead of naively chunking PDFs, we:
1. Detect tables with PyMuPDF `find_tables()` (ruled tables)
2. Convert each table to Markdown with its header row, and prepend the caption above it (e.g. *"Bang 1: ... (don vi: ty dong)"*)
3. Store the entire table as one chunk; tables larger than `TABLE_MAX_CHARS` are split by rows with the header repeated in every part
4. Remove the table's cells from the page text so they are not indexed twice

Borderless tables (no ruling lines) are not detected as tables and end up as text.

### 2. Charts → Text
Vector charts (drawing clusters) and embedded images are rendered at `PDF_DPI` and described by LLaVA. The axis labels and legend text inside the chart area are passed to LLaVA as hints and stored with the description, so the chart stays searchable even if the vision model fails.

### 3. Fitting 6GB VRAM
- Both LLaVA-7B and Llama3-8B run with Q4 quantization via Ollama
- Ingestion runs in phases — layout for every page, then all LLaVA calls, then all page summaries — so Ollama swaps models once instead of once per page
- The LLM context window is `LLM_NUM_CTX` (6144 by default); the retrieved context is trimmed to fit it

### 4. Hybrid Metadata Store
Every chunk stores:
```json
{
  "page": 2,
  "content_type": "table",
  "page_summary": "Q3 financial results with revenue breakdown",
  "source_file": "annual_report_2024.pdf",
  "bbox": "x1,y1,x2,y2"
}
```

---

## Benchmark

`scripts/benchmark.py` ingests `test_data/sample_report.pdf` into a separate `smartdoc_benchmark` collection (your knowledge base is not touched) and asks 9 Vietnamese questions (6 table, 1 text, 2 chart).

Scoring is answer-based: each question lists the facts the answer must contain (e.g. `2.340` for Q3 revenue). Numbers are normalized (`2.340`, `2,340` and `2340` all match) and a fact may not appear in the question itself, so restating the question scores 0.

```bash
python scripts/benchmark.py --doc test_data/sample_report.pdf --output benchmark_results.json
```

Results are written to the `--output` file (default `benchmark_results.json`).

> The previous README figures (84% overall) were measured with an earlier keyword scorer that gave credit for words already present in the question, and before the table/chart extraction was rewritten. Re-run the benchmark to get current numbers for your hardware.

The `bert.embeddings.position_ids | UNEXPECTED` warning when loading the cross-encoder model is normal and can be safely ignored.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, Ubuntu 20.04+ | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| RAM | 8GB | 16GB |
| GPU VRAM | Not required | 6GB (RTX 4050+) |
| Storage | 15GB (models + deps) | 20GB |

---

## Installation

### Step 1: Install Ollama

Download and install Ollama from https://ollama.ai

After installation, Ollama runs in the background automatically. Verify it is running:

```
curl http://localhost:11434/api/tags
```

If it returns JSON, Ollama is running correctly. If you see `bind: Only one usage` when running `ollama serve`, Ollama is already running — ignore that error.

### Step 2: Pull AI Models

Each model downloads several GB, so a stable internet connection is required:

```
ollama pull llama3:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

Verify all three models are available with `ollama list`.

### Step 3: Clone the Project

```
git clone <repo_url>
cd smartdoc-insight
```

### Step 4: Create a Virtual Environment

**Windows:**

```
python -m venv .venv
.venv\Scripts\activate
```

**Linux / Mac:**

```
python3 -m venv .venv
source .venv/bin/activate
```

### Step 5: Install Python Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

Optional — OCR for image uploads and scanned PDFs:

```
pip install -r requirements-ocr.txt
```

Without it, images and scanned pages are described by LLaVA instead of OCR'd.

### Step 6: Configure Environment Variables

```
copy .env.example .env        # Windows
cp .env.example .env          # Linux / Mac
```

Make sure `OLLAMA_BASE_URL` points to `http://localhost:11434` for a local Ollama. `src/config.py` loads `.env` automatically (values in `.env` override the shell environment).

---

## Running the Application

```
streamlit run app/streamlit_app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## Usage

### Uploading a Document

1. Drag a PDF or image into the **Upload Document** area in the left sidebar, or click **Browse files**
2. Click **Process Document**
3. Wait for the progress bar to complete — the sidebar shows the number of chunks and the detected regions (text / table / figure)

Uploading a file with the same name as an indexed document replaces it.

### Asking Questions

Once the sidebar shows indexed chunks, type a question into the chat input. Answers stream in, followed by the source chunks used.

```
What was the Q3 revenue?
Which business segment had the highest growth?
Summarize the main risks mentioned in the report.
Compare performance between 2023 and 2024.
```

Use **Chat with this doc** in the sidebar to restrict answers to one document. The Top-K sliders apply to your session only.

### Debug Tab

Switch to the **Debug** tab to inspect retrieval: chunk scores before and after reranking (respects the document filter).

---

## Running Tests

```
pytest tests/ -v
```

Tests do not require Ollama (it is mocked) and use a temporary ChromaDB. Lint with `ruff check src/ app/ tests/ scripts/`.

---

## Running with Docker

```
docker-compose -f docker/docker-compose.yml up --build
```

The compose file starts Ollama, pulls the three models (the app waits until this finishes), then starts the app at `http://localhost:8501`. Build with `--build-arg INSTALL_OCR=1` to include PaddleOCR.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `LLM_MODEL` | `llama3:8b` | Model used to generate answers |
| `VISION_MODEL` | `llava:7b` | Model used to describe charts and figures |
| `EMBED_MODEL` | `nomic-embed-text` | Model used to generate embeddings |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds (vision calls get 2×) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Maximum answer length in tokens |
| `LLM_NUM_CTX` | `6144` | LLM context window; retrieved context is trimmed to fit |
| `PDF_DPI` | `200` | Resolution for rendering charts and scanned pages |
| `CHUNK_SIZE` | `1200` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between text chunks |
| `TABLE_MAX_CHARS` | `6000` | Tables longer than this are split by rows (header repeated) |
| `RETRIEVAL_TOP_K` | `10` | Number of candidate chunks retrieved initially |
| `RERANK_TOP_K` | `4` | Number of chunks kept after reranking |
| `PAGE_SUMMARY` | `true` | One LLM call per page for `page_summary` metadata; `false` speeds up ingestion |
| `OCR_ENABLED` | `true` | Use PaddleOCR when installed |
| `OCR_LANG` | `vi` | PaddleOCR language |
| `CHROMA_DIR` | `data/chroma_db` | ChromaDB location |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking |

---

## Troubleshooting

**Problem: `0 chunks indexed` / "No content could be extracted"**

The file has no text layer (scan or image) and OCR is not installed — install `requirements-ocr.txt`. Also check that `OLLAMA_BASE_URL` in `.env` points to a running Ollama: embedding failures now stop ingestion with an error instead of storing empty vectors.

**Problem: Answers got worse after upgrading**

Embeddings now use the `search_document:` / `search_query:` prefixes that `nomic-embed-text` expects. Documents indexed by an older version must be re-indexed: clear the knowledge base and upload them again.

**Problem: `ollama serve` reports `bind: Only one usage`**

Ollama is already running. No action needed.

**Problem: Model not found when querying**

Re-pull the required models (`ollama pull llama3:8b`, `ollama pull nomic-embed-text`).

**Problem: Port 8501 is already in use**

```
streamlit run app/streamlit_app.py --server.port 8502
```

**Problem: Want to clear all indexed data**

Use the **Clear Knowledge Base** button in the sidebar (then **Confirm**), or delete the `data/chroma_db/` directory while the app is stopped.

---

## Diagnostics

If you encounter an error that cannot be resolved, run the diagnostic script and share the output:

```
python debug.py
```

It checks the configuration, Ollama models, layout extraction on the sample PDF, embeddings and a test query (in a separate collection).

## Note
Due to hardware limitations, the vision and text models cannot both stay in 6GB of VRAM, so Ollama swaps them. Ingestion is ordered to minimize swaps, but expect noticeably slower responses than on a GPU that fits both models.
