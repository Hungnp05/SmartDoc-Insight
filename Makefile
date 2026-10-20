# SmartDoc-Insight Makefile
# ─────────────────────────
# Recipes use `python` and Python one-liners so they also work from Git Bash on Windows.

PYTHON ?= python
DOC ?= test_data/sample_report.pdf

.PHONY: setup venv run run-dev docker-up docker-down docker-logs pull-models check-models \
        test test-cov benchmark lint format clean clean-db clean-uploads info

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "Setting up SmartDoc-Insight..."
	bash scripts/setup.sh

venv:
	$(PYTHON) -m venv .venv
	@echo "Activate with 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows),"
	@echo "then run: pip install -r requirements.txt"

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	streamlit run app/streamlit_app.py \
		--server.port 8501 \
		--server.maxUploadSize 50 \
		--browser.gatherUsageStats false

run-dev:
	streamlit run app/streamlit_app.py \
		--server.port 8501 \
		--server.maxUploadSize 50 \
		--server.runOnSave true \
		--browser.gatherUsageStats false

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d
	@echo "SmartDoc-Insight running at http://localhost:8501"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f app

# ── Models ────────────────────────────────────────────────────────────────────
pull-models:
	@echo "Pulling AI models via Ollama..."
	ollama pull llama3:8b
	ollama pull llava:7b
	ollama pull nomic-embed-text
	@echo "All models ready!"

check-models:
	ollama list

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

benchmark:
	@echo "Running benchmark on $(DOC) (requires Ollama + models)..."
	$(PYTHON) scripts/benchmark.py --doc $(DOC) --output benchmark_results.json

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src/ app/ tests/ scripts/

format:
	$(PYTHON) -m ruff format src/ app/ tests/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for n in ('__pycache__', '.pytest_cache') for p in pathlib.Path('.').rglob(n) if '.venv' not in p.parts]"

clean-db:
	@echo "Clearing ChromaDB..."
	$(PYTHON) -c "import shutil; shutil.rmtree('data/chroma_db', ignore_errors=True)"
	@echo "Database cleared."

clean-uploads:
	@echo "Clearing uploaded files..."
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('data/uploads').glob('*') if p.is_file()]"
	@echo "Uploads cleared."

# ── Info ──────────────────────────────────────────────────────────────────────
info:
	@echo "SmartDoc-Insight"
	@echo "================"
	@echo "App:      http://localhost:8501"
	@echo "Ollama:   http://localhost:11434"
	@echo ""
	@echo "Models:"
	@echo "  LLM:    llama3:8b"
	@echo "  Vision: llava:7b"
	@echo "  Embed:  nomic-embed-text"
	@echo ""
	@echo "Tech Stack: Python · PyMuPDF · ChromaDB · sentence-transformers · Ollama · Streamlit"
