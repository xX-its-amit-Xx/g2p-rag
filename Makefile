.PHONY: help install install-dev test test-api test-internal lint typecheck download-index package-index build publish clean

CHROMA_RELEASE_URL ?= https://github.com/$(shell git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/' 2>/dev/null || echo "amitshenoy/g2p-rag")/releases/download/v0.1.0/chroma_index_v0.1.0.tar.gz
INDEX_DIR ?= data

help:
	@echo "Available targets:"
	@echo "  install          Install project dependencies with uv"
	@echo "  install-dev      Install project + dev dependencies with uv"
	@echo "  test             Run full test suite"
	@echo "  test-api         Run API tests only"
	@echo "  test-internal    Run internal tests only"
	@echo "  lint             Run ruff linter"
	@echo "  typecheck        Run mypy type checker"
	@echo "  download-index   Download pre-built ChromaDB snapshot from GitHub Releases"
	@echo "  package-index    Package local ChromaDB snapshot for upload to GitHub Releases"
	@echo "  build            Build distribution packages"
	@echo "  publish          Publish to PyPI (set PYPI_TOKEN env var first)"
	@echo "  clean            Remove build artifacts and caches"

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	python -m pytest tests/ -v

test-api:
	python -m pytest tests/test_api.py -v

test-internal:
	python -m pytest tests/internal/ -v

lint:
	python -m ruff check src/ tests/ || true

typecheck:
	python -m mypy src/g2p_rag/ --ignore-missing-imports || true

download-index:
	@echo "Downloading pre-built index from GitHub Releases..."
	@mkdir -p $(INDEX_DIR)
	@if command -v curl >/dev/null 2>&1; then \
		curl -L "$(CHROMA_RELEASE_URL)" | tar -xz -C $(INDEX_DIR); \
	elif command -v wget >/dev/null 2>&1; then \
		wget -qO- "$(CHROMA_RELEASE_URL)" | tar -xz -C $(INDEX_DIR); \
	else \
		echo "Error: curl or wget required. Install one and retry."; exit 1; \
	fi
	@echo "Index downloaded to $(INDEX_DIR)/chroma"

package-index:
	@echo "Packaging local index for release..."
	tar -czf chroma_index_v$(shell python -c "import g2p_rag; print(g2p_rag.__version__)").tar.gz -C data chroma/
	@echo "Upload the .tar.gz to GitHub Releases as a release asset."

build:
	uv build

publish:
	uv publish

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
