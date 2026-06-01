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
	@VERSION=$$(python -c "import g2p_rag; print(g2p_rag.__version__)"); \
	TARBALL=chroma_index_v$$VERSION.tar.gz; \
	SHAFILE=chroma_index_v$$VERSION.sha256; \
	BASEURL=$$(echo "$(CHROMA_RELEASE_URL)" | sed 's|/[^/]*$$||'); \
	if command -v curl >/dev/null 2>&1; then \
		curl -L -o $$TARBALL "$$BASEURL/$$TARBALL" || { echo "Error: failed to download $$TARBALL"; exit 1; }; \
		curl -L -o $$SHAFILE "$$BASEURL/$$SHAFILE" || { echo "Error: failed to download $$SHAFILE (no integrity file published)"; exit 1; }; \
	elif command -v wget >/dev/null 2>&1; then \
		wget -q -O $$TARBALL "$$BASEURL/$$TARBALL" || { echo "Error: failed to download $$TARBALL"; exit 1; }; \
		wget -q -O $$SHAFILE "$$BASEURL/$$SHAFILE" || { echo "Error: failed to download $$SHAFILE (no integrity file published)"; exit 1; }; \
	else \
		echo "Error: curl or wget required. Install one and retry."; exit 1; \
	fi; \
	echo "Verifying SHA256 of $$TARBALL against $$SHAFILE..."; \
	python -c "import hashlib, sys; \
	expected = open(sys.argv[2]).read().split()[0].strip().lower(); \
	h = hashlib.sha256(); \
	f = open(sys.argv[1], 'rb'); \
	[h.update(b) for b in iter(lambda: f.read(65536), b'')]; \
	f.close(); \
	actual = h.hexdigest().lower(); \
	(print('OK: sha256 matches', actual) if actual == expected else (print('FAIL: sha256 mismatch\n  expected:', expected, '\n  actual:  ', actual), sys.exit(1)))" \
		$$TARBALL $$SHAFILE || { echo "Aborting: integrity check failed. Refusing to extract a tampered or corrupt snapshot."; rm -f $$TARBALL $$SHAFILE; exit 1; }; \
	tar -xzf $$TARBALL -C $(INDEX_DIR); \
	rm -f $$TARBALL $$SHAFILE; \
	echo "Index downloaded and verified to $(INDEX_DIR)/chroma"

package-index:
	@echo "Packaging local index for release..."
	@VERSION=$$(python -c "import g2p_rag; print(g2p_rag.__version__)"); \
	TARBALL=chroma_index_v$$VERSION.tar.gz; \
	SHAFILE=chroma_index_v$$VERSION.sha256; \
	tar -czf $$TARBALL -C data chroma/; \
	echo "Computing SHA256 of $$TARBALL..."; \
	python -c "import hashlib, sys; \
	h = hashlib.sha256(); \
	f = open(sys.argv[1], 'rb'); \
	[h.update(b) for b in iter(lambda: f.read(65536), b'')]; \
	f.close(); \
	open(sys.argv[2], 'w').write(h.hexdigest() + '  ' + sys.argv[1] + '\n'); \
	print('wrote', sys.argv[2], '=', h.hexdigest())" $$TARBALL $$SHAFILE; \
	echo ""; \
	echo "Created $$TARBALL and $$SHAFILE."; \
	echo "Upload BOTH files to GitHub Releases as release assets so that"; \
	echo "'make download-index' can verify integrity on the consumer side."

build:
	uv build

publish:
	uv publish

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
