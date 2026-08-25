PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff

.PHONY: build test test-all format format-check lint typecheck release help clean install-dev install-torch check-package examples

help:
	@echo "Targets:"
	@echo "  install-dev   - install project + dev extras into .venv (uv)"
	@echo "  install-torch - install CPU-only torch extra into .venv"
	@echo "  build         - build sdist + wheel into dist/"
	@echo "  test          - run pytest excluding slow tests (model tests skip without torch)"
	@echo "  test-all      - run all pytest tests (slow + model included)"
	@echo "  format        - ruff format"
	@echo "  format-check  - ruff format --check + ruff check"
	@echo "  lint          - ruff check"
	@echo "  typecheck     - mypy"
	@echo "  release       - build wheel + sdist (local artifacts only)"
	@echo "  check-package - build wheel, install and smoke-test"
	@echo "  examples      - run all example scripts"
	@echo "  clean         - remove build artifacts"

install-dev:
	uv venv .venv --python python3.11 || uv venv .venv
	uv pip install --python .venv/bin/python -e ".[dev]"

install-torch:
	uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

build:
	$(PYTHON) -m build --wheel --no-isolation

test:
	$(PYTHON) -m pytest -q -m "not slow"

test-all:
	$(PYTHON) -m pytest -q

format:
	$(RUFF) format src tests examples scripts

format-check:
	$(RUFF) format --check src tests examples scripts && $(RUFF) check src tests examples scripts

lint:
	$(RUFF) check src tests examples scripts

typecheck:
	$(PYTHON) -m mypy src/hypofuse

release:
	$(PYTHON) -m build --wheel --sdist --no-isolation

check-package:
	$(PYTHON) scripts/check_package.py --python $(PYTHON)

examples:
	$(PYTHON) scripts/run_examples.py

clean:
	rm -rf build dist src/hypofuse.egg-info .pytest_cache .ruff_cache .coverage htmlcov