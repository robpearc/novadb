# NovaDB Makefile
# AlphaFold3-style dataset curation and feature engineering pipeline
#
# Usage:
#   make help          - Show this help message
#   make install       - Install package in development mode
#   make test          - Run tests
#   make lint          - Run linters
#   make format        - Format code
#   make clean         - Clean build artifacts

.PHONY: help install install-dev test test-fast test-cov lint format type-check \
        clean clean-build clean-pyc clean-test docs docs-serve build publish \
        pre-commit setup-hooks check-external

# Default Python interpreter
PYTHON ?= python3
PIP ?= pip3

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo "$(BLUE)NovaDB Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Installation:$(RESET)"
	@echo "  make install        Install package in development mode"
	@echo "  make install-dev    Install with all development dependencies"
	@echo "  make setup-hooks    Install pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run all tests"
	@echo "  make test-fast      Run tests without slow markers"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-parallel  Run tests in parallel"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  make lint           Run all linters (ruff, black --check)"
	@echo "  make format         Format code with black and isort"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make pre-commit     Run pre-commit on all files"
	@echo "  make check          Run all checks (lint, type-check, test)"
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "$(GREEN)Build & Release:$(RESET)"
	@echo "  make build          Build distribution packages"
	@echo "  make publish        Publish to PyPI (requires credentials)"
	@echo ""
	@echo "$(GREEN)Cleanup:$(RESET)"
	@echo "  make clean          Remove all build artifacts"
	@echo "  make clean-build    Remove build artifacts"
	@echo "  make clean-pyc      Remove Python file artifacts"
	@echo "  make clean-test     Remove test artifacts"
	@echo ""
	@echo "$(GREEN)External Tools:$(RESET)"
	@echo "  make check-external Check for required external tools (HMMER, etc.)"

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,docs,performance]"
	@echo "$(GREEN)Development installation complete$(RESET)"

setup-hooks:
	pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed$(RESET)"

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short -m "not slow"

test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=novadb --cov-report=term-missing --cov-report=html

test-parallel:
	$(PYTHON) -m pytest $(TEST_DIR) -v -n auto

test-integration:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "integration"

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint:
	@echo "$(BLUE)Running ruff...$(RESET)"
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "$(BLUE)Checking formatting with black...$(RESET)"
	$(PYTHON) -m black --check $(SRC_DIR) $(TEST_DIR)

format:
	@echo "$(BLUE)Formatting with isort...$(RESET)"
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(BLUE)Formatting with black...$(RESET)"
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Formatting complete$(RESET)"

type-check:
	@echo "$(BLUE)Running mypy...$(RESET)"
	$(PYTHON) -m mypy $(SRC_DIR)

pre-commit:
	pre-commit run --all-files

check: lint type-check test
	@echo "$(GREEN)All checks passed$(RESET)"

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------

docs:
	cd $(DOCS_DIR) && make html

docs-serve:
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server

#------------------------------------------------------------------------------
# Build & Release
#------------------------------------------------------------------------------

build: clean-build
	$(PYTHON) -m build

publish: build
	$(PYTHON) -m twine upload dist/*

publish-test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.mypy_cache' -exec rm -rf {} +
	find . -name '.ruff_cache' -exec rm -rf {} +

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/

#------------------------------------------------------------------------------
# External Tools Check
#------------------------------------------------------------------------------

check-external:
	@echo "$(BLUE)Checking for external tools...$(RESET)"
	@echo ""
	@echo "HMMER Suite:"
	@which jackhmmer > /dev/null 2>&1 && echo "  $(GREEN)✓ jackhmmer$(RESET)" || echo "  $(RED)✗ jackhmmer (not found)$(RESET)"
	@which hmmbuild > /dev/null 2>&1 && echo "  $(GREEN)✓ hmmbuild$(RESET)" || echo "  $(RED)✗ hmmbuild (not found)$(RESET)"
	@which hmmsearch > /dev/null 2>&1 && echo "  $(GREEN)✓ hmmsearch$(RESET)" || echo "  $(RED)✗ hmmsearch (not found)$(RESET)"
	@which nhmmer > /dev/null 2>&1 && echo "  $(GREEN)✓ nhmmer$(RESET)" || echo "  $(RED)✗ nhmmer (not found)$(RESET)"
	@echo ""
	@echo "HH-Suite:"
	@which hhblits > /dev/null 2>&1 && echo "  $(GREEN)✓ hhblits$(RESET)" || echo "  $(RED)✗ hhblits (not found)$(RESET)"
	@echo ""
	@echo "MMseqs2:"
	@which mmseqs > /dev/null 2>&1 && echo "  $(GREEN)✓ mmseqs$(RESET)" || echo "  $(RED)✗ mmseqs (not found)$(RESET)"
	@echo ""
	@echo "$(YELLOW)Note: Install missing tools via conda:$(RESET)"
	@echo "  conda install -c bioconda hmmer hhsuite mmseqs2"

#------------------------------------------------------------------------------
# Development Utilities
#------------------------------------------------------------------------------

# Generate requirements.txt from pyproject.toml
requirements:
	$(PIP) freeze > requirements-freeze.txt
	@echo "$(GREEN)Generated requirements-freeze.txt$(RESET)"

# Run a single test file
test-file:
	$(PYTHON) -m pytest $(FILE) -v --tb=long

# Profile a specific function
profile:
	$(PYTHON) -m cProfile -s cumulative $(FILE)

# Memory profile
memprofile:
	$(PYTHON) -m memory_profiler $(FILE)
