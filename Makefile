.PHONY: install install-dev format lint test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml]"

format:
	black .
	ruff format .

lint:
	ruff check .

test:
	pytest -q
