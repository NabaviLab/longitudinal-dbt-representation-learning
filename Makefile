.PHONY: install install-dev format lint test

install:
	pip install -e .

install-dev:
	python -m pip install -e ".[dev,dicom,graphs]"

format:
	python -m black src tests
	python -m ruff format src tests

lint:
	python -m ruff check src tests
	python -m compileall -q src tests

test:
	PYTHONPATH=src python -m unittest discover -s tests -v
