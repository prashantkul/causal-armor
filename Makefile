.PHONY: lint format typecheck test check build clean publish-test publish

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/causal_armor/

test:
	pytest tests/ -v

check: lint typecheck test

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*
