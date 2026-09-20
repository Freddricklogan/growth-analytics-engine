.PHONY: check lint typecheck test build security
check: lint typecheck test security build
lint:
	uv run ruff check . && uv run ruff format --check .
typecheck:
	uv run mypy
test:
	uv run pytest --cov --cov-report=term
security:
	uv run bandit -q -r src && uv run pip-audit --skip-editable
build:
	uv run growth-engine report --out dist --seed 42
