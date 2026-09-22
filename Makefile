.PHONY: check lint typecheck test build security
check: lint typecheck test security build
lint: contrast
contrast:
	python3 src/growth_engine/shell/contrast_check.py src/growth_engine/shell/exec-shell.css
	uv run ruff check . && uv run ruff format --check .
typecheck:
	uv run mypy
test:
	uv run pytest --cov --cov-report=term
security:
	uv run bandit -q -r src && uv run pip-audit --skip-editable
build:
	uv run growth-engine report --out dist --seed 42
