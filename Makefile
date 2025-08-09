.PHONY: setup dev test lint demo bench

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pre-commit
	pre-commit install || true

dev:
	pip install -e .

test:
	pytest -q

lint:
	pre-commit run --all-files || true

demo:
	python -m demos.run_synthetic --no-show --save-plot outputs/sample_run.png

bench:
	python -m tools.benchmark


