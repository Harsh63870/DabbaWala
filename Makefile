.PHONY: all dev test dataset benchmark worker clean help

help:
	@echo "Available commands:"
	@echo "  make dev        - Run FastAPI dev server with auto-reload (http://localhost:8000)"
	@echo "  make dataset    - Generate synthetic 5,000-row delivery dataset (data/orders.csv)"
	@echo "  make benchmark  - Run TabPFN vs XGBoost vs LinearRegression benchmark & generate docs/benchmark.png"
	@echo "  make worker     - Run Temporal background workflow worker"
	@echo "  make test       - Run pytest test suite"

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dataset:
	python scripts/make_dataset.py

benchmark:
	python scripts/benchmark.py

worker:
	python workflows/worker.py

test:
	pytest -v

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
