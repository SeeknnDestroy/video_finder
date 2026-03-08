.PHONY: install run test

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload

test:
	pytest
