.PHONY: ruf install preview test_pre test testv0 testv1 testv2 testv3 mypy

ruf:
	ruff format tests/ src/

install:
	make ruf && \
	uv pip install .[]

preview:
	uv sync --group docs && \
	make install && \
	cd docs && \
	quartodoc build --verbose && \
	quarto preview --port 8001

test_pre:
	uv sync --group test && \
	make install

test:
	make test_pre && \
	uv run pytest

testv0:
	make test_pre && \
	uv run pytest -s

testv1:
	make test_pre && \
	uv run pytest -v -s

testv2:
	make test_pre && \
	uv run pytest -vv -s

testv3:
	make test_pre && \
	uv run pytest -vvv -s

mypy:
	uv run mypy src/
