.PHONY: install preview test

install:
	uv pip install .[]

preview:
	uv sync --group docs && \
	make install && \
	cd docs && \
	quartodoc build --verbose && \
	quarto preview --port 8001

test:
	uv sync --group test && \
	make install && \
	uv run pytest --cov --cov-branch --cov-report=xml
