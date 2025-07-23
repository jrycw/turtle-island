.PHONY: isort ruf fmt install preview test_pre test testv0 testv1 testv2 testv3 mypy version tag

VERSION := $(shell python -c "import turtle_island as ti; print(ti.__version__)")

isort:
	isort tests/ src/

ruf:
	ruff format tests/ src/

fmt:
	make isort && \
	make ruf

install:
	make fmt && \
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

version:
	@echo "Current version is $(VERSION)"

tag:
	@echo "Tagging version v$(VERSION)"
	git tag -a v$(VERSION) -m "Creating version v$(VERSION)"
	git push origin v$(VERSION)
