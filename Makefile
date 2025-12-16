.PHONY: fmt install preview test_pre test testv0 testv1 testv2 testv3 mypy version tag

fmt:
	ruff format tests/ src/ && \
	ruff check tests/ src/ --fix && \
	ruff check --select I --fix tests/ src/

install:
	make fmt && \
	uv pip install -e ".[dev,docs,test]"

upgrade:
	uv sync --upgrade --all-extras && \
	make install

preview:
	make install && \
	cd docs && \
	quartodoc build --verbose && \
	quarto preview --port 8001

test_pre:
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

spell:
	uv run codespell -L ser,te,revered,indx

version:
	@echo "Current version is $$(python -c 'import turtle_island as ti; print(ti.__version__)')"

tag:
	@VERSION=$$(python -c 'import turtle_island as ti; print(ti.__version__)') && \
	echo "Tagging version v$$VERSION" && \
	git tag -a v$$VERSION -m "Creating version v$$VERSION" && \
	git push origin v$$VERSION
