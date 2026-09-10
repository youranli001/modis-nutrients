.PHONY: install test lint notebook clean

install:
	pip install -e ".[dev,keras]"

test:
	pytest -q

lint:
	ruff check src tests aws

# re-run the results notebook from a clean kernel (writes the README figures to docs/figures)
notebook:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/02_results.ipynb

clean:
	rm -rf notebooks/.ipynb_checkpoints src/*.egg-info
