PYTHON ?= python3
RUN = PYTHONPATH=src $(PYTHON) -m movie_knn

.PHONY: install fetch evaluate submission test zip

install:
	$(PYTHON) -m pip install -r requirements.txt

fetch:
	$(RUN) fetch

evaluate:
	$(RUN) evaluate

submission:
	$(RUN) predict --output submission.csv --params resources/user_params.csv

test:
	$(PYTHON) -m pytest -q

zip:
	rm -f code.zip
	zip -r code.zip src tests resources Makefile pyproject.toml requirements.txt README.md .env.example \
		-x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
