PYTHON ?= python3
RUN = PYTHONPATH=src $(PYTHON) -m movie_cf

.PHONY: install evaluate tune submission test zip

install:
	$(PYTHON) -m pip install -r requirements.txt

evaluate:
	$(RUN) evaluate

tune:
	$(RUN) tune

submission:
	$(RUN) predict --output submission.csv

test:
	$(PYTHON) -m pytest -q

zip:
	rm -f code.zip
	zip -r code.zip src tests movie resources Makefile pyproject.toml requirements.txt README.md \
		-x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
