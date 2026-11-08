PYTHON ?= python3
RUN = PYTHONPATH=src $(PYTHON) -m movie_dt

.PHONY: install fetch evaluate submission tree test zip

install:
	$(PYTHON) -m pip install -r requirements.txt

fetch:
	$(RUN) fetch

evaluate:
	$(RUN) evaluate

submission:
	$(RUN) predict --tree-output submission_tree.csv --forest-output submission_forest.csv --tree-image tree.png

tree:
	$(RUN) draw --tree-image tree.png

zip:
	rm -f code.zip
	zip -r code.zip src Makefile pyproject.toml requirements.txt README.md .env.example \
		-x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
