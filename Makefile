install:
	pip install -r requirements.txt

data:
	python scripts/generate_sample_data.py

train: data
	python scripts/train_model.py

eda: data
	python scripts/eda.py

test: train
	pytest -q
