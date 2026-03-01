# ML Project Template (Sklearn + FastAPI + Docker)

A minimal, reproducible template for ML projects:
- Train a baseline sklearn model and save artifacts
- Serve predictions via FastAPI
- Package and run the API with Docker
- Run tests with pytest

## Quickstart (WSL Ubuntu)
### 1) Create environment
conda env create -f environment.yml
conda activate mle

### 2) Train model
python src/train.py

### 3) Run API locally
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Health check: http://127.0.0.1:8000/health  
Docs: http://127.0.0.1:8000/docs

### 4) Run tests
pytest -q

## Docker
### Build
docker build -t ml-project-template:0.1 .

### Run
# Train first so artifacts/model.joblib exists
python src/train.py
docker run --rm -p 8000:8000 ml-project-template:0.1