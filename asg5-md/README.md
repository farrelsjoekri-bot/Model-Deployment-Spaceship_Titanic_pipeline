# ASG 05 Model Deployment - Farrel - Spaceship Titanic

This repository contains a modular machine learning project for the Spaceship Titanic dataset.

## Project structure

```text
main/
├── pyproject.toml
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── apps/
│   ├── __init__.py
│   └── main.py
├── artifacts/
│   └── spaceship_titanic_pipeline.pkl
├── data/
│   └── raw/
│       └── train.csv
└── src/
    └── spaceship_titanic/
        ├── __init__.py
        ├── pipeline.py
        ├── data/
        │   ├── __init__.py
        │   └── ingestion.py
        ├── features/
        │   ├── __init__.py
        │   └── preprocessing.py
        └── models/
            ├── __init__.py
            ├── train.py
            └── evaluate.py
```

## How to train the pipeline

```bash
python -m spaceship_titanic.pipeline
```

## How to run the app locally

```bash
streamlit run apps/main.py
```