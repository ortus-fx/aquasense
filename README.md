# Aquasense — Fish Feeding Advisor

A machine-learning powered feeding advisor for fish farmers. Enter 6 water quality readings and instantly receive a science-based recommendation: **Feed Now**, **Reduce Feed**, or **Halt Feeding**.

Built with XGBoost (FastAPI backend) + Streamlit (responsive frontend).

## Project Structure

```
aquasense/
├── app/
│   ├── pondiq_api.py              # FastAPI server wrapping the XGBoost model
│   ├── streamlit_app.py           # Streamlit frontend (web + mobile responsive)
│   ├── pyproject.toml             # Python project config & dependencies
│   ├── requirements_api.txt       # API dependencies (pip-compatible)
│   └── requirements_streamlit.txt # Streamlit dependencies
├── src/
│   └── models/
│       ├── feed_classifier.pkl  # Trained XGBoost model
│       ├── feature_list.pkl     # Feature names
│       └── class_names.pkl      # Class labels
├── data/                      # Raw & processed datasets
├── notebooks/                 # Jupyter notebooks (EDA, modeling)
└── docs/                      # Literature & photos
```

## Prerequisites

- **Python ≥ 3.13** (managed via [uv](https://docs.astral.sh/uv/))
- **libomp** (OpenMP runtime, required by XGBoost on macOS)

```bash
# macOS only — install OpenMP
brew install libomp
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the ML API server

```bash
uv run uvicorn pondiq_api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

**Endpoints:**

| Method | Path             | Description                                     |
| ------ | ---------------- | ----------------------------------------------- |
| `GET`  | `/health`        | Liveness check + model info                     |
| `POST` | `/predict`       | Single prediction from 6 water-quality readings |
| `POST` | `/predict/batch` | Batch predictions                               |

### 3. Start the Streamlit app

In a separate terminal:

```bash
uv run streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Alternative: use pip

If you prefer pip over uv, install from the requirements files:

```bash
pip install -r requirements_api.txt
pip install -r requirements_streamlit.txt

# Then start the servers:
python pondiq_api.py            # API on port 8000
streamlit run streamlit_app.py  # App on port 8501
```

## Model

The XGBoost classifier was trained on pond water-quality data with these 6 features:

| Parameter             | Unit | Ideal Range |
| --------------------- | ---- | ----------- |
| Dissolved Oxygen (DO) | mg/L | 5–9         |
| pH                    | —    | 6.5–8.5     |
| Ammonia               | mg/L | < 0.5       |
| Temperature           | °C   | 26–32       |
| Nitrate               | PPM  | < 40        |
| Turbidity             | NTU  | 2–3         |

**Output classes:**

- **Feed Now** — All parameters in range; efficient feed conversion expected
- **Reduce Feed** — Some parameters borderline; feed at 50%
- **Halt Feeding** — Critical parameters out of range; stop feeding for 24–48 hours

## Tech Stack

| Layer    | Technology                              |
| -------- | --------------------------------------- |
| ML Model | XGBoost (scikit-learn pipeline)         |
| API      | FastAPI + Uvicorn                       |
| Frontend | Streamlit (responsive, mobile-friendly) |
| Data     | Pandas, NumPy                           |
