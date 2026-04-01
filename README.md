# PathMNIST — End-to-End MLOps Pipeline (ALU BSE)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

## Video demo (rubric)

**[Insert your YouTube link here]**  

Requirements addressed on camera: **prediction** on an uploaded image and **retraining trigger** (bulk upload + button), with the API/UI visible.

---

## Project description (assignment alignment)

| Requirement | How this repo satisfies it |
|-------------|---------------------------|
| Non-tabular data | **PathMNIST** histology images (28×28 RGB, 9 classes) |
| Offline model + deploy | Notebook trains/exports `.h5`; FastAPI serves inference |
| Metrics in notebook | **Accuracy, loss, precision, recall, F1** (≥4) + early stopping, pretrained MobileNetV2 |
| API (Python) | `src/prediction.py` — `/predict`, `/retrain`, `/health` |
| UI | `app.py` (Streamlit) — uptime/health, insights, prediction, bulk retrain |
| Retraining | Zip upload → SQLite log → **preprocess** → **fine-tune** pretrained model; MedMNIST subset fallback if zip layout is wrong |
| Load test | `locustfile.py` + record latency (see below) |
| Repo layout | Matches spec: `notebook/`, `src/`, `data/`, `models/` |

### Retrain zip layout (for *your* bulk data)

At the root of the zip, include folders named exactly **`0` … `8`**, each containing at least one image (`.png`/`.jpg`/`.jpeg`). This matches PathMNIST class ids.  

If this layout is missing, the API still runs a **PathMNIST subset** fine-tune so the pipeline is demonstrable end-to-end.

---

## URLs (local / cloud)

| Service | URL |
|--------|-----|
| API docs | `http://127.0.0.1:8000/docs` |
| Streamlit UI | `http://127.0.0.1:8501` |
| Health | `http://127.0.0.1:8000/health` |

Replace host with your cloud instance when deployed. Set `API_URL` for the UI if needed:

```powershell
$env:API_URL = "http://your-host:8000"
streamlit run app.py
```

---

## Setup (Windows / macOS / Linux)

1. **Python 3.10+** recommended (TensorFlow wheels).  
2. Clone / unzip the repository and enter the folder:

```bash
cd MLOP
```

3. Create and activate a virtual environment, then install dependencies:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. **Model file:** train with `notebook/pathmnist_mlop.ipynb` and save to `models/mobilenet_pathmnist.h5`, or copy your trained file to that path.

5. **Run the API:**

```powershell
python -m uvicorn src.prediction:app --reload --port 8000
```

6. **Run the UI** (second terminal):

```powershell
streamlit run app.py
```

7. **Optional — Locust** (with API running):

```powershell
locust -f locustfile.py --host=http://127.0.0.1:8000
```

Open the Locust web UI (default `http://localhost:8089`), set users/spawn rate, run, then export charts/tables for your report.

8. **Optional — Docker**

```bash
docker build -t pathmnist-mlops .
docker run -p 8000:8000 -p 8501:8501 pathmnist-mlops
```

---

## Flood simulation results (fill in for submission)

| Containers | Users | RPS (approx.) | p50 latency (ms) | p95 latency (ms) | Notes |
|------------|-------|---------------|------------------|------------------|-------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

*Use Locust “Statistics” / “Charts” after runs with different replica counts (e.g. Docker Compose scaling).*

---

## Repository structure

```text
MLOP/
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py                 # Streamlit UI (uptime, insights, predict, retrain)
├── locustfile.py
├── notebook/
│   └── pathmnist_mlop.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/             # uploads / extracted zips (created at runtime)
│   └── test/              # optional sample_image.jpg for Locust
└── models/
    └── mobilenet_pathmnist.h5
```

---

## Rubric checklist (self-audit)

- **Video:** Camera on; show prediction + retraining.  
- **Retraining:** Upload + DB + preprocessing + fine-tune pretrained model (`retrain_pipeline`).  
- **Prediction:** Image upload; show class name + confidence.  
- **Notebook:** Preprocessing, pretrained model, optimization, **≥4 metrics**.  
- **Deployment package:** Web UI + insights; Docker optional but provided.

---

## License / credits

PathMNIST via [MedMNIST](https://github.com/MedMNIST/MedMNIST). Use for coursework per your institution’s policy.
