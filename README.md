# PathMNIST MLOps pipeline

Histology image classification on **PathMNIST** (MedMNIST): train offline in the notebook, serve predictions via **FastAPI**, dashboard via **Streamlit**, deploy on **Render**, load-test with **Locust** and **Docker Compose** scaling.

## Submission

| Item | Link |
|------|------|
| GitHub | https://github.com/davidbirenzi/MLOP |
| Video demo | https://youtu.be/CH3YgUBzAPc |
| API (cloud) | https://pathmnist-api-service.onrender.com |
| UI (cloud) | https://pathmnist-ui-services.onrender.com |

API docs: `/docs` · Health: `/health`


- **Non-tabular data:** PathMNIST images  
- **Pipeline:** acquisition → preprocessing → model → test → **retrain** (upload zip + trigger) → **API**  
- **Notebook:** `notebook/pathmnist_mlop.ipynb` — preprocessing, MobileNetV2 + early stopping, **accuracy, loss, precision, recall, F1**  
- **Model file:** `models/mobilenet_pathmnist.h5`  
- **UI:** `app.py` — uptime/health, data insight charts, predict, retrain  
- **Cloud:** API + UI on Render  
- **Flood test:** `locustfile.py` → `POST /predict`; results + screenshots in `locust_screenshots/` (see `locust_screenshots/REPORT.md`)
- **Database:** SQLite log for each retrain upload (see below)

## Sample retrain zip

`exported_samples.zip` in the repo root is a ready-made upload: unzip and you get `exported_samples/` with class folders `0`–`8` and one image per class (PathMNIST-style layout). Use it in the UI or `POST /retrain` to demo the pipeline without building your own archive.

## Retrain zip layout

```
your_zip.zip
└── exported_samples/   (name can vary; API finds nested class folders)
    ├── 0/
    ├── 1/
    …
    └── 8/
```

## Database (SQLite)

Retrain uploads are logged in **`data/retraining_log.db`** (created when the API first runs). Table **`uploads`**: `id`, `filename`, `upload_date`, `sample_count`. On `/retrain`, the API inserts a row when the zip is saved, then updates `sample_count` after extraction and preprocessing finish. That file is **gitignored** (local/cloud runtime state); graders see the schema and behavior in `src/prediction.py` (`init_db`, `INSERT`, `UPDATE`).

## Run locally

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate
pip install -r requirements.txt
```

Terminal 1 — API: `python -m uvicorn src.prediction:app --reload --port 8000`  
Terminal 2 — UI: `streamlit run app.py`  
(Optional) UI → set `API_URL` if API is not on `http://127.0.0.1:8000`.

## Docker

Build: `docker build -t pathmnist-mlops .`  
Compose (scaled API + Nginx): `docker compose up -d --scale api=N`  
Locust against LB: `locust -f locustfile.py --host http://127.0.0.1:8000` → http://localhost:8089

## Render (what we used)

- **API:** Python 3.10, build `pip install -r requirements-docker.txt`, start `python -m uvicorn src.prediction:app --host 0.0.0.0 --port $PORT`  
- **UI:** same build, start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, env `API_URL=https://pathmnist-api-service.onrender.com`

## Locust results (local, 5 users)

| Containers | RPS | p50 (ms) | p95 (ms) | Failures |
|-----------:|----:|---------:|---------:|----------|
| 1 | 2.4 | 170 | 290 | ~0.2% (1/470) |
| 2 | 2.2 | 170 | 270 | 0% |
| 3 | 1.9 | 200 | 470 | 0% |

Detail: `locust_screenshots/REPORT.md`

## Repo layout

```
MLOP/
├── README.md
├── exported_samples.zip          # sample retrain bundle (0–8 class folders)
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── app.py
├── locustfile.py
├── locust_screenshots/
├── notebook/pathmnist_mlop.ipynb
├── src/preprocessing.py, model.py, prediction.py
├── data/train, data/test         # + retraining_log.db at runtime (gitignored)
└── models/mobilenet_pathmnist.h5
```

Dataset: [MedMNIST / PathMNIST](https://github.com/MedMNIST/MedMNIST).
