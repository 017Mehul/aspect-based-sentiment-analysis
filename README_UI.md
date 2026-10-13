# Aspectly web UI

The repository now includes a Next.js interface with an interactive Three.js hero and a Node.js API proxy. The Python project remains the source of truth for inference.

## Architecture

- **Next.js / React**: responsive white and yellow-green interface, 3D animation, review input and result visualization.
- **Next.js route**: `POST /api/analyze` validates requests and forwards them to the Python API.
- **FastAPI**: `api.py` calls the existing `absa_project.py` functions. Aspect polarity is explicitly labeled as the lexicon/proximity heuristic. A review-level sentiment prediction is included only when a trained `best_review_sentiment_pipeline.joblib` artifact is present.
- **No fabricated metrics**: no model artifact means no review-level prediction. Missing API configuration or inference errors are surfaced to the user.

## Run the web application

Node.js 20 or newer is recommended.

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Run the Python inference service

Use the existing Python setup from the main README, then install the added API dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API exposes `GET /health` and `POST /analyze`. Supply aspect terms in the UI to avoid automatic spaCy extraction; otherwise the spaCy English model must be installed.

To enable review-level prediction, first train the existing review classifier using the documented Amazon dataset workflow. It writes `outputs/best_review_sentiment_pipeline.joblib`. Keep this artifact on the Python service host, or set `ABSA_REVIEW_MODEL_PATH` to its location. The artifact is deliberately not committed to Git.

## Connect Node.js to Python

Set this environment variable for the Next.js server:

```env
ABSA_API_URL=http://localhost:8000
```

For deployment, set `ABSA_API_URL` to the reachable base URL of your deployed FastAPI service (without a trailing slash). Do not use a browser-only URL or put secrets in a `NEXT_PUBLIC_*` variable. Deploy the Python service separately from the Next.js app, because the Python/scikit-learn/spaCy runtime and trained artifacts are not part of a standard Next.js deployment.

## Current model scope

- Aspect extraction uses the existing spaCy noun-phrase/fallback extraction code unless aspects are supplied manually.
- Aspect polarity uses the existing **lexicon/proximity heuristic**. It is a transparent demo baseline, not a supervised SemEval aspect-polarity model.
- Optional review-level classification uses the trained scikit-learn pipeline if the artifact is present. Its labels are review-level and must not be interpreted as aspect-level predictions.
- The SemEval supervised experiment and its reported benchmark results are separate from live inference. This UI does not present benchmark scores as predictions.
