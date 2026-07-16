# Production Deployment — Design

## Scope

Harden the Docker setup for production, consolidate to a single service, and add CI/CD via GitHub Actions.

## Changes

### 1. Single Dockerfile (multi-stage)

```
Stage 1 (node:20-alpine):  npm install + npm run build → frontend/dist/
Stage 2 (python:3.11-slim): copy backend/, copy frontend/dist/ → serve by FastAPI
```

### 2. Backend serves frontend

Add to `backend/main.py`:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

This serves the SPA at `/` while `/api/` routes still hit the API.

### 3. docker-compose.prod.yml

- Single `app` service (no separate frontend container)
- `restart: unless-stopped`
- `healthcheck` — curl `/api/health`
- `logging` — `max-size: "10m"`, `max-file: "3"`
- No `--reload`
- `ports: ["80:8000"]`

### 4. .dockerignore

- `node_modules/`, `.venv/`, `__pycache__/`, `.git/`, `.env`, `*.md`, `.vscode/`

### 5. CI/CD (GitHub Actions)

`.github/workflows/deploy.yml`:
- Trigger: push to `main`
- Steps: checkout → build Docker image → scp to server → SSH restart

Uses `docker save | ssh` or a registry. Simple deploy script pattern.

## Files changed/created

| File | Action |
|------|--------|
| `Dockerfile` | Create (project root, multi-stage) |
| `.dockerignore` | Create |
| `docker-compose.prod.yml` | Create |
| `.github/workflows/deploy.yml` | Create |
| `backend/main.py` | Add StaticFiles mount |
| `backend/Dockerfile` | Delete |
| `frontend/Dockerfile` | Delete |
| `docker-compose.yml` | Keep (dev use) |

## Non-goals

- No nginx reverse proxy (FastAPI handles it fine for this scale)
- No SSL cert management (assumes server-level reverse proxy or self-managed)
- No secrets management (API has no secrets beyond what's in env vars)
- No database (the app is stateless — no data to persist)
