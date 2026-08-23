<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Veloitt RAG Assistant

A Docker-based AI assistant for answering questions about Veloitt and the eTimeTracker platform using a grounded RAG pipeline. This guide explains how to take the repo, run it on a server, and get the bot working with the least amount of friction.[^1]

## What this project includes

This project runs as three connected services:

- Backend — FastAPI API for retrieval and answer generation.
- Frontend — web app UI for chatting with the bot.
- Ollama — local LLM runtime used by the backend.

The backend reads its prepared knowledge from `output/chunks.jsonl` and `qdrant_data`, and the current Docker Compose setup mounts those folders into the backend container at runtime.[^1]

## Server prerequisites

Before starting, make sure the server has:

- Git
- Docker
- Docker Compose plugin

The deployment flow assumes the repo can be cloned onto the server and then started with Docker commands.[^1]

## Deployment flow

### 1. Clone the repo

```bash
git clone <repo-url>
cd Veloit-RAG
```

If the code is in an organization repo, clone that repo directly. If you forked it first, clone your fork or the org repo depending on your team workflow.

### 2. Start the full stack

Run the full project with one command:

```bash
docker compose up --build -d
```

This builds the backend image from the root `Dockerfile`, builds the frontend image from `frontend/Dockerfile`, and starts the Ollama service alongside them.[^1]

### 3. Pull the Ollama model

After the services are up, pull the model required by the backend:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```

The backend is currently configured to use `qwen2.5:1.5b`, so this step is required before the bot can answer properly.[^1]

### 4. Check service health

Check that all containers are running:

```bash
docker compose ps
```

Then test the backend health endpoint:

```bash
curl http://localhost:8000/health
```

A healthy response confirms that the backend is up, the API is reachable, and the bot service is ready. The backend healthcheck in Compose also uses this same `/health` route.[^1]

### 5. Open the app

Once the stack is running:

- Frontend: `http://<server-ip>:3000`
- Backend health: `http://<server-ip>:8000/health`

The frontend talks to the backend internally through Docker networking using `http://backend:8000`.[^1]

## How the Docker setup works

The current Compose setup runs three services:


| Service | Role | Port |
| :-- | :-- | :-- |
| backend | FastAPI API and RAG logic | 8000 [^1] |
| frontend | Chat UI | 3000 [^1] |
| ollama | Local model runtime | internal service, persistent model storage [^1] |

The backend is configured with:

- `OLLAMA_BASE_URL=http://ollama:11434`
- `OLLAMA_MODEL=qwen2.5:1.5b`
- `CHUNKS_FILE=/app/output/chunks.jsonl`
- `QDRANT_PATH=/app/qdrant_data`[^1]

The backend also mounts:

- `./output:/app/output`
- `./qdrant_data:/app/qdrant_data`

That means the bot reads its prepared chunks and vector data from the server’s local project folders, which keeps the data persistent across container rebuilds.[^1]

## Typical server workflow

A teammate deploying this on a server usually only needs these commands:

```bash
git clone <repo-url>
cd Veloit-RAG
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose ps
curl http://localhost:8000/health
```

That is the shortest path from repo to working bot.[^1]

## Updating the deployment

When the repo gets new commits:

```bash
git pull
docker compose up --build -d
```

This rebuilds the containers with the latest code and restarts the services.

If only the model is missing on a new server, pull it again:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```


## Logs and troubleshooting

### Check all services

```bash
docker compose ps
```


### View backend logs

```bash
docker compose logs -f backend
```


### View frontend logs

```bash
docker compose logs -f frontend
```


### View Ollama logs

```bash
docker compose logs -f ollama
```


### Restart everything

```bash
docker compose down
docker compose up --build -d
```


## Important notes

- The backend depends on the Ollama service being available internally through Docker.[^1]
- The frontend depends on the backend becoming healthy before it starts fully.[^1]
- The current setup exposes the frontend on port 3000 and backend on port 8000.[^1]
- The prepared data lives in `output/` and `qdrant_data/`, which the backend mounts at runtime.[^1]


## In simple terms

The deployment flow is:

1. Clone the repo onto the server.
2. Start Docker Compose.
3. Pull the Ollama model.
4. Confirm `/health` works.
5. Open the frontend and test the bot.

That’s it — once those steps pass, the bot is live on the server.

<div align="center">⁂</div>

