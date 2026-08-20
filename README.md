# Veloit RAG Service

## Docker Deploy

Deploy the full application stack (Backend, Frontend, and Ollama) on a Linux server with Docker Compose.

### Quick Start (One Command)

```bash
docker compose up --build -d
```

### Pull Ollama Model (First Time Setup)

Once the containers are running, pull the LLM model inside the running Ollama container:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```

### Useful Management Commands

- **Check Service Status**:
  ```bash
  docker compose ps
  ```

- **View Logs**:
  ```bash
  docker compose logs -f
  ```

- **Stop Services**:
  ```bash
  docker compose down
  ```

- **Rebuild and Restart**:
  ```bash
  docker compose up --build -d
  ```

### Port Mapping Summary

- **Backend API**: `http://<server-ip>:8000` (Health Check: `http://<server-ip>:8000/health`)
- **Frontend App**: `http://<server-ip>:3000`
- **Ollama LLM**: `http://<server-ip>:11434`
