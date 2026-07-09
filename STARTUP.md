# RAG LLM Startup Guide

This guide will walk you through launching the local RAG LLM application.

## Quick Start (Single Command / Click)

You can launch the entire stack (Redis, Backend, Frontend, and Browser Window) at once:

* **Via Double-Click (Windows Explorer)**: Double-click the `start-app.bat` file in the root folder.
* **Via PowerShell (Terminal)**:
  ```powershell
  .\start-app.ps1
  ```

---

## Prerequisites

1. **Ollama**: Ensure Ollama is running and has the models pulled:
   ```powershell
   ollama pull nomic-embed-text
   ollama pull qwen2.5:0.5b
   ```
2. **Redis**: Ensure a Redis server is installed and running on port `6379`.
   - **Default installation folder**: `C:\Program Files\Redis`
   - **Default config file**: `redis.windows.conf` (in the installation folder)

---

## Step 1: Start Redis Server

Depending on how you installed Redis:

* **If installed via winget (default location):**
  Open a command prompt or PowerShell and start the server:
  ```powershell
  redis-server
  ```
  *(If the command is not recognized, run it directly from its install folder:)*
  ```powershell
  & "C:\Program Files\Redis\redis-server.exe"
  ```
* **Verify Redis connection:**
  To check if Redis is active, you can test it with `redis-cli`:
  ```powershell
  redis-cli ping
  # Expected output: PONG
  ```

---

## Step 2: Start the Backend (FastAPI)

1. Open a new terminal window.
2. Activate your conda environment:
   ```powershell
   conda activate RAG
   ```
3. Navigate to the backend directory and launch the server:
   ```powershell
   cd backend
   uvicorn main:app --reload --port 8000
   ```
4. Verify the backend is up by opening the interactive API docs:
   - [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Step 3: Start the Frontend (React / Vite)

1. Open a third terminal window.
2. Activate the conda environment:
   ```powershell
   conda activate RAG
   ```
3. Navigate to the frontend directory and start the dev server:
   ```powershell
   cd frontend
   npm run dev
   ```
4. Launch the application UI in your web browser:
   - [http://localhost:5173](http://localhost:5173)

---

## Summary of Ports & Links

| Component | Port | Link |
|-----------|------|------|
| **Frontend Web UI** | `5173` | [http://localhost:5173](http://localhost:5173) |
| **Backend API Docs** | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Redis Server** | `6379` | `localhost:6379` |
