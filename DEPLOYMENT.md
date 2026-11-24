# 🚀 Production Deployment Guide: Korba PM2.5 & PM10 Air Quality Forecasting System

This project is built with **PyTorch**, **Streamlit**, and **ReportLab**. It provides:
1. **Interactive Guided Web Dashboard** (Streamlit on `app/dashboard.py`)
2. **Serverless Web Portal & REST API** (Vercel on `api/index.py` via `vercel.json`)
3. **Containerized Production Microservice** (Docker on `Dockerfile` & `Procfile`)

---

## ⚡ Deployment Options Overview

| Platform | Best For | Persistent WebSockets | Setup Complexity | Cost |
| :--- | :--- | :---: | :---: | :---: |
| **Streamlit Community Cloud** | Full interactive 4-stage UI & charts | ✅ Full Support | 1-Click (30 seconds) | Free |
| **Vercel** | Serverless REST API & Web Landing Portal | ❌ Ephemeral Request/Response | Git Push (Instant) | Free |
| **Render / Railway / Fly.io** | Production Docker container | ✅ Full Support | 1-Click Git Connect | Free tier / Pay as you go |
| **Docker Self-Hosted** | On-premise industrial server | ✅ Full Support | `docker run -p 8501:8501` | Self-hosted |

---

## Option 1: Deploy to Vercel (Serverless Web Portal & REST API)

Vercel is pre-configured with `vercel.json` and `api/index.py`.

### Steps to Deploy on Vercel:
1. **Push your code to GitHub / GitLab / Bitbucket**:
   ```bash
   git add .
   git commit -m "Production release v1.0"
   git push origin main
   ```
2. **Import into Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in.
   - Click **Add New...** → **Project**.
   - Select your GitHub repository.
   - Framework Preset: **Other**.
   - Click **Deploy**.
3. **Vercel automatically detects `vercel.json`**:
   - Deploys the web portal at your production domain (`https://your-project.vercel.app`).
   - Serves live JSON REST APIs:
     - `https://your-project.vercel.app/api/health`
     - `https://your-project.vercel.app/api/stations`
     - `https://your-project.vercel.app/api/info`

> [!NOTE]
> **Vercel Architectural Context**: Vercel is built for stateless, short-lived serverless lambdas. Streamlit requires a persistent, long-running WebSocket connection to manage interactive user sessions (such as reactive file uploads and slider reruns). The included `api/index.py` provides a serverless web portal and REST API on Vercel. For the live interactive Streamlit UI, deploy using **Option 2 (Streamlit Cloud)** or **Option 3 (Docker / Render)**.

---

## Option 2: Deploy to Streamlit Community Cloud (Recommended & 100% Free)

Streamlit Community Cloud is the official, zero-config hosting platform for Streamlit applications. It natively supports persistent WebSockets, large file uploads, and PyTorch deep learning inference.

### Steps:
1. **Push your repository to GitHub**:
   Ensure `requirements.txt`, `app/dashboard.py`, and `models/model_package.pt` are pushed.
2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account.
   - Click **New app**.
   - Select your repository and branch (`main`).
   - Main file path: `app/dashboard.py`.
   - Click **Deploy!**
3. **Live in ~60 seconds**:
   Your app will be live with a public URL like `https://korba-aqi-forecast.streamlit.app`.

---

## Option 3: Deploy with Docker (Render, Railway, Fly.io, Cloud Run)

The repository includes an optimized multi-stage `Dockerfile` and `Procfile`.

### Local Docker Build & Run:
```bash
# Build the production image
docker build -t korba-aqi-forecaster:v1.0 .

# Run container on port 8501
docker run -d -p 8501:8501 --name korba-forecaster korba-aqi-forecaster:v1.0

# Open in browser:
# http://localhost:8501
```

### Deploy to Render:
1. Go to [render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** as the environment (Render automatically uses the included `Dockerfile`).
5. Click **Create Web Service**.

### Deploy to Railway:
1. Go to [railway.app](https://railway.app).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Railway automatically detects `Dockerfile` and binds the `$PORT` dynamically.

---

## Option 4: Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Choose **Streamlit** SDK.
4. Push your repository to the Hugging Face Space git remote.

---

## 🛠️ Included Production Files in Repository

- `Dockerfile`: Production-hardened container definition with lightweight CPU PyTorch.
- `.dockerignore`: Excludes virtualenvs, cache, and test artifacts.
- `Procfile`: Dynamic port binding for PaaS environments (`$PORT`).
- `.streamlit/config.toml`: Production server parameters (CORS, headless, memory limit).
- `vercel.json`: Vercel serverless routing configuration.
- `api/index.py`: Serverless REST API and landing portal for Vercel.
- `models/model_package.pt`: Pre-trained multi-station model bundle (383 KB).
- `tests/`: 41 automated unit, integration, and robustness tests (`pytest` 100% passing).
