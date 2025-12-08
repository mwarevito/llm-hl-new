# ☁️ Cloud Deployment Guide

This guide explains how to deploy the Hyperliquid Trading Bot to a cloud server (VPS) using Docker.

## Prerequisites

1.  **A Cloud Server (VPS)**:
    *   Providers: AWS (EC2), DigitalOcean (Droplet), Hetzner, Vultr, etc.
    *   OS: Ubuntu 22.04 LTS (Recommended)
    *   Specs: 1 vCPU, 1GB RAM is sufficient.

2.  **Docker & Docker Compose**: Installed on the server.

## Step 1: Prepare the Server

SSH into your server:
```bash
ssh root@your-server-ip
```

Install Docker and Docker Compose (if not already installed):
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
docker compose version
```

## Step 2: Deploy the Bot

1.  **Create a directory for the bot**:
    ```bash
    mkdir -p ~/hyperliquid-bot
    cd ~/hyperliquid-bot
    ```

2.  **Copy files**:
    You need to copy the following files from your local machine to the server. You can use `scp` or simply create them on the server using `nano`.

    Files needed:
    *   `Dockerfile`
    *   `docker-compose.yml`
    *   `requirements.txt`
    *   `agent_improved.py`
    *   `.env` (Make sure this has your REAL keys!)

    **Option A: Using SCP (run this from your LOCAL machine)**
    ```bash
    scp Dockerfile docker-compose.yml requirements.txt agent_improved.py .env root@your-server-ip:~/hyperliquid-bot/
    ```

    **Option B: Copy/Paste**
    Use `nano filename` to create each file and paste the content.

## Step 3: Start the Bot

Run the bot in the background using Docker Compose:

```bash
docker compose up -d --build
```

The bot will start in **CONTINUOUS** mode (because `RUN_MODE=continuous` is set in `docker-compose.yml`).

## Step 4: Monitor

**View Logs:**
```bash
docker compose logs -f
```

**Check Status:**
```bash
docker compose ps
```

**Stop the Bot:**
```bash
docker compose down
```

## Maintenance

**Update the Bot:**
1.  Upload new code (e.g., `agent_improved.py`).
2.  Rebuild and restart:
    ```bash
    docker compose up -d --build
    ```

**Backup Data:**
The `data/` directory contains your trade history and state.
```bash
# Copy data back to local machine
scp -r root@your-server-ip:~/hyperliquid-bot/data ./backup_data
```

## Option 2: Deploy on Railway (PaaS)

Railway is an easier alternative to a VPS.

### 1. Setup
1.  Push your code to a **GitHub Repository**.
2.  Sign up at [Railway.app](https://railway.app/).

### 2. Create Project
1.  Click **"New Project"** -> **"Deploy from GitHub repo"**.
2.  Select your repository.
3.  Railway will automatically detect the `Dockerfile` and start building.

### 3. Configure Variables
1.  Go to the **"Variables"** tab in your Railway project.
2.  Add all variables from your `.env` file:
    *   `HYPERLIQUID_ADDRESS`
    *   `HYPERLIQUID_PRIVATE_KEY`
    *   `OPENAI_API_KEY` (or Anthropic)
    *   `RUN_MODE` = `continuous`

### 4. ⚠️ CRITICAL: Persist Data
Railway files are ephemeral (deleted on restart). You **MUST** create a Volume to save your trade history.

1.  Click on your **Service** (the card with your repo name) in the Railway canvas.
2.  Look at the tabs at the top (Deployments, Variables, Settings, etc.). Click **"Volumes"**.
    *   *Note: If you don't see "Volumes", check under "Settings" or look for a "Storage" section.*
3.  Click **"Add Volume"** (or "New Volume").
    *   **Pro Tip**: Press `Cmd + K` (Mac) or `Ctrl + K` (Windows) in Railway and type "Add Volume" to find it instantly.
3.  Mount path: `/app/data`
4.  This ensures `risk_state.json` and trade history are saved even if the bot restarts.

### 5. Deploy
Railway will automatically redeploy when you push changes to GitHub.

### Troubleshooting: "Healthcheck Failed"
If you see "Healthcheck failed", it's because Railway expects a web server by default.
1.  Go to **Settings** -> **Deploy**.
2.  Remove the **Healthcheck Path** (make it empty).
3.  Or, ensure your `railway.toml` does NOT have `healthcheckPath` defined.


