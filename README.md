---
title: SOC-AgentLab
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.11.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# SOC-AgentLab

OpenEnv-style environment for evaluating AI agents performing SOC alert triage.

## Overview
Logs → Threat Analyzer → SOC Environment → Agent Action → Reward

## Setup
pip install -r requirements.txt

## Environment Variables
API_BASE_URL=${API_BASE_URL:-https://router.huggingface.co/v1}
HF_TOKEN=${HF_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY} (supported fallback)
API_KEY=${API_KEY} (optional fallback when HF_TOKEN is unset)
MODEL_NAME=${MODEL_NAME:-soc-agentlab-rule}
LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME}

## Run
python inference.py
python benchmark.py
python app.py

## Inference Modes
- `python inference.py` runs a short local rollout and exits (evaluation-friendly).
- `python inference.py --serve` starts the FastAPI server mode.

## Validate
python validate_env.py

## Docker
docker build -t soc-agentlab .
docker run -p 7860:7860 soc-agentlab