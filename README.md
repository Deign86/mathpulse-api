---
title: MathPulse API
emoji: 🧮
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# MathPulse AI - ML Backend

Python backend providing ML-powered educational features using Hugging Face.

## Features

- **AI Math Tutor**: Conversational AI for helping students with math concepts
- **Risk Prediction**: ML-based student risk assessment using zero-shot classification  
- **Learning Path Generation**: AI-powered personalized learning recommendations
- **Class Analytics**: Daily insights and trend analysis

## Quick Start

### 1. Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Hugging Face token
# Get token from: https://huggingface.co/settings/tokens
```

### 3. Run the Server

```bash
python main.py
```

Or use the batch script (Windows):
```bash
run.bat
```

Server starts at: http://localhost:8000

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status

### AI Tutor
- `POST /api/chat` - Chat with AI math tutor
  ```json
  {
    "message": "Explain the chain rule",
    "conversationHistory": []
  }
  ```

### Risk Prediction
- `POST /api/predict-risk` - Single student risk prediction
  ```json
  {
    "id": "1",
    "name": "Sarah Chen",
    "engagementScore": 45,
    "avgQuizScore": 58,
    "weakestTopic": "Calculus - Derivatives"
  }
  ```

- `POST /api/predict-risk/batch` - Batch risk prediction

### Learning Paths
- `POST /api/learning-path` - Generate personalized learning path
  ```json
  {
    "studentId": "1",
    "weakestTopic": "Calculus - Derivatives",
    "avgQuizScore": 58,
    "engagementScore": 45
  }
  ```

- `GET /api/learning-path/{topic}` - Quick path by topic

### Analytics
- `POST /api/analytics/daily-insight` - Class-wide daily insight
- `GET /api/analytics/summary` - Available features

## Hugging Face Models Used

- **Chat/Generation**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Classification**: `facebook/bart-large-mnli` (zero-shot)

## Fallback Behavior

If the Hugging Face API is unavailable or no token is configured, the backend automatically falls back to rule-based responses, ensuring the app always works.

## Development

API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
