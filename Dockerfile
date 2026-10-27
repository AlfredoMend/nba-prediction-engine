# NBA Prediction Engine - Streamlit UI + pipeline in one image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp is required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8501
# Render/Heroku-style platforms inject $PORT; default to 8501 locally.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
