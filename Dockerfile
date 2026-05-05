# ── Image de base ─────────────────────────────────────────────────────────────
# Python 3.11 slim = légère, compatible avec nos packages
FROM python:3.11-slim

# ── Métadonnées ────────────────────────────────────────────────────────────────
LABEL maintainer="Belvin Tsadjio"
LABEL description="Churn Prediction API"
LABEL version="1.0.0"

# ── Dossier de travail dans le container ──────────────────────────────────────
WORKDIR /app

# ── Copie des dépendances en premier (optimise le cache Docker) ───────────────
COPY requirements.txt .

# ── Installation des dépendances ──────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code ─────────────────────────────────────────────────────────────
COPY app/main.py .

# ── Copie des artefacts depuis models/ ────────────────────────────────────────
COPY models/model.pkl .
COPY models/scaler.pkl .
COPY models/ordinal_encoder.pkl .
COPY models/feature_names.json .

# ── Port exposé par l'API ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Commande de lancement ─────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]