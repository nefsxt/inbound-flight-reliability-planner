# Aegean Inbound Reliability Planner
#
# This single image runs both the CLI pipeline steps (fetch/features/train)
# and the Streamlit dashboard. Which one happens depends on the command you
# pass at `docker run` / `docker compose run` time -- see README.md.
FROM python:3.11-slim

# xgboost needs libgomp at runtime; keep the rest minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/models are meant to be bind-mounted from the host (see
# docker-compose.yml) so pipeline output survives container restarts and
# rebuilds -- these mkdirs just make sure the paths exist even if you run
# the plain `docker run` form without the compose volumes.
RUN mkdir -p data/raw data/processed models

EXPOSE 8501

# Default: launch the dashboard. Override the command to run pipeline
# steps instead, e.g.:
#   docker compose run --rm app python -m src.fetch_opensky
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
