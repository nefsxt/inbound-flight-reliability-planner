# 🛬 Inbound Flight-Time Anomaly Prediction

A proof-of-concept / minimum viable product (MVP) decision-support tool for ground-operations buffer planning on inbound flights. Instead of predicting a single point estimate for delays, this tool generates a range of probabilistic arrival outcomes (**Typical / Cautious / Safe**) derived from historical flight records and real-time weather forecasts. This allows stakeholders to evaluate live conditions against trained models before deciding exactly how much buffer time to allocate.

---

## ⏱️ What the App Predicts & How

### The Target Variable: Flight Duration Anomaly
The model does not measure "delay vs. scheduled commercial time" because the pipeline utilizes **OpenSky Network transponder data** rather than proprietary airline scheduling systems. Instead, the model targets the **Flight Duration Anomaly**, calculated as:

`Flight Duration Anomaly = Actual Gate-to-Gate Duration - Route's Historical Median`

By benchmarking every flight against its historical median duration over a specific past collection period, "on time" is defined as meeting or beating typical historical performance. 

More specifically the historical median provides a route-specific reference for typical observed flight duration. The resulting anomaly measures deviation from that reference, rather than delay against an airline's scheduled arrival time. Consequently, the model supports planning around deviations in observed flight duration. It does not independently establish whether an inbound flight will meet its published commercial schedule or an airport-specific operational deadline.


### Multi-Quantile Decision Support
To help operations teams balance the cost of wasting ground time against the risk of late arrivals, the system outputs three distinct quantiles via **XGBoost Quantile Regression**:

* **P50 (Typical Baseline):** The central conditional estimate of the flight-duration anomaly distribution.

* **P90 (Cautious / 90% Target Quantile):** A conditional quantile intended to cover approximately 90% of observed flight-duration anomalies under appropriate calibration. It provides a more conservative planning reference than P50, but does not guarantee a 90% operational reliability level.

  * *e.g: A predicted P90 anomaly of 14 minutes indicates that the model estimates approximately 90% of comparable anomalies to fall at or below 14 minutes, subject to calibration and evaluation conditions.*

* **P95 (Safe / 95% Target Quantile):** A higher conditional quantile intended to cover approximately 95% of observed flight-duration anomalies under appropriate calibration. It provides a more conservative planning reference than P90, but does not guarantee a 95% operational reliability level.


---

## Model Hierarchy & Architecture

The application automatically trains two tiers of quantile regression models **per specific route**:

* **Global Model:** A pooled model trained across data from all carriers flying that route, utilizing the specific *airline* as an input feature.
* **Per-Airline Models:** A dedicated, hyper-localized model trained exclusively on a single carrier's data, provided they meet the minimum required historical data threshold defined in `config.MIN_ROWS_FOR_MODEL_FEASIBILITY`.

🔄 **Fallback Logic:** If a dedicated airline model is unavailable due to insufficient historical data, the Live Predictor automatically falls back to the route's **Global Model**, ensuring continuous decision support.

> 📖 *For an in-depth breakdown of the operations research context (including a Newsvendor Problem analogy), pipeline execution, and infrastructure selection and design, see `ARCHITECTURE.md`.*



## 🛑 Current Limitations 
The pipeline and app are designed to support adding more
routes without code changes (see `config.ROUTES`).

However, this tool is intentionally restricted to a single high-density route (**Frankfurt [EDDF] → Thessaloniki [LGTS]**) with 3 years of historical data due to the following strict data infrastructure barriers:

### 1. Free Tier OpenSky Restrictions 
The [OpenSky Network](https://opensky-network.org) restricts direct access to its comprehensive historical database exclusively to formal research institutions and government bodies. To bypass this for a personal project, the training data has to be compiled via the available public endpoints and pre-processed scientific flight slices. This constraint directly cascades into the next bottleneck.

### 2. The Multi-Threading / Concurrency Wall (Rate-Limiting)
Because the pipeline relies on the free/anonymous tier of `OpenSky` and `OpenMeteo`, the application faces strict request volume caps. 
* **Structural Barriers:** The data pipeline is designed to execute requests sequentially to operate safely within the rate limits of the free API tiers. Concurrent execution, such as running massive historical backfills alongside live daily updates, is intentionally disabled. Attempting parallel data pulls risks exceeding the project’s request budget, which can lead to API IP throttling, token exhaustion, or connection drops.
* **Current Mitigation:** The current architecture is designed to execute sequentially, handling one data pipeline step and one route at a time to remain compliant with external API terms. This bottleneck makes the simultaneous implementation of multiple routes highly impractical and heavily reliant on manual, offline data collection.



## 📊 Data sources (free, no paid keys)

| Source | Used for |
|---|---|
| [OpenSky Network REST API](https://openskynetwork.github.io/opensky-api/rest.html) | Historical arrival/departure times, callsign → airline |
| [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | Historical weather at both airports (training data) |
| [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | Live weather forecast (Live Predictor tab) |

## 📱The app

Two tabs:

1. **Dashboard**: pick a route, see the global model's results first, then
   each dedicated per-airline model's results (for airlines with enough data only).
2. **Live Predictor**: pick a route, hour, and airline; today's Open-Meteo
   forecast is fetched (once per day, cached) and fed into the matching
   trained model for a live buffer recommendation.

## Use the live app online

**Live app:** [https://inbound-flight-time-anomaly-prediction.streamlit.app](https://inbound-flight-time-anomaly-prediction.streamlit.app/)


## Use it via Docker

Requires Docker and Docker Compose.

```bash
# Create a .env file with (only needed for the pipeline steps, not the dashboard):
#   OPENSKY_CLIENT_ID=...
#   OPENSKY_CLIENT_SECRET=...
#   HF_TOKEN=...              # only if you want to push/pull from Hugging Face
#   DATA_SOURCE=local         # default; set to "hf" to sync from HF instead

docker compose build
docker compose up      # dashboard at http://localhost:8501
```

`data/` and `models/` are bind-mounted from the host, so pipeline output
and OpenSky credit spend survive rebuilds. With no data/models present yet,
the Dashboard and Live Predictor tabs will say so per route rather than
show anything fabricated.

Run pipeline steps as one-off containers:

```bash
docker compose run --rm app python -m src.fetch_opensky
docker compose run --rm app python -m src.fetch_weather
docker compose run --rm app python -m src.features
docker compose run --rm app python -m src.train_model
docker compose run --rm app pytest tests/ -v
```

## Use it locally (no Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root (loaded automatically via
`python-dotenv`):

```
OPENSKY_CLIENT_ID=your_client_id
OPENSKY_CLIENT_SECRET=your_client_secret
```

Get free OpenSky credentials at [opensky-network.org](https://opensky-network.org).

Run the pipeline in order, then launch the dashboard:

```bash
python -m src.fetch_opensky      # 1. raw flight data
python -m src.fetch_weather      # 2. matching historical weather
python -m src.features           # 3. build the feature matrix
python -m src.train_model        # 4. train all model tiers
streamlit run app/streamlit_app.py
```

## Keeping data and models fresh (Docker or No Docker)

- **`daily-update.yml`**: runs `src.update_data` once a day, fetching only
  what's missing since the last stored date, then pushes to Hugging Face.
  A freshness check (via `src.hf_storage`, listing remote file metadata —
  no data download) skips the whole cycle on a route that's already current.
- **`monthly-retrain.yml`**: rebuilds the feature matrix and retrains all
  model tiers monthly. This is also the **manual training trigger**: use
  the "Run workflow" button in GitHub Actions (optionally enabling
  hyperparameter tuning) for an initial deploy or an on-demand retrain.

Both require `HF_TOKEN` (and `daily-update.yml` also needs
`OPENSKY_CLIENT_ID`/`OPENSKY_CLIENT_SECRET`) as GitHub Actions repository
secrets. See `ARCHITECTURE.md` for the full pipeline design.

## Testing

```bash
pytest tests/ -v
```
For an individual test (example):
```bash
pytest tests/test_fetch_opensky.py
```

