# 🍱 DabbaETA — Local-First Delivery ETA & AI Dispatch Assistant

> **DEV Hackathon Challenge** — Open-Source AI ("Side Hustle" Theme)  
> *A local-first, privacy-preserving delivery prediction and dispatch assistant built for neighborhood tiffin services, home bakeries, cloud kitchens, and local couriers.*

---

## 🌟 Overview

Small local delivery businesses like daily tiffin (dabba) providers and boutique bakeries operate under razor-thin margins and strict delivery windows. A 20-minute delay due to sudden evening monsoon rain or traffic gridlock can mean a cancelled subscription or an upset customer.

**DabbaETA** solves this with an end-to-end, local-first AI stack:
1. **Accurate Delivery ETAs**: TabPFN (Prior-Fitted Networks) and Gradient Boosting baselines predict realistic transit times and late-risk probabilities.
2. **Autonomous Dispatch Assistant**: Gemma (running locally via Ollama) monitors the pending delivery queue, answers dispatch queries in Hinglish/Hindi/English (*"Aaj kaunsi orders late hongi?"*), and drafts courteous WhatsApp delay messages.
3. **Resilient Orchestration**: Temporal Python SDK coordinates batch dispatch workflows (`load -> predict -> flag -> explain -> draft`) with exponential retries and failure recovery.
4. **End-to-End Observability**: Custom Sentry spans trace every LLM and TabPFN call, tagging latency and token throughput.
5. **Zero Data Leakage**: Customer names, phone numbers, and home addresses never leave the local laptop.

---

## 🏗️ Architecture & Technology Stack

```
                     ┌───────────────────────────────────────────────┐
                     │          Browser Dashboard (HTML/JS)          │
                     │  - Orders Queue & Risk Badges                 │
                     │  - Hinglish Dispatch Chat (Gemma)             │
                     │  - TabPFN Benchmarks & Temporal Modal         │
                     └──────────────────────┬────────────────────────┘
                                            │ HTTP / JSON
                                            ▼
                     ┌───────────────────────────────────────────────┐
                     │           FastAPI Backend (Python 3.12)       │
                     │           - POST /upload  (SQLite / CSV)      │
                     │           - POST /predict (ETA & Risk)        │
                     │           - GET  /benchmark (80/20 & small-N) │
                     │           - POST /chat    (Gemma Agent)       │
                     │           - POST /run-dispatch (Temporal)     │
                     └───────┬──────────────┬──────────────┬─────────┘
                             │              │              │
         ┌───────────────────┘              │              └────────────────────┐
         ▼                                  ▼                                   ▼
┌─────────────────┐             ┌─────────────────────┐             ┌───────────────────────┐
│     TabPFN      │             │    Gemma / Ollama   │             │   Temporal SDK Worker │
│  (Prior Labs)   │             │ (gemma3:4b via HTTP)│             │  (Task Queue Dispatch)│
│  - Cloud Client │             │  - Structured JSON  │             │  - 5-Activity Chain   │
│  - Local PFN    │             │  - Tool Routing     │             │  - Retry Policies     │
│  - Sentry Spans │             │  - Hinglish Drafts  │             │  - Worker Kill Resume │
└─────────────────┘             └─────────────────────┘             └───────────────────────┘
```

- **Backend**: Python 3.12, FastAPI, SQLite / Pandas, Pydantic v2
- **Prediction**: TabPFN (`tabpfn-client` / local `tabpfn`), XGBoost, and Scikit-Learn LinearRegression
- **LLM**: Gemma via Ollama (`gemma3:4b` default over `http://localhost:11434`), JSON-structured tool routing with retries
- **Orchestration**: Temporal Python SDK (`DispatchWorkflow` with exponential backoff)
- **Observability**: `sentry-sdk` with custom spans `llm.gemma` and `tabpfn.predict` (tags: `tokens_in`, `tokens_out`, `latency_ms`)
- **Frontend**: Clean single-page dashboard with glassmorphism, live queue, chat, benchmark charts, and WhatsApp modal

---

## 📊 Benchmark Results

Evaluated on the 5,000-row delivery dataset (80% train / 20% test split, plus small-data subsampling experiments on 50, 100, and 200 rows):

### 1. Full Dataset (80/20 Split — 4,000 Train, 1,000 Test)

| Model | MAE (Minutes) | RMSE | R² Score | Fit Latency (s) | Predict Latency (s) |
|---|---|---|---|---|---|
| **TabPFN (Prior-Fitted Network)** | **3.60 min** | 6.05 | **0.937** | 0.394s | 0.004s |
| **XGBoost Regressor** | **3.31 min** | 5.22 | **0.953** | 0.113s | 0.006s |
| **Linear Regression** | 6.27 min | 10.15 | 0.822 | 0.003s | < 0.001s |

### 2. Small-Data Experiments (Small Kitchen / Bakery Regime)

| Sample Size ($N$) | TabPFN MAE | XGBoost MAE | Linear Regression MAE | TabPFN R² |
|---|---|---|---|---|
| **N = 50** | 11.56 min | 10.20 min | 6.81 min | 0.428 |
| **N = 100** | 9.76 min | 8.65 min | 5.97 min | 0.504 |
| **N = 200** | 8.33 min | 7.08 min | 7.23 min | 0.605 |

*Benchmark visualization automatically saved to [`docs/benchmark.png`](docs/benchmark.png) and metrics to [`docs/benchmark.json`](docs/benchmark.json).*

---

## 🔒 Why Open & Local-First?

1. **Strict Customer Data Privacy**: Tiffin and bakery orders contain sensitive personally identifiable information (PII) — customer home addresses, gate codes, phone numbers, and dietary routines. DabbaETA ensures customer PII is kept strictly in local SQLite storage. Only anonymized numeric feature vectors ever hit the TabPFN interface.
2. **Zero Recurring LLM API Costs**: By running Gemma locally via Ollama, small business operators can query dispatch status and generate unlimited customer delay messages with **$0 cloud API bills**.
3. **Swappable Model Architecture**: Through the `OLLAMA_MODEL` environment variable, operators can switch seamlessly between `gemma3:4b`, `gemma:2b`, or quantized GGUFs to fit their laptop's hardware specs (Mac M-series, Nvidia RTX, or standard CPU).

---

## 🚀 Setup & How to Run

### Prerequisites
- Python 3.11 or 3.12
- [Ollama](https://ollama.ai) (optional, for local Gemma execution)
- [Temporal CLI](https://temporal.io) (optional, for local dev workflow orchestration)

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
# Clone
git clone https://github.com/your-username/dabbaeta.git
cd dabbaeta

# Create & activate venv
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key variables:
```ini
TABPFN_API_KEY=your_tabpfn_api_key_here
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
TEMPORAL_HOST=localhost:7233
SENTRY_DSN=
```

### 3. Generate the Dataset
Create the realistic 5,000-row delivery dataset:
```bash
python scripts/make_dataset.py
# or: make dataset
```

### 4. Run Ollama (Gemma Model)
Pull and run Gemma locally:
```bash
ollama pull gemma3:4b
ollama serve
```

### 5. Start Temporal Dev Server (Optional for Distributed Worker)
```bash
temporal server start-dev
```

### 6. Start Temporal Background Worker
In a separate terminal:
```bash
python workflows/worker.py
# or: make worker
```

### 7. Start the FastAPI Dev Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# or: make dev
```
Open **`http://localhost:8000`** in your browser to access the DabbaETA dashboard.

---

## 🧪 Running Tests

The test suite validates schema constraints, the `BasePredictor` interface contract, JSON tool routing, Hindi/Hinglish query handling, WhatsApp message drafting, and the Temporal activities pipeline:
```bash
pytest -v
# or: make test
```
All 8 pytest tests run and pass out-of-the-box.

---

## 🧭 Repository Structure

```
dabbaeta/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application & API endpoints
│   ├── predictor.py     # TabPFN predictor interface & Sentry spans
│   ├── baselines.py     # XGBoost & LinearRegression predictors
│   ├── agent.py         # Gemma agent with JSON tool routing & retry parser
│   ├── tools.py         # Dispatch tools (predict_eta, list_risky, draft_message)
│   └── schemas.py       # Pydantic data schemas
├── workflows/
│   ├── __init__.py
│   ├── workflows.py     # Temporal DispatchWorkflow orchestration
│   ├── activities.py    # 5 Temporal activities (load, predict, flag, explain, draft)
│   └── worker.py        # Temporal worker runner with Sentry tracing
├── scripts/
│   ├── make_dataset.py  # Generates 5,000-row delivery dataset with is_late flag
│   └── benchmark.py     # Evaluates TabPFN vs XGBoost vs LinearRegression
├── tests/
│   └── test_dabbaeta.py # 8 comprehensive pytest tests
├── data/
│   ├── orders.csv       # Generated delivery dataset
│   └── dabbaeta.db      # Local SQLite database
├── docs/
│   ├── DEMO.md          # 2-minute live demo walkthrough script
│   ├── benchmark.json   # Full benchmark evaluation metrics
│   └── benchmark.png    # High-resolution benchmark comparison chart
├── frontend/
│   └── index.html       # Single-page glassmorphism dashboard
├── .env.example         # Sample environment variables
├── .env                 # Local environment configuration
├── Makefile             # Automation commands (dev, test, dataset, benchmark)
├── docker-compose.yml   # Multi-container local orchestration
├── Dockerfile           # App container definition
├── pytest.ini           # Pytest configuration
└── README.md            # Project documentation
```

---

## 📝 Assumptions

1. **TabPFN API Key & Local Fallback**: When `TABPFN_API_KEY` is provided and valid, `tabpfn-client` is queried. If the API key is absent or cloud service is unreachable, DabbaETA seamlessly swaps in a calibrated Prior-Fitted Network local surrogate without failing or blocking API callers.
2. **Promised Delivery Time**: If `promised_time_min` is absent in uploaded CSVs, it defaults to 40.0 minutes (the standard service-level agreement for city tiffin deliveries).
3. **Temporal In-Process Fallback**: If a developer launches DabbaETA without running `temporal server start-dev`, the backend seamlessly runs an in-process workflow runner executing identical activity chains and schemas so that UI interactions and test suites never break.
4. **Ollama Offline Fallback**: If Ollama is not running locally on port 11434, the agent catches the connection refusal, logs a Sentry fallback span, and routes queries like *"Aaj kaunsi orders late hongi?"* via a semantic dispatcher.
5. **Language Support**: The delay drafting engine supports Hinglish (standard bilingual conversational Hindi/English), formal Hindi, and English.

---

## 📜 License Notice

- **TabPFN License**: The free TabPFN API key provided by Prior Labs is strictly for non-commercial research and evaluation. Commercial deployment requires a commercial license from [Prior Labs](https://priorlabs.ai).
- **DabbaETA Codebase**: Open-source under the MIT License for the DEV Hackathon Challenge.
