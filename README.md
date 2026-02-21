# Stuff-made-easy

**Natural language → manufacturable PCB.** Describe a circuit in plain English; get schematics, netlists, PCB layout, Gerber files, BOM, and simulation.

---

## What it does

- **Natural language input** — Describe your circuit; the pipeline parses and structures it.
- **AI code generation** — LLM produces executable SKiDL from your description.
- **Full pipeline** — SKiDL → netlist → KiCad PCB → Gerber / drill / STEP, plus BOM and simulation.
- **Verification** — ERC, DRC, and DFM checks with clear reporting.
- **Web UI** — React app: prompt input, live status, preview, and file download.

## Quick start (one command)

From the repo root (Windows PowerShell):

```powershell
.\run.ps1
```

This script will: create `.env` from `.env.example` if missing, install Python dependencies, run auth and NLP tests, then start the API at http://localhost:8000. Press Ctrl+C to stop.

**Manual option:**

```bash
# Backend (from repo root)
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then set OPENAI_API_KEY, DATABASE_URL, REDIS_URL if needed
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm start
```

**Optional:** Start Postgres and Redis for full features:

```bash
docker-compose up -d postgres redis
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | API root |
| http://localhost:8000/docs | API docs (Swagger) |
| http://localhost:3000 | Web UI |

## Project layout

```
├── src/                 # Backend
│   ├── api/             # FastAPI routes, schemas
│   ├── models/          # SQLAlchemy models
│   └── services/        # NLP, LLM, SKiDL, KiCad, verification, BOM, simulation
├── tests/               # Unit + property-based tests
├── frontend/            # React + TypeScript UI
├── docker/              # Postgres init, etc.
├── .kiro/               # Specs and steering (requirements, design, tasks)
├── DEVELOPMENT_STATUS.md   # Status, roadmap, and plan for later
├── CONTRIBUTING.md      # How to contribute
└── docker-compose.yml   # Postgres, Redis, optional API/frontend
```

## Status and roadmap

- **Current status:** MVP pipeline is implemented (NLP → LLM → SKiDL → KiCad → Gerber, BOM, simulation, UI). See **[DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)** for a full task breakdown.
- **Next:** Security (JWT auth, encryption, audit), integration tests, then deployment.

## Testing

```bash
pytest tests/unit/ -v
pytest tests/property/ -v --hypothesis-show-statistics
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT.
