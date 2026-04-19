---
description: "Workspace instructions for the stock-model repository. Use when editing Python, backend, optimize, simulation, risk, or test code in this project."
---

# stock-model workspace instructions

## What this repository is
- A Python-based quantitative trading platform with modules for data loading, signal prediction, strategy decision-making, backtesting, risk management, optimization, and simulation.
- A separate `backend/` service built with async FastAPI, Alembic migrations, and Docker Compose for database and service orchestration.
- Primary entrypoints are Python scripts and modules rather than a packaged library.

## Key areas
- `backend/` — async HTTP API, database models, migrations, and backend service tooling.
- `optimize/` — hyperparameter search, optimization orchestration, and scoring logic.
- `simulator/`, `strategy/`, `backtest/` — historical simulation and strategy evaluation.
- `data/` — market data ingestion, live/simulated data adapters, and ticker registry.
- `predict/`, `equity/`, `risk/` — prediction, position sizing, equity/risk engines.
- `tests/` — pytest-based tests.

## Recommended conventions
- Use `python -m` commands and `pytest` for execution/testing rather than relying on hidden packaging metadata.
- Backend development should follow existing async patterns in `backend/`, including `uvicorn main:app --reload` and Alembic migrations under `backend/migrations/`.
- Preserve repository structure and avoid moving large domain modules unless there is a strong reason.
- Keep file and symbol names consistent with existing snake_case and CamelCase conventions.
- Avoid introducing new dependencies unless necessary; existing docs suggest a Conda environment (`chronos`) and backend requirements in `backend/requirements.txt`.

## When editing
- For backend changes, reference `backend/readme.md` for Docker Compose and Alembic workflows.
- For optimization tasks, use `python -m optimize.auto_tune` as the canonical entrypoint.
- For tests, use `pytest` and keep new tests inside `tests/`.
- Do not modify local runtime/state artifacts in `sqllite/`, `.pytest_cache/`, `logs/`, `outputs/`, or `state/` unless explicitly requested.

## Best practices for AI assistance
- Link solutions to existing patterns in `Readme.md` and `backend/readme.md` instead of inventing new project structure.
- Prefer small, incremental changes over broad refactors unless the code path is clearly wrong or inconsistent.
- Keep domain terms like `regime`, `backtest`, `predict`, `alpha`, `KELLY`, and `ATR_STOP` intact.
- When adding new features, keep backend service and data access logic separate from strategy/math logic.
