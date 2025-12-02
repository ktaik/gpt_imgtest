# Repository Guidelines

## Project Structure & Module Organization
- `main.py` trims images, sends frames to GPT-4o, applies the online HMM filter, and writes reports/plots. Add helpers inside this file unless a clear split is needed.
- `images/` holds raw Robovie frame sequences; `images_trimmed/` is created on first run with cropped and scaled variants (`original/`, `quarter/`). Avoid mutating source images—add new folders if you bring data.
- `outputs/` is regenerated per run with JSON summaries (`results_*.json`), plots, and `summary.txt`. Do not rely on its contents persisting.

## Setup, Build, Test, and Development Commands
- Activate a virtualenv: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -U matplotlib numpy pillow python-dotenv openai`.
- Run the pipeline: `python main.py --pause 1.0` (writes to `outputs/` and `outputs/plots/`).
- Faster feedback: `python main.py --limit 10 --pause 0.5` (first 10 frames per scale) or reuse existing trims: `python main.py --trimmed-root images_trimmed`.

## Coding Style & Naming Conventions
- Python 3.11+ with type hints throughout; prefer small, pure functions. Match existing 4-space indentation and snake_case names for variables, functions, and paths.
- Keep constants uppercase (`OUTPUT_DIR`, `CROP_BOX`, `SCALE_CONFIGS`); place configuration near the top of `main.py`.
- No formatter is enforced—run `black` or similar before opening a PR; keep imports grouped (stdlib, third-party, local).
- Print output is part of the workflow; keep logging concise and flush when awaiting async tasks to preserve order.

## Testing Guidelines
- There is no formal test suite; validate changes by running `python main.py --limit …` and inspect `outputs/summary.txt` plus plots.
- When altering parsing, naming, or metrics, inspect `results_*.json` for correct frame indices, labels, and reasons. Spot-check latency plots for obvious spikes.

## Commit & Pull Request Guidelines
- Follow conventional, action-oriented commit messages (imperative mood). Keep commits focused: data preparation, model calls, and plotting changes should be split when possible.
- PRs should include: a short description, commands run, and before/after notes for outputs (mention key files in `outputs/`). Link related issues. Screenshots of plots help when visuals change.

## Security & Configuration Tips
- Provide `OPENAI_API_KEY` via `.env` (loaded by `python-dotenv`); never commit secrets or raw datasets. If adding new config, prefer environment variables with sensible fallbacks and document them in this file.
- Keep generated artifacts out of version control; if new output folders are needed, add them under `outputs/` to avoid touching source data.
