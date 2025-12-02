# Repository Guidelines

## Project Structure & Data Layout
- Core logic lives in `main.py`; there is no package scaffolding.
- Raw Robovie frame dumps are expected under `images/participant/xxx.png`; cropped versions are written to `images_trimmed/original/…` on first run, with scaled variants in `images_trimmed/{quarter,…}`.
- All run artifacts land in `outputs/` (JSON metrics, plots, heatmaps, summary.txt); the directory is cleared at the start of each run.
- Secrets stay in `.env` (not committed); runtime virtualenvs belong in `.venv/`.

## Setup & Dependencies
- Use Python 3.10+ in a venv: `python -m venv .venv && source .venv/bin/activate`.
- Install runtime deps explicitly: `pip install openai python-dotenv pillow matplotlib numpy`.
- Ensure `.env` contains `OPENAI_API_KEY=<token>` before invoking the script.

## Build, Test, and Development Commands
- Quick eval on a subset to save tokens: `python main.py --limit 10 --pause 1.0`.
- Full run with defaults: `python main.py --image-root images --trimmed-root images_trimmed`.
- Custom plots/outputs: pass `--frame-plot myplot.png --latency-plot mylatency.png`; names are suffixed per scale.
- Each invocation deletes and recreates `outputs/`; copy artifacts elsewhere before rerunning.

## Coding Style & Naming Conventions
- Match current file style: tabs for indentation, snake_case functions, UPPER_SNAKE_CASE module constants, and type hints throughout.
- Keep helpers pure where possible; prefer small functions with explicit paths and return types.
- JSON writing uses UTF-8 with `ensure_ascii=False`; preserve this when extending serialization.

## Testing Guidelines
- No automated tests exist; rely on scripted runs. Start with `--limit` to validate parsing and plotting quickly.
- Verify outputs: `outputs/summary.txt` for metrics, `outputs/plots/` for heatmaps/latency charts, and the JSON results for label correctness.
- If adding logic around image layouts, add defensive checks (file existence, label parsing) to avoid silent skips.

## Commit & Pull Request Guidelines
- Commits in history are short and lower-case (e.g., “finish”); continue using concise, imperative summaries under ~72 chars.
- PRs should describe intent, include key CLI flags used, note dataset scope (e.g., limit or scale), and attach representative artifacts (plot paths, sample JSON excerpt).
- Link related issues or TODOs inline in the description; call out any API-costing steps so reviewers can reproduce selectively.

## Security & Configuration Tips
- Never commit `.env` or API keys; prefer environment variables for CI/local overrides.
- Large image sets should remain local unless explicitly needed; avoid pushing generated plots in `outputs/` unless they are required evidence.
