# Spotter

AI squat coach. Upload a video of a back squat and get pose-estimated kinematic feedback plus a corrected-form video.

The pipeline runs as four agents:

1. **Vision** — MediaPipe extracts pose landmarks frame-by-frame.
2. **Diagnostics** — kinematic checks (depth, knee tracking, torso angle, tempo).
3. **Coach** — Gemini turns the diagnostics into plain-English feedback.
4. **Synthesis** — generates a corrected-form demo video.

Architectural detail lives in [AGENTS.md](agents.md).

## Requirements

- Python 3.12 (managed via [`uv`](https://docs.astral.sh/uv/))
- A Gemini API key — get one at https://aistudio.google.com/apikey

## Setup

```bash
uv sync                 # creates .venv, installs deps
cp .env.example .env    # then edit .env and paste your Gemini key
```

## Run

```bash
uv run streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Layout

```
app.py            Streamlit UI + orchestration entrypoint
src/              Agent implementations
  vision_agent.py
  diagnostic_agent.py
  coach_agent.py
  synthesis_agent.py
  orchestrator.py
  utils/logger.py
tests/            pytest suite
data/             input videos (git-ignored)
models/           MediaPipe weights (git-ignored)
cache/            temp uploads and intermediate artifacts (git-ignored)
outputs/          generated corrected-form videos (git-ignored)
logs/             app.log (git-ignored)
```

## Tests

```bash
uv run pytest
```
