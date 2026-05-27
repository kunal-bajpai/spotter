# Spotter — Project Instructions

Project-specific rules for Claude when working in this directory. Inherits global rules from `../../CLAUDE.md`.

## What this project is

AI Squat Coach — analyzes squat videos via a multi-agent pipeline (vision → kinematic diagnostics → coach feedback → corrected-form synthesis). Streamlit app; uses MediaPipe for pose estimation and Gemini for coaching feedback.

Architectural detail lives in `agents.md` (multi-agent design doc).

## Tech stack

- Language: Python (3.12 — see `.python-version`)
- Dependency manager: `uv` (lockfile: `uv.lock`)
- UI: Streamlit (`app.py`)
- Pose estimation: MediaPipe (`models/pose_landmarker_full.task`)
- LLM: Gemini

## How to run it locally

```
uv sync                                       # one-time per machine, recreates .venv
cp .env.example .env                          # one-time, then edit .env and paste your Gemini key
uv run streamlit run app.py                   # start the app
```

The Gemini API key is read from `.env` at startup (loaded via `python-dotenv` in `app.py`). The `.env` file is git-ignored — never commit it.

## Git workflow

- Branch: `master` (this is an older repo; the branch name predates GitHub's switch to `main`).
- **Push directly to master.** No feature branches, no pull requests for this project.
- Remote: `https://github.com/kunal-bajpai/spotter.git`

## Conventions specific to this project

- `.venv/`, `cache/`, `logs/`, `outputs/`, `models/`, `data/`, and `*.mp4` / `*.task` files are git-ignored — they're regenerable runtime artifacts.
- Runtime layout: inputs in `data/`, ML weights in `models/`, temp files in `cache/`, generated videos in `outputs/`, logs in `logs/`.
- The pose model `models/pose_landmarker_full.task` is git-ignored and auto-downloaded by `VisionAgent` on first run. To re-fetch manually, delete it; it'll be re-downloaded from the MediaPipe model zoo.

## Known gotchas

- `.venv/` won't work if the project folder is moved between paths — recreate it with `uv sync`.
- Streamlit holds file locks on `.venv/Scripts/*.pyd` while running. Stop the Streamlit server (Ctrl+C) before moving the folder or modifying the venv, or `mv` will fail with "Device or resource busy".
