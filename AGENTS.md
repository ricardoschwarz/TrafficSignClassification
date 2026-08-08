# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Dependencies are pinned in `requirements.txt` (tensorflow 2.16.1, validated against Python 3.12). TensorFlow does not yet publish wheels for very new CPython releases (e.g. 3.14 at time of writing) — if the system `python3` is too new for the pin, use a tool like `uv python install 3.12` to get a compatible interpreter for a venv rather than assuming `pip install -r requirements.txt` will resolve on any Python.
- Run order: `image_preprocessing.py` once (resizes `data/train/**/*.ppm` in place to 50x50, overwriting originals), then `training.py`. See README Setup section for dataset download/layout.
- `data/` and `models/` are gitignored; `training.py` writes the trained complex model to `models/complex_model`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
