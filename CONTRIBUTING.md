# Contributing

Thanks for your interest in improving this industrial scheduling RL codebase.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pillow
```

## Code quality

```bash
black .
ruff .
python -m pytest -q
```

Pull requests should stay green on the GitHub Actions workflows:

- environment validity check (`check_env.yml`)
- formatting / lint (`code_convention.yml`)

## Scope guidelines

- Do **not** silently change reward weights, observation semantics, or trained-result pickles used for published benchmarks.
- New algorithms or ablations should land as additive scripts/configs with clearly named output artifacts.
- Prefer regenerating documentation assets through `scripts/generate_docs_assets.py`.

## Suggested contribution areas

1. Action masking for illegal job–machine pairs
2. Deterministic benchmark harness with seed matrix
3. W&B / MLflow logging adapters
4. Interactive episode viewer
5. Packaging a stable `FactoryEnv` API under `src/`
