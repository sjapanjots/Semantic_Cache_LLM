# Contributing

Thanks for helping out!

```bash
git clone https://github.com/sjapanjots/Semantic_Cache_LLM
cd Semantic_Cache_LLM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Before opening a pull request:

```bash
ruff check . && ruff format --check .
pytest -q
```

Guidelines:

- Keep the core dependency-free apart from `numpy`. Provider SDKs and model libraries
  belong behind optional extras and lazy imports.
- Add a test for every behaviour change.
- Note user-visible changes in `CHANGELOG.md`.

To add a new storage back-end, subclass `semcache.BaseStore`. To add an embedder,
implement `name` and `embed(texts) -> np.ndarray` returning L2-normalised rows.
