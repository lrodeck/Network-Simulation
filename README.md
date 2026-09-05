# Discourse Lab

Vectorized, non-agentic simulation toolbox for theories of large-scale online
discourse. Dynamics are numeric; language is an offline rendering pass that
never runs inside the tick loop (see `discourse-lab-spec.md` for the model,
`discourse-lab-dev.md` for design decisions, `TODO.txt` for build status).

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from discourse_lab import Config, cached_run, load_run

cfg = Config()
run_dir = cached_run(cfg, seed=0)
handle = load_run(cfg, seed=0)
print(handle.metrics())
```

A run is keyed by its config's structural hash and seed
(`dlab/runs/{cfg_hash}/{seed}/`), so `cached_run` reuses a completed run
instead of recomputing it. The workspace root defaults to `./dlab`, `DLAB_HOME`
if set, or `/content/dlab` under Colab.

## LLM realization (optional)

Text generation is an offline pass over a completed run, never inside the
tick (`discourse_lab.llm`). It talks to [Ollama Cloud](https://ollama.com) by
default:

```bash
export OLLAMA_API_KEY=...   # https://ollama.com/settings/keys
```

```python
from discourse_lab.llm import OllamaCloudClient, realize

client = OllamaCloudClient(model="gpt-oss:120b-cloud")
texts = realize(client, cfg, posts, pop, post_ids=[...])  # lazy: only these posts
```

Point `OllamaCloudClient(base_url=...)` at a local `ollama serve` instead if
you'd rather not use the hosted service.

## Tests

```bash
pytest
```
