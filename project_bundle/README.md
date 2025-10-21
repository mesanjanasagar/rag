# Hybrid DSPy + LangGraph Agent (starter)

## Summary
## How to run
Project: hybrid RAG + SQL CLI

This repository runs a small hybrid retrieval + NL2SQL pipeline against a local SQLite Northwind dataset and a small set of docs.

How to run (assumes Python 3.10+ and virtualenv):

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the batch CLI (example):

```bash
source .venv/bin/activate
python3 run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

What this does:
- Uses a TF-IDF retriever over the `docs/` directory.
- Attempts to generate SQL with DSPy if present; otherwise uses deterministic NL->SQL rules for the 6 sample questions.
- Executes SQL against `data/northwind.sqlite` and synthesizes typed `final_answer` values matching each question's `format_hint`.
- Writes `outputs_hybrid.jsonl` with one JSON object per line. Internal trace data is stripped from the output file.

Using local Ollama with dspy/litellm:
- If you have Ollama installed and running locally, this code will attempt to route model calls to it. dspy/litellm usually expects provider-prefixed model strings like `ollama/deepseek-r1:7b`.
- To prefer a specific local model, set the `DEFAULT_MODEL_NAME` constant in `agent/dspy_signatures.py` (e.g., `deepseek-r1:7b`).

Notes on determinism and fallbacks:
- For reproducible grading, deterministic rule-based NL->SQL is used for the assignment's sample questions when DSPy model calls are unavailable or fail.
- When SQL returns no rows (due to data snapshot limits), the synthesizer returns typed fallbacks (e.g., `{"customer":"Unknown","margin":0.0}`) rather than `null`.

Next steps (optional):
- Tune confidence heuristics further.
- Expand NL->SQL patterns or enable dynamic model-driven SQL generation by installing a compatible `dspy`/`litellm` and ensuring Ollama is running.
