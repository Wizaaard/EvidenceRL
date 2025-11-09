# EvidenceRL

A minimal prototype that demonstrates how to combine retrieval-augmented
generation (RAG) with a reward signal that measures how closely the generated
answer aligns with the retrieved evidence.  The pipeline performs the following
steps:

1. **Retrieve evidence** using a lightweight TF-IDF retriever built over a small
   document collection.
2. **Generate an answer** by composing snippets from the retrieved evidence.
3. **Align evidence post-generation** by comparing the generated answer with the
   retrieved passages to identify which ones were actually used.
4. **Compute a reward** by combining overlap in document identifiers with
   embedding similarity between the pre- and post-generation evidence sets.

The code lives in `src/evidence_rl` and is designed to be easy to extend with
more sophisticated models or reward functions.

## Quickstart

Create a virtual environment (optional) and run the demo script:

```bash
python -m venv .venv
source .venv/bin/activate
python main.py
```

Running `main.py` prints the query, the evidence retrieved before generation, a
simple generated answer, the evidence inferred from the answer, and the final
reward value.

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest
pytest
```
