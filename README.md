# EvidenceRL

A minimal prototype that demonstrates how to combine retrieval-augmented
generation (RAG) with a reward signal that measures how closely the generated
answer aligns with the retrieved evidence.  The pipeline performs the following
steps:

1. **Retrieve evidence** using a lightweight TF-IDF retriever built over a small
   document collection.
2. **Generate an answer** with a Hugging Face language model that is prompted
   with the retrieved evidence.
3. **Align evidence post-generation** by comparing the generated answer with the
   retrieved passages to identify which ones were actually used.
4. **Compute a reward** by combining overlap in document identifiers with
   embedding similarity between the pre- and post-generation evidence sets.

The code lives in `src/evidence_rl` and is designed to be easy to extend with
more sophisticated models or reward functions.

## Quickstart

Create a virtual environment (optional), install the dependencies, and run the
demo script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers pytest
python main.py
```

The first run downloads the default `sshleifer/tiny-gpt2` checkpoint used for
answer generation.  Subsequent runs print the query, the evidence retrieved
before generation, the model's answer, the evidence inferred from the answer,
and the final reward value.

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest transformers
pytest
```

To experiment with a different generator, pass the `model_name` parameter when
constructing `RagRlPipeline`, or create your own generator implementation that
follows the `EvidenceGenerator` protocol and supply it via the `generator`
argument.
