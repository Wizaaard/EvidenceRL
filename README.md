# EvidenceRL

A minimal prototype that demonstrates how to combine retrieval-augmented
generation (RAG) with a reward signal that measures how closely the generated
answer aligns with the retrieved evidence.  The pipeline performs the following
steps:

1. **Retrieve evidence** using a lightweight TF-IDF retriever built over a
   curated medical knowledge base.
2. **Generate an answer** with a Hugging Face language model that is prompted
   with the retrieved evidence.
3. **Align evidence post-generation** by comparing the generated answer with the
   retrieved passages to identify which ones were actually used.
4. **Compute a reward** by combining overlap in document identifiers with
   embedding similarity between the pre- and post-generation evidence sets.
   Scores are scaled into the [-1, 1] range where -1 indicates conflicting
   evidence and +1 denotes perfect agreement.

When ground-truth answers are available—such as in our simulated clinical
question answering dataset—the alignment score is multiplied by an
**LLM-as-a-judge** correctness flag.  A compact Hugging Face model evaluates the
model output against the gold answer and returns `correct` or `incorrect`.  The
reward is preserved only for correct answers, while incorrect ones are reduced
to zero to reflect uncertainty.

The code lives in `src/evidence_rl` and is designed to be easy to extend with
more sophisticated models or reward functions.

## Quickstart

Create a virtual environment (optional), install the dependencies, and run the
demo script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers pytest matplotlib
python main.py --plot-dir plots
```

The first run downloads the default `sshleifer/tiny-gpt2` checkpoint used for
answer generation and the lightweight judge model.  The demo iterates over a
handful of synthetic obstetrics questions, printing the ground truth, retrieved
and inferred evidence, alignment score, LLM-judged correctness, and reward.  If
`--plot-dir` is supplied, two scatter plots are written to disk showing reward
vs. correctness and alignment vs. correctness across the batch.

If CUDA devices are available, both the generator and the judge automatically
run on GPU.  A single GPU loads the model onto `cuda:0`, while multi-GPU setups
leverage `device_map="auto"` so the weights are distributed across all visible
GPUs without extra configuration.

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest transformers matplotlib
pytest
```

To experiment with a different generator or judge, pass the `model_name` (for
the generator) or `judge_model_name` (for the LLM answer judge) parameters when
constructing `RagRlPipeline`.  You can also implement custom components that
follow the exposed `EvidenceGenerator` and `AnswerJudge` protocols and supply
them via the `generator` and `answer_judge` arguments.
