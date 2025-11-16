# EvidenceRL

A minimal prototype that demonstrates how to combine retrieval-augmented
generation (RAG) with a reward signal that measures how closely the generated
answer aligns with the retrieved evidence.  The pipeline performs the following
steps:

1. **Retrieve evidence** using a lightweight TF-IDF retriever that applies a
   concept-constrained similarity rule over a curated medical knowledge base.
   A document is only eligible when its tagged medical concepts overlap with
   the query, mirroring the expertise-aware retrieval described in recent RAG
   literature.
2. **Generate an answer** with a Hugging Face language model that is prompted
   with the retrieved evidence.
3. **Align evidence post-generation** by querying the entire knowledge base with
   the generated answer to identify which passages it most closely mirrors.
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
python main.py --plot-dir plots --json-output runs/
```

The first run downloads the default `sshleifer/tiny-gpt2` checkpoint used for
answer generation and the lightweight judge model.  The demo iterates over an
expanded set of synthetic obstetrics questions—including decoy passages meant to
confuse retrieval—printing the ground truth, retrieved and inferred evidence,
alignment score, LLM-judged correctness, and reward.  If `--plot-dir` is
supplied, two scatter plots are written to disk showing reward vs. correctness
and alignment vs. correctness across the batch.

Because the generated answer is re-retrieved against the full corpus, the sample
run demonstrates all reward regimes:

* **Positive rewards** when correct answers stick to the initial evidence.
* **Zero rewards** when the judge flags an incorrect answer and the alignment
  score is suppressed.
* **Negative rewards** when a correct answer leans on evidence that was not part
  of the original retrieval set, indicating a mismatch between retrieval and
  usage.

If CUDA devices are available, both the generator and the judge automatically
run on GPU.  A single GPU loads the model onto `cuda:0`, while multi-GPU setups
leverage `device_map="auto"` so the weights are distributed across all visible
GPUs without extra configuration.

Supplying `--json-output` writes detailed intermediate artifacts for each case
into the provided directory (for example `runs/case_01.json`) and produces an
aggregated `results.json` summary that includes prompts, retrieved evidence, the
generated answer, judge verdict (`"correct"`/`"incorrect"`), correctness flags,
and reward values.  Passing a filename such as `--json-output results.json`
skips the per-case files and only writes the summary payload.

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest transformers matplotlib
pytest
```

To experiment with a different generator or judge, pass the `model_name` (for
the generator) or `judge_model_name` (for the LLM answer judge) parameters when
constructing `RagRlPipeline` or use the corresponding command-line arguments in
`main.py`.  Each `RagRlResult` instance also exposes `to_dict()` and
`save_json()` helpers so programmatic callers can persist intermediate artifacts
with custom naming schemes.  You can implement custom components that follow the
exposed `EvidenceGenerator` and `AnswerJudge` protocols and supply them via the
`generator` and `answer_judge` arguments.
