# EvidenceRL

A minimal prototype that demonstrates how to combine retrieval-augmented
generation (RAG) with a reward signal that measures how closely the generated
answer aligns with the retrieved evidence. The demo now focuses on cardiac
disease diagnosis using the MIMIC-IV-Ext cardiac extracts described below. The
pipeline performs the following steps:

1. **Retrieve evidence** using dense embeddings from a Hugging Face encoder and
   a concept-constrained similarity rule. Documents are chunked, embedded, and
   filtered to only those whose tagged medical concepts overlap with the query,
   mirroring the expertise-aware retrieval described in recent RAG literature.
2. **Generate an answer** with a Hugging Face language model that is prompted
   with the retrieved evidence.
3. **Align evidence post-generation** by querying the entire knowledge base with
   the generated answer to identify which passages it most closely mirrors.
4. **Compute a reward** by combining overlap in document identifiers with
   embedding similarity between the pre- and post-generation evidence sets.
   Scores are scaled into the [-1, 1] range where -1 indicates conflicting
   evidence and +1 denotes perfect agreement.

When ground-truth answers are available—such as in the cardiac QA cases—the
alignment score is multiplied by an **LLM-as-a-judge** correctness flag. A
compact Hugging Face model evaluates the model output against the gold answer
and returns `correct` or `incorrect`. The reward is preserved only for correct
answers, while incorrect ones are reduced to zero to reflect uncertainty.

The code lives in `src/evidence_rl` and is designed to be easy to extend with
more sophisticated models or reward functions.

## Quickstart

Create a virtual environment (optional), install the dependencies, and run the
demo script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers sentence-transformers pytest matplotlib
python main.py --plot-dir plots --json-output runs/
```

The first run downloads the default `sshleifer/tiny-gpt2` checkpoint used for
answer generation, the lightweight judge model, and the
`sentence-transformers/all-MiniLM-L6-v2` embedder used for retrieval. The demo
iterates over cardiac cases derived from the MIMIC-IV-Ext cardiac extracts—e.g.,
ischemic heart disease, heart failure, arrhythmias, and post-MI complications—
printing the ground truth, retrieved and inferred evidence, alignment score,
LLM-judged correctness, and reward. If `--plot-dir` is
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

### Using the real MIMIC-IV-Ext cardiac data

If you have access to the MIMIC-IV-Ext cardiac extracts, point the demo at the
dataset directory and it will build one document per ICD prefix (I20–I25,
I30–I50, 410–414, and 420–428) using the clinical long titles from
`heart_diagnoses_all_true.csv` (or the truncated `heart_diagnoses_all.csv`):

```bash
python main.py --data-path /path/to/mimic-cardiac --embedding-model-name sentence-transformers/all-MiniLM-L6-v2
```

Programmatic users can call `load_cardiac_icd_documents(data_path)` to obtain
the corpus and feed it directly into `RagRlPipeline`. The loader preserves ICD
codes and representative titles in document metadata so the concept-aware
retriever remains effective against the real knowledge base.

### RAG baseline with clinical guideline PDFs

If you maintain a folder of guideline PDFs (or `.txt` exports) under
`Medical_Knowledge/`, you can use it as the RAG knowledge base instead of the
toy ICD corpus. The ingestion helpers read each file, detect section/subsection
headings, chunk the content, and optionally persist the chunks as JSONL for
later reuse:

```bash
python main.py \
  --model-name /path/to/model_name \
  --judge-model-name /path/to/judge_model_name \
  --patient-data-path /path/to/patient_data \
  --patient-pipeline rag \
  --knowledge-path /path/to/Medical_Knowledge \
  --knowledge-jsonl /tmp/knowledge.jsonl \
  --chunk-size 400 --chunk-overlap 80 \
  --max-patients 10 \
  --json-output /path/to/json_output
```

If `--knowledge-jsonl` already exists, the chunks are loaded directly; if not,
the processed chunks are exported to that path for future runs. You can also use
the building blocks programmatically:

```python
from evidence_rl import load_pdf_knowledge_documents, export_documents_jsonl
docs = load_pdf_knowledge_documents("/path/to/Medical_Knowledge", chunk_size=400, overlap=80)
export_documents_jsonl(docs, "knowledge.jsonl")
```

The resulting documents can be fed into `DocumentStore` for embedding and
retrieval or supplied directly to `RagRlPipeline`.

### Prompt-only baseline on patient notes

For a lightweight baseline that avoids retrieval entirely, you can point the
demo at the full MIMIC-IV-Ext cardiac directory using `--patient-data-path`.
The tool will textualise each patient note from `heart_diagnoses.csv`, prompt an
LLM for the top 5 diagnoses and procedures, and compute precision/recall at
cut-offs 1–5 against the labels in `heart_diagnoses_all.csv` and
`heart_procedures.csv`. Predictions are adjudicated by an LLM-as-a-judge so
metrics account for paraphrased or equivalent terminology rather than strict
string matches.

```bash
python main.py --patient-data-path /path/to/mimic-cardiac --json-output baseline.json
```

Use `--max-patients` to subsample cases when experimenting locally. To maximise
GPU utilisation, the baseline batches prompts through the generator using a
dataset-style call; tune `--batch-size` to suit your hardware. The JSON payload
captures the exact prompt, generated text, per-case metrics, the ground-truth
diagnoses/procedures used for judging, and an aggregate precision/recall
summary.

If you want to keep the patient-note prompts but add retrieval from your
clinical guideline corpus, switch to the RAG pipeline with `--patient-pipeline
rag` and point the CLI at your `Medical_Knowledge` folder (or a prebuilt
JSONL) using `--knowledge-path`/`--knowledge-jsonl`:

```bash
python main.py --patient-data-path /path/to/mimic-cardiac \
  --patient-pipeline rag --knowledge-path /path/to/Medical_Knowledge \
  --json-output rag_results.json
```

This uses the same patient textualisation and diagnosis/procedure prompt as the
baseline while injecting retrieved guideline chunks into the generator prompt
before computing alignment-aware rewards.

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
