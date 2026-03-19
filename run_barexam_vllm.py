#!/usr/bin/env python3
"""Entry point for running the BarExam QA benchmark with vLLM.

Generates answers (with reasoning) for MBE multiple-choice questions.
For RAG mode, retrieves legal passages to provide as evidence context.
For no-RAG mode, the model answers from parametric knowledge only.

Usage:
    python run_barexam_vllm.py \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --barexam-data-path data/barexam/ \
        --split test \
        --json-output barexam_output/llama8b_rag.json

    # Closed-book baseline:
    python run_barexam_vllm.py \
        --model-name ... --barexam-data-path data/barexam/ \
        --no-rag --json-output barexam_output/llama8b_norag.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl.barexam_data import BarExamCase, load_barexam_cases, download_barexam
from evidence_rl.domains import get_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BarExam QA benchmark with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--model-name", required=True, help="HuggingFace model path/name")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # LoRA
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--max-lora-rank", type=int, default=64)

    # Data
    parser.add_argument("--barexam-data-path", required=True,
                        help="Path to BarExam data directory")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-rag", action="store_true",
                        help="Run without passages (closed-book baseline)")
    parser.add_argument("--max-passages", type=int, default=5,
                        help="Number of retrieved passages per question (default: 5)")
    parser.add_argument("--faiss-index-dir", default=None,
                        help="Path to pre-built FAISS index for passage retrieval. "
                             "When set, retrieves --max-passages from the index instead of using gold.")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model for FAISS retrieval (default: all-MiniLM-L6-v2)")
    parser.add_argument("--include-gold", action="store_true",
                        help="When using FAISS retrieval, always include gold passage in evidence")
    parser.add_argument("--download", action="store_true",
                        help="Download BarExam data before running")

    # Extractor
    parser.add_argument("--extractor-model", default=None,
                        help="Path to extractor model for fallback extraction when JSON "
                             "parsing fails. If not set, reuses the generator model.")
    parser.add_argument("--no-llm-extraction", action="store_true",
                        help="Disable LLM extraction fallback (use only regex)")

    # Output
    parser.add_argument("--json-output", required=True, help="Output JSON path")

    return parser


def _format_question_with_choices(case: BarExamCase) -> str:
    """Format the full question text including preamble, question, and choices."""
    parts = []

    # Preamble (fact pattern)
    if case.prompt and str(case.prompt).strip() and str(case.prompt) != "nan":
        parts.append(str(case.prompt).strip())

    # Question
    parts.append(case.question.strip())

    # Choices
    parts.append("")
    for letter in ("A", "B", "C", "D"):
        parts.append(f"({letter}) {case.choices[letter]}")

    return "\n".join(parts)


def generate_batch(
    llm,
    sampling_params,
    cases: list[BarExamCase],
    domain,
    no_rag: bool = False,
    gold_as_evidence: bool = True,
    retriever=None,
    max_passages: int = 3,
    include_gold: bool = False,
    lora_request=None,
) -> list[dict]:
    """Generate answers for a batch of BarExam cases.

    Args:
        gold_as_evidence: If True and retriever is None, use gold passage as evidence.
        retriever: FAISSRetriever instance for passage retrieval. When set,
            retrieves max_passages from the index per question.
        max_passages: Number of passages to retrieve (default: 3).
        include_gold: If True and retriever is set, always include gold passage.
    """
    prompts = []
    for case in cases:
        question_text = _format_question_with_choices(case)
        if no_rag:
            prompt = domain.build_norag_prompt(question_text)
        else:
            if retriever is not None:
                # FAISS retrieval
                retrieved_docs = retriever.retrieve(case.full_question, top_k=max_passages)
                evidence = [doc.document.text.strip() for doc in retrieved_docs
                            if doc.document.text.strip()]
                # Optionally include gold if not already retrieved
                if include_gold and case.gold_passage:
                    gold_text = case.gold_passage.strip()
                    if gold_text not in evidence:
                        evidence.insert(0, gold_text)
            elif gold_as_evidence and case.gold_passage:
                evidence = [case.gold_passage]
            else:
                evidence = []
            prompt = domain.build_rag_prompt(question_text, evidence)
        prompts.append(prompt)

    # Build chat messages
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    chat_kwargs = {"sampling_params": sampling_params}
    if lora_request:
        chat_kwargs["lora_request"] = lora_request

    outputs = llm.chat(conversations, **chat_kwargs)

    results = []
    for case, output, prompt in zip(cases, outputs, prompts):
        raw_output = output.outputs[0].text.strip() if output.outputs else ""
        raw_output = _strip_thinking(raw_output)

        parsed = domain.parse_output(raw_output)

        result = {
            "case_id": case.case_id,
            "question": case.full_question,
            "choices": case.choices,
            "gold_answer": case.answer,
            "gold_passage": case.gold_passage,
            "source": case.source,
            "subject": case.subject,
            "prompt": prompt,
            "raw_output": raw_output,
            "parse_success": parsed is not None,
            "structured_output": None,
        }

        if parsed:
            result["structured_output"] = {
                "answer": parsed[0].name,
                "reasoning": parsed[0].reasoning,
            }
            acc = domain.compute_accuracy(parsed, [case.answer])
            result["accuracy"] = acc
            result["correct"] = acc == 1.0

        results.append(result)

    return results


def _build_extraction_prompt(raw_output: str, choices: dict) -> str:
    """Build a prompt to extract answer letter from malformed output."""
    choice_text = "\n".join(f"({k}) {v}" for k, v in choices.items())
    return f'''The following is a response to a multiple-choice bar exam question. Extract the answer letter (A, B, C, or D) and the reasoning.

The choices were:
{choice_text}

Response to extract from:
{raw_output}

Return ONLY valid JSON in this format:
{{"answer": "X", "reasoning": "Brief summary of the reasoning..."}}

Begin your response with {{'''


def llm_extract_batch(
    llm,
    failed_results: list[dict],
    domain,
    extractor_llm=None,
) -> list[dict]:
    """Use LLM to extract answers from malformed outputs.

    Args:
        llm: The generator LLM instance (used if extractor_llm is None)
        failed_results: Results where parse_success is False
        domain: BarExamDomain instance
        extractor_llm: Optional separate LLM for extraction

    Returns:
        Updated results with extracted answers where possible.
    """
    from vllm import SamplingParams

    if not failed_results:
        return failed_results

    active_llm = extractor_llm if extractor_llm is not None else llm

    # Build extraction prompts
    prompts = []
    for r in failed_results:
        prompts.append(_build_extraction_prompt(r["raw_output"], r["choices"]))

    conversations = [[{"role": "user", "content": p}] for p in prompts]

    extraction_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
    )

    outputs = active_llm.chat(conversations, sampling_params=extraction_params)

    extracted_count = 0
    for r, output in zip(failed_results, outputs):
        extracted_text = output.outputs[0].text.strip() if output.outputs else ""
        extracted_text = _strip_thinking(extracted_text)

        parsed = domain.parse_output(extracted_text)
        if parsed:
            r["parse_success"] = True
            r["structured_output"] = {
                "answer": parsed[0].name,
                "reasoning": parsed[0].reasoning,
            }
            r["extraction_method"] = "llm"
            acc = domain.compute_accuracy(parsed, [r["gold_answer"]])
            r["accuracy"] = acc
            r["correct"] = acc == 1.0
            extracted_count += 1

    print(f"  LLM extraction: recovered {extracted_count}/{len(failed_results)} failed parses")
    return failed_results


def _strip_thinking(text: str) -> str:
    """Strip gpt-oss thinking/analysis channel markers."""
    if "<|channel|>final" in text:
        text = text[text.index("<|channel|>final") + len("<|channel|>final"):].strip()
        for marker in ("<|end|>", "<|return|>", "<|endoftext|>"):
            if marker in text:
                text = text[:text.index(marker)].strip()
        return text
    if text.startswith("analysis"):
        boundary = "assistantfinal"
        idx = text.find(boundary)
        if idx >= 0:
            return text[idx + len(boundary):].strip()
    return text


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 60)
    print("BarExam QA Pipeline (vLLM)")
    print(f"  Split: {args.split}")
    if args.no_rag:
        mode_str = "no-RAG (closed-book)"
    elif args.faiss_index_dir:
        mode_str = f"RAG (FAISS retrieval, k={args.max_passages})"
    else:
        mode_str = "RAG (with gold passage)"
    print(f"  Mode: {mode_str}")
    print(f"  Model: {args.model_name}")
    print("=" * 60)

    if args.download:
        download_barexam(args.barexam_data_path)

    # Load data
    cases = load_barexam_cases(
        data_path=args.barexam_data_path,
        split=args.split,
        max_cases=args.max_cases,
    )
    print(f"Loaded {len(cases)} BarExam cases ({args.split} split)")

    # Initialize domain
    domain = get_domain("barexam")

    # Initialize FAISS retriever (if requested) — before vLLM to use GPU for embedding
    retriever = None
    if args.faiss_index_dir and not args.no_rag:
        from evidence_rl.faiss_retrieval import FAISSDocumentStore, FAISSRetriever
        from evidence_rl.retrieval import HuggingFaceEmbedder

        print(f"\nLoading FAISS index from: {args.faiss_index_dir}")
        embedder = HuggingFaceEmbedder(
            model_name=args.embedding_model_name,
            force_cpu=True,  # GPU will be used by vLLM
        )
        store = FAISSDocumentStore.load(args.faiss_index_dir, embedder=embedder, use_gpu=False)
        retriever = FAISSRetriever(store=store)
        print(f"  FAISS index loaded: {store.index.ntotal} vectors")
        print(f"  Retrieval: k={args.max_passages}, include_gold={args.include_gold}")

    # Initialize vLLM
    from vllm import LLM, SamplingParams
    import torch

    print(f"\nInitializing vLLM...")
    gpu_count = torch.cuda.device_count()
    tp_size = min(args.tensor_parallel_size, gpu_count) if gpu_count > 0 else 1

    llm_kwargs = {
        "model": args.model_name,
        "tensor_parallel_size": tp_size,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=args.lora_path,
        )
        print(f"  LoRA adapter: {args.lora_path}")

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=1.15,
        frequency_penalty=0.1,
    )
    print("vLLM initialized")

    # Generate
    print(f"\nGenerating answers for {len(cases)} cases...")
    t0 = time.time()

    all_results = generate_batch(
        llm=llm,
        sampling_params=sampling_params,
        cases=cases,
        domain=domain,
        no_rag=args.no_rag,
        retriever=retriever,
        max_passages=args.max_passages,
        include_gold=args.include_gold,
        lora_request=lora_request,
    )

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.1f}s ({len(cases)/elapsed:.1f} cases/sec)")

    # ── LLM Extraction for failed parses ──────────────────────────────────
    failed = [r for r in all_results if not r["parse_success"]]
    if failed and not args.no_llm_extraction:
        print(f"\n{len(failed)} cases failed JSON+regex parsing. Running LLM extraction...")

        extractor_llm = None
        if args.extractor_model:
            # Load separate extractor model — need to destroy generator first
            print(f"  Loading extractor model: {args.extractor_model}")
            from vllm.distributed.parallel_state import destroy_model_parallel
            import gc
            destroy_model_parallel()
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            extractor_llm = LLM(
                model=args.extractor_model,
                tensor_parallel_size=tp_size,
                dtype="bfloat16",
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
            )
        else:
            # Reuse generator model
            extractor_llm = llm

        llm_extract_batch(
            llm=llm if not args.extractor_model else None,
            failed_results=failed,
            domain=domain,
            extractor_llm=extractor_llm,
        )

        if args.extractor_model and extractor_llm is not None:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            del extractor_llm
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    n_parsed = sum(1 for r in all_results if r["parse_success"])
    n_correct = sum(1 for r in all_results if r.get("correct", False))
    accuracies = [r["accuracy"] for r in all_results if "accuracy" in r]
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

    summary = {
        "total_cases": len(all_results),
        "parse_success": n_parsed,
        "parse_rate": n_parsed / len(all_results) if all_results else 0.0,
        "correct": n_correct,
        "accuracy": avg_acc,
        "generation_time_sec": elapsed,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total cases:   {summary['total_cases']}")
    print(f"  Parse success: {summary['parse_success']} ({summary['parse_rate']:.1%})")
    print(f"  Correct:       {summary['correct']} / {summary['total_cases']} ({summary['accuracy']:.1%})")

    # Save
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": args.model_name,
        "split": args.split,
        "no_rag": args.no_rag,
        "max_passages": args.max_passages,
        "faiss_index_dir": args.faiss_index_dir,
        "include_gold": args.include_gold,
        "embedding_model": args.embedding_model_name if args.faiss_index_dir else None,
        "retrieval_mode": "faiss" if args.faiss_index_dir else ("gold" if not args.no_rag else "none"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": tp_size,
        "lora_path": args.lora_path,
        "domain": "barexam",
    }

    payload = {
        "config": config,
        "summary": summary,
        "results": all_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
