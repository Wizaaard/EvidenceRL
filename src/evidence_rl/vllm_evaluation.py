#!/usr/bin/env python3
"""vLLM-based evaluation components for EvidenceRL.

This module provides vLLM implementations of the LLM judge,
offering 14-24x faster inference compared to HuggingFace Transformers.

Key classes:
- VLLMAnswerJudge: Grades diagnosis correctness using vLLM

Usage:
    from evidence_rl.vllm_evaluation import VLLMAnswerJudge

    judge = VLLMAnswerJudge(
        model_name="path/to/model",
        tensor_parallel_size=2,
    )
    is_correct = judge.is_correct(prompt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .generation import DEFAULT_MODEL_NAME

# Check vLLM availability
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


def check_vllm_available() -> None:
    """Raise ImportError if vLLM is not available."""
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "For the HuggingFace version, use evaluation.py instead."
        )


@dataclass
class VLLMAnswerJudge:
    """vLLM-based LLM judge for grading answer correctness.

    This class provides the same interface as LLMAnswerJudge
    but uses vLLM for significantly faster inference.

    Key differences from HuggingFace version:
    - Uses vLLM's LLM class with tensor parallelism
    - PagedAttention for efficient memory management
    - Continuous batching for optimal throughput
    - 14-24x faster inference

    Parameters
    ----------
    model_name:
        Model identifier on HuggingFace Hub or local path.
    tensor_parallel_size:
        Number of GPUs for tensor parallelism.
    max_tokens:
        Maximum tokens to generate for verdict.
    gpu_memory_utilization:
        Fraction of GPU memory to use (0.0-1.0).
    generation_kwargs:
        Extra keyword arguments for SamplingParams.
    """

    model_name: str = DEFAULT_MODEL_NAME
    tensor_parallel_size: int = 2
    max_tokens: int = 32
    gpu_memory_utilization: float = 0.90
    generation_kwargs: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        check_vllm_available()

        # Build generation kwargs
        base_kwargs: MutableMapping[str, Any] = {
            "temperature": 0.0,  # Deterministic for judge
            "top_p": 1.0,
        }
        if self.generation_kwargs:
            base_kwargs.update(dict(self.generation_kwargs))
        self._generation_kwargs = dict(base_kwargs)

        # Lazy initialization
        self._llm: Optional[Any] = None
        self._sampling_params: Optional[Any] = None

        # State tracking
        self.last_prompt: Optional[str] = None
        self.last_answer: Optional[str] = None
        self.last_verdicts: Optional[List[str]] = None
        self.last_outputs: Optional[List[str]] = None

    def _init_llm(self) -> None:
        """Lazily initialize the LLM instance."""
        if self._llm is not None:
            return

        import torch

        print(f"[vLLM Judge] Loading model: {self.model_name}")
        print(f"[vLLM Judge] Tensor parallel size: {self.tensor_parallel_size}")
        print(f"[vLLM Judge] GPU memory utilization: {self.gpu_memory_utilization}")

        gpu_count = torch.cuda.device_count()
        print(f"[vLLM Judge] Available GPUs: {gpu_count}")

        tp_size = min(self.tensor_parallel_size, gpu_count) if gpu_count > 0 else 1
        if tp_size != self.tensor_parallel_size:
            print(f"[vLLM Judge] Adjusted tensor parallel size to {tp_size}")

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

        self._sampling_params = SamplingParams(
            temperature=self._generation_kwargs.get("temperature", 0.0),
            top_p=self._generation_kwargs.get("top_p", 1.0),
            max_tokens=self.max_tokens,
        )

        print("[vLLM Judge] Model loaded successfully.")

    @staticmethod
    def _extract_verdict(text: str) -> str:
        """Extract verdict from generated text."""
        return text.strip().lower()

    def is_correct(self, prompt: str) -> bool:
        """Check if a single answer is correct.

        Args:
            prompt: The pre-built judge prompt

        Returns:
            True if verdict starts with 'true', False otherwise
        """
        self._init_llm()

        self.last_prompt = prompt
        outputs = self._llm.generate([prompt], self._sampling_params)

        if not outputs or not outputs[0].outputs:
            self.last_answer = ""
            self.last_verdicts = [""]
            self.last_outputs = []
            return False

        generated = outputs[0].outputs[0].text.strip()
        verdict = self._extract_verdict(generated)

        self.last_answer = verdict
        self.last_verdicts = [verdict]
        self.last_outputs = [generated]

        return verdict.startswith("true")

    def is_correct_batch(
        self,
        prompts: List[Optional[str]],
    ) -> List[bool]:
        """Check if multiple answers are correct using vLLM batch inference.

        vLLM handles batching internally with continuous batching,
        so we pass all prompts at once for optimal throughput.

        Args:
            prompts: List of pre-built judge prompts (None entries are skipped)

        Returns:
            List of boolean verdicts (True = correct, False = incorrect)
        """
        self._init_llm()

        # Handle None prompts
        valid_prompts = [p for p in prompts if p is not None]
        prompt_indices = [i for i, p in enumerate(prompts) if p is not None]

        if not valid_prompts:
            return [False] * len(prompts)

        self.last_prompt = valid_prompts[-1]

        # Generate all at once (vLLM handles batching internally)
        outputs = self._llm.generate(valid_prompts, self._sampling_params)

        # Extract verdicts
        verdicts: List[str] = []
        raw_outputs: List[Optional[str]] = []

        for output in outputs:
            if output.outputs:
                generated = output.outputs[0].text.strip()
                verdicts.append(self._extract_verdict(generated))
                raw_outputs.append(generated)
            else:
                verdicts.append("")
                raw_outputs.append(None)

        # Build result list with None handling
        results = [False] * len(prompts)
        for i, (verdict, idx) in enumerate(zip(verdicts, prompt_indices)):
            results[idx] = verdict.startswith("true")

        self.last_answer = verdicts[-1] if verdicts else None
        self.last_verdicts = verdicts
        self.last_outputs = raw_outputs

        return results


__all__ = [
    "VLLM_AVAILABLE",
    "check_vllm_available",
    "VLLMAnswerJudge",
]
