#!/usr/bin/env python3
"""vLLM-based generation components for EvidenceRL.

This module provides vLLM implementations of the generation components,
offering 14-24x faster inference compared to HuggingFace Transformers.

Key classes:
- VLLMStructuredDiagnosisGenerator: Generates structured diagnoses using vLLM
- VLLMDiagnosisExtractor: Fallback extractor for malformed JSON outputs

Usage:
    from evidence_rl.vllm_generation import VLLMStructuredDiagnosisGenerator

    generator = VLLMStructuredDiagnosisGenerator(
        model_name="path/to/model",
        tensor_parallel_size=2,
    )
    output = generator.generate(patient_context, pre_evidence)
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from .documents import RetrievedDocument
from .evidence_pipeline import (
    StructuredDiagnosisOutput,
    DiagnosisWithReasoning,
    _build_structured_diagnosis_prompt,
    _parse_structured_output,
)

# Check vLLM availability
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None

# Check structured output availability (vLLM >= 0.15)
try:
    from vllm.sampling_params import StructuredOutputsParams
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    StructuredOutputsParams = None
    STRUCTURED_OUTPUT_AVAILABLE = False

# JSON schema for guided decoding — forces valid output structure
DIAGNOSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "reasoning"],
            },
            "minItems": 5,
            "maxItems": 5,
        }
    },
    "required": ["diagnoses"],
}


def check_vllm_available() -> None:
    """Raise ImportError if vLLM is not available."""
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "For the HuggingFace version, use evidence_pipeline.py instead."
        )


def _strip_thinking_tokens(text: str) -> str:
    """Strip thinking/analysis channel content from model output.

    Models with channel-based output (e.g., gpt-oss) separate reasoning from
    the final answer using channel markers. This function extracts only the
    'final' channel content, discarding analysis/thinking channels.

    For models without channel markers (gemma, llama, etc.), the text is
    returned unchanged.

    Handles two formats:
    1. Special token format: <|channel|>final (if vLLM preserves delimiters)
    2. Decoded text format: analysis[thinking]assistantfinal{JSON} (vLLM strips
       special token delimiters, leaving only the text components)
    """
    # Format 1: Full special token delimiters preserved
    special_marker = "<|channel|>final"
    if special_marker in text:
        text = text[text.index(special_marker) + len(special_marker):].strip()
        for marker in ("<|end|>", "<|return|>", "<|endoftext|>"):
            if marker in text:
                text = text[:text.index(marker)].strip()
        return text

    # Format 2: Decoded tokens (vLLM strips <|...|> delimiters)
    # gpt-oss output: "analysis[thinking]assistantfinal{JSON}"
    if text.startswith("analysis"):
        boundary = "assistantfinal"
        idx = text.find(boundary)
        if idx >= 0:
            return text[idx + len(boundary):].strip()

    return text


class VLLMDiagnosisExtractor:
    """vLLM-based fallback extractor for malformed JSON outputs.

    Uses vLLM for batch extraction when JSON parsing fails.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 2,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.90,
        llm_instance: Optional[Any] = None,
    ) -> None:
        """Initialize the vLLM extractor.

        Args:
            model_name: HuggingFace model path or name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: Fraction of GPU memory to use
            llm_instance: Optional pre-initialized LLM instance to reuse
        """
        check_vllm_available()

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        # Reuse existing LLM instance if provided
        self._llm = llm_instance
        self._owns_llm = llm_instance is None

        self._sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_tokens,
        )

    def _init_llm(self) -> None:
        """Lazily initialize the LLM instance."""
        if self._llm is not None:
            return

        import torch
        gpu_count = torch.cuda.device_count()
        tp_size = min(self.tensor_parallel_size, gpu_count) if gpu_count > 0 else 1

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

    def _build_extraction_prompt(self, malformed_output: str) -> str:
        """Build a prompt to extract diagnoses from malformed output."""
        return f'''Extract the 5 diagnoses with their reasoning from this malformed output.
Return ONLY a valid JSON array with 5 objects, each with "name" and "reasoning" fields.

Malformed output:
{malformed_output}

Return the JSON array:
['''

    def extract(self, malformed_output: str) -> List[DiagnosisWithReasoning]:
        """Extract diagnoses from a single malformed output."""
        results = self.extract_batch([malformed_output])
        return results[0] if results else []

    def extract_batch(
        self,
        malformed_outputs: List[str],
        show_progress: bool = True,
    ) -> List[List[DiagnosisWithReasoning]]:
        """Extract diagnoses from multiple malformed outputs using vLLM batch inference."""
        self._init_llm()

        if not malformed_outputs:
            return []

        # Build prompts
        prompts = [self._build_extraction_prompt(output) for output in malformed_outputs]

        # Generate with vLLM (handles batching internally)
        outputs = self._llm.generate(prompts, self._sampling_params)

        results = []
        for output in outputs:
            if output.outputs:
                generated = "[" + output.outputs[0].text.strip()
                diagnoses = self._parse_extraction(generated)
                results.append(diagnoses)
            else:
                results.append([])

        return results

    def _parse_extraction(self, text: str) -> List[DiagnosisWithReasoning]:
        """Parse extracted diagnoses from LLM output."""
        diagnoses = []

        # Try to find JSON array
        try:
            # Clean up the text
            text = text.strip()
            if not text.startswith('['):
                text = '[' + text

            # Find closing bracket
            bracket_count = 0
            end_idx = 0
            for i, char in enumerate(text):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > 0:
                text = text[:end_idx]

            data = json.loads(text)
            if isinstance(data, list):
                for item in data[:5]:  # Limit to 5 diagnoses
                    if isinstance(item, dict):
                        name = item.get('name', '') or item.get('diagnosis', '')
                        reasoning = item.get('reasoning', '') or item.get('explanation', '')
                        if name:
                            diagnoses.append(DiagnosisWithReasoning(
                                name=str(name).strip(),
                                reasoning=str(reasoning).strip() if reasoning else "",
                            ))
        except (json.JSONDecodeError, Exception):
            pass

        return diagnoses


class VLLMStructuredDiagnosisGenerator:
    """vLLM-based generator for structured JSON diagnosis output.

    This class provides the same interface as StructuredDiagnosisGenerator
    but uses vLLM for significantly faster inference.

    Key differences from HuggingFace version:
    - Uses vLLM's LLM class with tensor parallelism
    - PagedAttention for efficient memory management
    - Continuous batching for optimal throughput
    - 14-24x faster inference
    - Supports LoRA adapters for DPO-trained models
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 2,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.90,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
        use_llm_extraction: bool = True,
        lora_path: Optional[str] = None,
        max_lora_rank: int = 64,
        use_guided_json: bool = False,
        extractor_model_name: Optional[str] = None,
        num_samples: int = 1,
    ) -> None:
        """Initialize the vLLM generator.

        Args:
            model_name: HuggingFace model path or name (base model for LoRA)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            generation_kwargs: Additional generation parameters
            use_llm_extraction: Whether to use LLM for fallback parsing
            lora_path: Path to LoRA adapter directory (for DPO-trained models).
                      If provided, the model will be loaded with LoRA enabled.
            max_lora_rank: Maximum LoRA rank to support (default: 64)
            use_guided_json: Whether to use guided JSON decoding to enforce valid
                           JSON output schema (virtually guarantees parse success).
            extractor_model_name: Optional path to a separate (larger) model for
                                 fallback extraction. When provided and different from
                                 model_name, the generator model is freed from GPU
                                 before loading the extractor model for extraction.
            num_samples: Number of completions per patient (n parameter for
                        SamplingParams). When > 1, generate_batch returns multiple
                        StructuredDiagnosisOutput per patient in the all_samples field.
        """
        check_vllm_available()

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_llm_extraction = use_llm_extraction
        self.use_guided_json = use_guided_json
        self.lora_path = lora_path
        self.max_lora_rank = max_lora_rank
        self.extractor_model_name = extractor_model_name
        self.num_samples = num_samples

        self._llm: Optional[Any] = None
        self._sampling_params: Optional[Any] = None
        self._generation_kwargs = self._build_generation_kwargs(generation_kwargs)
        self._lora_request: Optional[Any] = None  # LoRARequest instance

        self.last_prompt: Optional[str] = None
        self.llm_extractor: Optional[VLLMDiagnosisExtractor] = None

    def _build_generation_kwargs(self, custom_kwargs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Build generation parameters for vLLM SamplingParams.

        Note: vLLM doesn't support no_repeat_ngram_size like HuggingFace.
        We compensate with higher repetition_penalty and frequency_penalty.
        """
        base_kwargs: MutableMapping[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.15,  # Slightly higher than HF (1.1) to compensate for no no_repeat_ngram_size
            "frequency_penalty": 0.1,    # Additional penalty for repeated tokens (vLLM-specific)
        }
        if custom_kwargs:
            base_kwargs.update(dict(custom_kwargs))
        return dict(base_kwargs)

    def _init_llm(self) -> None:
        """Lazily initialize the LLM instance."""
        if self._llm is not None:
            return

        import torch

        print(f"[vLLM] Loading model: {self.model_name}")
        print(f"[vLLM] Tensor parallel size: {self.tensor_parallel_size}")
        print(f"[vLLM] GPU memory utilization: {self.gpu_memory_utilization}")

        if self.lora_path:
            print(f"[vLLM] LoRA adapter: {self.lora_path}")

        gpu_count = torch.cuda.device_count()
        print(f"[vLLM] Available GPUs: {gpu_count}")

        tp_size = min(self.tensor_parallel_size, gpu_count) if gpu_count > 0 else 1
        if tp_size != self.tensor_parallel_size:
            print(f"[vLLM] Adjusted tensor parallel size to {tp_size}")

        # Build LLM kwargs
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": tp_size,
            "dtype": "bfloat16",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }

        # Enable LoRA support if adapter path is provided
        if self.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
            print(f"[vLLM] LoRA enabled with max_rank={self.max_lora_rank}")

        self._llm = LLM(**llm_kwargs)

        # Create LoRA request if adapter path is provided
        if self.lora_path:
            self._lora_request = LoRARequest(
                lora_name="dpo_adapter",
                lora_int_id=1,
                lora_path=self.lora_path,
            )
            print(f"[vLLM] LoRA request created for adapter: {self.lora_path}")

        # Build sampling params with optional guided JSON decoding
        sampling_kwargs = {
            "temperature": self._generation_kwargs.get("temperature", 0.7),
            "top_p": self._generation_kwargs.get("top_p", 0.9),
            "top_k": self._generation_kwargs.get("top_k", 50),
            "max_tokens": self.max_tokens,
            "repetition_penalty": self._generation_kwargs.get("repetition_penalty", 1.15),
            "frequency_penalty": self._generation_kwargs.get("frequency_penalty", 0.1),
        }

        if self.num_samples > 1:
            sampling_kwargs["n"] = self.num_samples
            print(f"[vLLM] Multi-sample mode: n={self.num_samples} completions per patient")

        if self.use_guided_json and STRUCTURED_OUTPUT_AVAILABLE:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=DIAGNOSIS_JSON_SCHEMA,
            )
            print(f"[vLLM] Structured JSON output ENABLED — output will conform to diagnosis schema")
        elif self.use_guided_json and not STRUCTURED_OUTPUT_AVAILABLE:
            print("[vLLM] WARNING: guided JSON requested but StructuredOutputsParams not available "
                  "(requires vLLM >= 0.15). Falling back to unguided generation.")

        self._sampling_params = SamplingParams(**sampling_kwargs)

        # Initialize LLM extractor
        # Note: For LoRA, we use the base model for extraction to avoid complexity
        if self.use_llm_extraction:
            if self._uses_separate_extractor():
                # Separate extractor model — will be lazily loaded after generator is freed
                print(f"[vLLM] Extractor model configured: {self.extractor_model_name}")
                print(f"[vLLM] Extractor will be loaded on-demand after generation (separate model)")
                self.llm_extractor = VLLMDiagnosisExtractor(
                    model_name=self.extractor_model_name,
                    tensor_parallel_size=tp_size,
                    max_tokens=self.max_tokens,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    llm_instance=None,  # Will create its own LLM instance
                )
            else:
                print("[vLLM] Initializing LLM diagnosis extractor (reusing model)...")
                self.llm_extractor = VLLMDiagnosisExtractor(
                    model_name=self.model_name,
                    tensor_parallel_size=tp_size,
                    max_tokens=self.max_tokens,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    llm_instance=self._llm,
                )

        mode_str = "DPO (with LoRA)" if self.lora_path else "Base Model"
        print(f"[vLLM] Model loaded successfully. Mode: {mode_str}")

    def _uses_separate_extractor(self) -> bool:
        """Check if we're using a separate model for extraction."""
        return (
            self.extractor_model_name is not None
            and self.extractor_model_name != self.model_name
        )

    def _free_generator_llm(self) -> None:
        """Free the generator LLM from GPU memory to make room for the extractor model."""
        if self._llm is not None:
            import gc
            import torch
            print("[vLLM] Freeing generator model from GPU memory...")
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[vLLM] Generator model freed.")

    def generate(
        self,
        patient_context: str,
        pre_evidence: List[RetrievedDocument],
    ) -> StructuredDiagnosisOutput:
        """Generate structured diagnosis output for a single case."""
        self._init_llm()

        prompt = _build_structured_diagnosis_prompt(patient_context, pre_evidence)
        self.last_prompt = prompt

        # Use chat() to apply model's native chat template
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = {"sampling_params": self._sampling_params}
        if self._lora_request:
            chat_kwargs["lora_request"] = self._lora_request

        outputs = self._llm.chat(messages, **chat_kwargs)

        if not outputs or not outputs[0].outputs:
            return StructuredDiagnosisOutput(
                raw_output="",
                parse_success=False,
                parse_error="No output from model",
            )

        generated = _strip_thinking_tokens(outputs[0].outputs[0].text.strip())

        # For single generate with separate extractor, free generator first if parse fails
        if self._uses_separate_extractor():
            result = _parse_structured_output(generated, llm_extractor=None, skip_fallback=True)
            if not result.parse_success and result.raw_output and self.llm_extractor is not None:
                self._free_generator_llm()
                extracted = self.llm_extractor.extract(result.raw_output)
                result.diagnoses = extracted
                complete = sum(1 for d in extracted if d.name and d.reasoning)
                if complete >= 3:
                    result.parse_success = True
                    result.parse_error = None
                else:
                    result.parse_error = f"LLM extraction: only {complete}/5 complete diagnoses"
            return result

        return _parse_structured_output(generated, llm_extractor=self.llm_extractor)

    def generate_batch(
        self,
        contexts_and_evidence: List[Tuple[str, List[RetrievedDocument]]],
        batch_size: int = 4,  # Ignored - vLLM handles batching internally
        show_progress: bool = True,
    ) -> List[StructuredDiagnosisOutput]:
        """Generate structured outputs for multiple cases using vLLM batch inference.

        vLLM handles batching internally with continuous batching,
        so we pass all prompts at once for optimal throughput.
        The batch_size parameter is kept for API compatibility but is ignored.
        """
        self._init_llm()

        # Build all conversations (one per patient case)
        conversations = [
            [{"role": "user", "content": _build_structured_diagnosis_prompt(ctx, evidence)}]
            for ctx, evidence in contexts_and_evidence
        ]

        if conversations:
            self.last_prompt = conversations[-1][0]["content"]

        if not conversations:
            return []

        mode_str = "DPO" if self._lora_request else "Base"
        print(f"[vLLM] Generating {len(conversations)} diagnoses (Mode: {mode_str})...")

        # Generate all at once using chat() to apply native chat templates
        chat_kwargs = {"sampling_params": self._sampling_params}
        if self._lora_request:
            chat_kwargs["lora_request"] = self._lora_request

        outputs = self._llm.chat(conversations, **chat_kwargs)

        # Parse outputs
        results: List[StructuredDiagnosisOutput] = []

        if self.num_samples > 1:
            # Multi-sample mode: collect all N completions per patient
            # Primary result uses first completion; all_samples stores all N
            for output in outputs:
                if not output.outputs:
                    results.append(StructuredDiagnosisOutput(
                        raw_output="",
                        parse_success=False,
                        parse_error="No output from model",
                    ))
                    continue

                # Parse all N completions for this patient
                all_samples = []
                for completion in output.outputs:
                    generated = _strip_thinking_tokens(completion.text.strip())
                    parsed = _parse_structured_output(generated, llm_extractor=None, skip_fallback=True)
                    all_samples.append(parsed)

                # Primary result = first completion (for backward compatibility)
                primary = all_samples[0]
                primary.all_samples = all_samples
                results.append(primary)
        else:
            for output in outputs:
                if not output.outputs:
                    results.append(StructuredDiagnosisOutput(
                        raw_output="",
                        parse_success=False,
                        parse_error="No output from model",
                    ))
                    continue

                generated = _strip_thinking_tokens(output.outputs[0].text.strip())
                # Parse without LLM extraction first (we'll batch failed ones later)
                results.append(_parse_structured_output(generated, llm_extractor=None, skip_fallback=True))

        # Batch LLM extraction for all failed parses
        # For multi-sample mode, extract for ALL failed samples across all patients
        if self.llm_extractor is not None:
            if self.num_samples > 1:
                # Collect all failed samples across all patients
                failed_refs = []  # (result_idx, sample_idx)
                failed_raw_outputs = []
                for ri, result in enumerate(results):
                    if result.all_samples:
                        for si, sample in enumerate(result.all_samples):
                            if not sample.parse_success and sample.raw_output:
                                failed_refs.append((ri, si))
                                failed_raw_outputs.append(sample.raw_output)

                if failed_refs:
                    if self._uses_separate_extractor():
                        self._free_generator_llm()
                    if show_progress:
                        print(f"[vLLM] Batched LLM extraction for {len(failed_refs)} failed samples...")
                    extracted_diagnoses = self.llm_extractor.extract_batch(
                        failed_raw_outputs, show_progress=show_progress,
                    )
                    for (ri, si), diagnoses in zip(failed_refs, extracted_diagnoses):
                        sample = results[ri].all_samples[si]
                        sample.diagnoses = diagnoses
                        complete = sum(1 for d in diagnoses if d.name and d.reasoning)
                        if complete >= 3:
                            sample.parse_success = True
                            sample.parse_error = None
                        else:
                            sample.parse_error = f"LLM extraction: only {complete}/5 complete diagnoses"
                    # Update primary result if it was the one that failed
                    for ri, result in enumerate(results):
                        if result.all_samples:
                            primary = result.all_samples[0]
                            result.diagnoses = primary.diagnoses
                            result.raw_output = primary.raw_output
                            result.parse_success = primary.parse_success
                            result.parse_error = primary.parse_error
            else:
                failed_indices = [
                    i for i, r in enumerate(results)
                    if not r.parse_success and r.raw_output
                ]

                if failed_indices:
                    failed_raw_outputs = [results[i].raw_output for i in failed_indices]

                    # If using a separate extractor model, free the generator first
                    if self._uses_separate_extractor():
                        self._free_generator_llm()

                    if show_progress:
                        print(f"[vLLM] Batched LLM extraction for {len(failed_indices)} failed parses...")

                    extracted_diagnoses = self.llm_extractor.extract_batch(
                        failed_raw_outputs,
                        show_progress=show_progress,
                    )

                    # Update results with extracted diagnoses
                    for idx, diagnoses in zip(failed_indices, extracted_diagnoses):
                        results[idx].diagnoses = diagnoses
                        complete_diagnoses = sum(1 for d in diagnoses if d.name and d.reasoning)
                        if complete_diagnoses >= 3:
                            results[idx].parse_success = True
                            results[idx].parse_error = None
                        else:
                            results[idx].parse_error = f"LLM extraction: only {complete_diagnoses}/5 complete diagnoses"

        total = len(results) * max(self.num_samples, 1)
        print(f"[vLLM] Generated {total} completions for {len(results)} patients.")
        return results


__all__ = [
    "VLLM_AVAILABLE",
    "check_vllm_available",
    "VLLMStructuredDiagnosisGenerator",
    "VLLMDiagnosisExtractor",
]
