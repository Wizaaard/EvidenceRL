"""Model-specific configuration for EvidenceRL generation.

This module provides recommended batch sizes and GPU configurations for different
model sizes, optimized for H200 GPUs (141GB VRAM each).

Usage:
    from evidence_rl.model_config import get_model_config, MODEL_CONFIGS

    # Get config for a specific model
    config = get_model_config("gemma-3-4b")
    print(config["batch_size_2gpu"])  # 64

    # Or use the raw configs
    print(MODEL_CONFIGS["gemma-3-4b"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model size.

    Attributes:
        model_size_gb: Approximate model size in GB (fp16).
        batch_size_2gpu: Recommended batch size for 2x H200 GPUs.
        batch_size_4gpu: Recommended batch size for 4x H200 GPUs.
        num_gpus_default: Default number of GPUs to request.
        max_new_tokens: Maximum tokens to generate.
        notes: Additional notes about the model configuration.
    """

    model_size_gb: float
    batch_size_2gpu: int
    batch_size_4gpu: int
    num_gpus_default: int = 4
    max_new_tokens: int = 1024
    notes: str = ""


# Model configurations optimized for H200 GPUs (141GB VRAM each)
# Conservative batch sizes to avoid OOM with long sequences (max_new_tokens=1024, ~2K input)
# Memory = Model weights (fp16) + KV cache (grows with batch × seq_len) + activations
#
# Generation: 4x H200 (564GB total)
# Metrics: 2x H200 (282GB total)
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # Gemma 3 family
    "gemma-3-1b": ModelConfig(
        model_size_gb=2.0,
        batch_size_2gpu=128,
        batch_size_4gpu=256,
        num_gpus_default=4,
        notes="Very small model, can use large batches",
    ),
    "gemma-3-4b": ModelConfig(
        model_size_gb=8.0,
        batch_size_2gpu=64,
        batch_size_4gpu=128,
        num_gpus_default=4,
        notes="Good balance of speed and quality",
    ),
    "gemma-3-12b": ModelConfig(
        model_size_gb=24.0,
        batch_size_2gpu=24,
        batch_size_4gpu=48,
        num_gpus_default=4,
        notes="Higher quality, moderate batch size",
    ),
    "gemma-3-27b": ModelConfig(
        model_size_gb=54.0,
        batch_size_2gpu=12,
        batch_size_4gpu=24,
        num_gpus_default=4,
        notes="High quality, smaller batches for KV cache",
    ),
    # Llama family
    "llama-3.2-1b": ModelConfig(
        model_size_gb=2.5,
        batch_size_2gpu=128,
        batch_size_4gpu=256,
        num_gpus_default=4,
        notes="Very small model, can use large batches",
    ),
    "llama-3.1-8b": ModelConfig(
        model_size_gb=16.0,
        batch_size_2gpu=32,
        batch_size_4gpu=64,
        num_gpus_default=4,
        notes="Strong baseline model",
    ),
    "llama-3.1-70b": ModelConfig(
        model_size_gb=140.0,
        batch_size_2gpu=4,
        batch_size_4gpu=8,
        num_gpus_default=4,
        notes="Large model, requires 4 GPUs, small batches",
    ),
    # Qwen family
    "qwen-2.5-7b": ModelConfig(
        model_size_gb=14.0,
        batch_size_2gpu=32,
        batch_size_4gpu=64,
        num_gpus_default=4,
        notes="Good multilingual support",
    ),
    "qwen-2.5-14b": ModelConfig(
        model_size_gb=28.0,
        batch_size_2gpu=16,
        batch_size_4gpu=32,
        num_gpus_default=4,
        notes="Strong reasoning capabilities",
    ),
    "qwen-2.5-72b": ModelConfig(
        model_size_gb=144.0,
        batch_size_2gpu=4,
        batch_size_4gpu=8,
        num_gpus_default=4,
        notes="Large model, requires 4 GPUs",
    ),
    # Mistral family
    "mistral-7b": ModelConfig(
        model_size_gb=14.0,
        batch_size_2gpu=32,
        batch_size_4gpu=64,
        num_gpus_default=4,
        notes="Fast and efficient",
    ),
    "mixtral-8x7b": ModelConfig(
        model_size_gb=93.0,
        batch_size_2gpu=8,
        batch_size_4gpu=16,
        num_gpus_default=4,
        notes="MoE model, sparse activation helps with memory",
    ),
    # Additional Gemma models
    "gemma-3-270m": ModelConfig(
        model_size_gb=0.5,
        batch_size_2gpu=256,
        batch_size_4gpu=512,
        num_gpus_default=4,
        notes="Tiny model, can use very large batches",
    ),
    "medgemma-4b": ModelConfig(
        model_size_gb=8.0,
        batch_size_2gpu=64,
        batch_size_4gpu=128,
        num_gpus_default=4,
        notes="Medical Gemma 4B, same as gemma-3-4b",
    ),
    "medgemma-27b": ModelConfig(
        model_size_gb=54.0,
        batch_size_2gpu=12,
        batch_size_4gpu=24,
        num_gpus_default=4,
        notes="Medical Gemma 27B, same as gemma-3-27b",
    ),
    # Llama 3.3 and Med42 family
    "llama-3.3-70b": ModelConfig(
        model_size_gb=140.0,
        batch_size_2gpu=4,
        batch_size_4gpu=8,
        num_gpus_default=4,
        notes="Latest Llama 70B, same resources as 3.1-70b",
    ),
    "med42-8b": ModelConfig(
        model_size_gb=16.0,
        batch_size_2gpu=32,
        batch_size_4gpu=64,
        num_gpus_default=4,
        notes="Medical Llama 8B",
    ),
    "med42-70b": ModelConfig(
        model_size_gb=140.0,
        batch_size_2gpu=4,
        batch_size_4gpu=8,
        num_gpus_default=4,
        notes="Medical Llama 70B",
    ),
    # Llama 4 MoE models (sparse activation)
    "llama-4-scout-17b": ModelConfig(
        model_size_gb=35.0,
        batch_size_2gpu=16,
        batch_size_4gpu=32,
        num_gpus_default=4,
        notes="Llama 4 Scout 17B with 16 experts (MoE)",
    ),
    "llama-4-maverick-17b": ModelConfig(
        model_size_gb=35.0,
        batch_size_2gpu=16,
        batch_size_4gpu=32,
        num_gpus_default=4,
        notes="Llama 4 Maverick 17B with 128 experts (MoE)",
    ),
    # GPT-OSS family
    "gpt-oss-20b": ModelConfig(
        model_size_gb=40.0,
        batch_size_2gpu=16,
        batch_size_4gpu=32,
        num_gpus_default=4,
        notes="GPT-OSS 20B model",
    ),
    "gpt-oss-120b": ModelConfig(
        model_size_gb=240.0,
        batch_size_2gpu=2,
        batch_size_4gpu=4,
        num_gpus_default=4,
        notes="GPT-OSS 120B, very large model",
    ),
}

# Aliases for common naming variations
MODEL_ALIASES: Dict[str, str] = {
    # Gemma aliases
    "gemma-3-270m-it": "gemma-3-270m",
    "gemma-3-1b-it": "gemma-3-1b",
    "gemma-3-4b-it": "gemma-3-4b",
    "gemma-3-12b-it": "gemma-3-12b",
    "gemma-3-27b-it": "gemma-3-27b",
    "google/gemma-3-4b-it": "gemma-3-4b",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "google/gemma-3-27b-it": "gemma-3-27b",
    # MedGemma aliases
    "medgemma-4b-it": "medgemma-4b",
    "medgemma-27b-it": "medgemma-27b",
    # Llama 3.x aliases
    "llama-3.2-1b-instruct": "llama-3.2-1b",
    "llama-3.1-8b-instruct": "llama-3.1-8b",
    "llama-3.1-70b-instruct": "llama-3.1-70b",
    "llama-3.3-70b-instruct": "llama-3.3-70b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    # Med42 aliases
    "llama3-med42-8b": "med42-8b",
    "llama3-med42-70b": "med42-70b",
    # Llama 4 aliases
    "llama-4-scout-17b-16e-instruct": "llama-4-scout-17b",
    "llama-4-maverick-17b-128e-instruct": "llama-4-maverick-17b",
    # Qwen aliases
    "qwen-2.5-7b-instruct": "qwen-2.5-7b",
    "qwen-2.5-14b-instruct": "qwen-2.5-14b",
    "qwen-2.5-72b-instruct": "qwen-2.5-72b",
    # Mistral aliases
    "mistral-7b-instruct": "mistral-7b",
    "mixtral-8x7b-instruct": "mixtral-8x7b",
}


def normalize_model_id(model_id: str) -> str:
    """Normalize a model ID to its canonical form.

    Args:
        model_id: Model identifier (path or name).

    Returns:
        Normalized model ID.
    """
    # Handle full paths
    model_name = model_id.split("/")[-1].lower()

    # Remove common suffixes
    for suffix in ["-it", "-instruct", "-chat", "-hf"]:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]

    # Check aliases
    if model_id in MODEL_ALIASES:
        return MODEL_ALIASES[model_id]
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]

    return model_name


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a model.

    Args:
        model_id: Model identifier (supports paths, aliases, and variations).

    Returns:
        ModelConfig if found, None otherwise.

    Examples:
        >>> config = get_model_config("gemma-3-4b-it")
        >>> config.batch_size_2gpu
        64
        >>> config = get_model_config("/path/to/models/gemma-3-4b-it")
        >>> config.batch_size_2gpu
        64
    """
    normalized = normalize_model_id(model_id)

    # Direct lookup
    if normalized in MODEL_CONFIGS:
        return MODEL_CONFIGS[normalized]

    # Fuzzy match - look for partial matches
    for key in MODEL_CONFIGS:
        if key in normalized or normalized in key:
            return MODEL_CONFIGS[key]

    return None


def get_batch_size(model_id: str, num_gpus: int = 2) -> int:
    """Get recommended batch size for a model and GPU configuration.

    Args:
        model_id: Model identifier.
        num_gpus: Number of GPUs available (2 or 4).

    Returns:
        Recommended batch size, or default (32) if model not found.
    """
    config = get_model_config(model_id)
    if config is None:
        # Default conservative batch size
        return 32

    if num_gpus >= 4:
        return config.batch_size_4gpu
    return config.batch_size_2gpu


def get_recommended_gpus(model_id: str) -> int:
    """Get recommended number of GPUs for a model.

    Args:
        model_id: Model identifier.

    Returns:
        Recommended number of GPUs (default: 4).
    """
    config = get_model_config(model_id)
    if config is None:
        return 4
    return config.num_gpus_default


def print_model_configs() -> None:
    """Print all model configurations in a formatted table."""
    print("\nEvidenceRL Model Configurations (for H200 GPUs)")
    print("=" * 80)
    print(f"{'Model':<20} {'Size':<10} {'2xH200 batch':<15} {'4xH200 batch':<15} {'Default GPUs':<12}")
    print("-" * 80)

    for name, config in sorted(MODEL_CONFIGS.items()):
        print(
            f"{name:<20} {config.model_size_gb:>6.1f} GB   "
            f"{config.batch_size_2gpu:>10}     "
            f"{config.batch_size_4gpu:>10}     "
            f"{config.num_gpus_default:>8}"
        )

    print("=" * 80)


if __name__ == "__main__":
    print_model_configs()
