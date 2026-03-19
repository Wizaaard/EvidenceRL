#!/usr/bin/env python3
"""
vLLM Environment Verification Script

Checks that all required packages are installed and compatible for running
the vLLM reasoning revision pipeline.

Usage:
    python check_vllm_environment.py
    python check_vllm_environment.py --model medgemma-27b-it  # Also test model loading
    python check_vllm_environment.py --quick                   # Skip slow tests

Exit codes:
    0 - All checks passed
    1 - Some checks failed (see output for details)
"""

import argparse
import sys
import os
from typing import Tuple, List, Optional


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_check(name: str, status: bool, details: str = "") -> None:
    """Print a check result."""
    icon = "✓" if status else "✗"
    color_start = "\033[92m" if status else "\033[91m"
    color_end = "\033[0m"

    if details:
        print(f"  {color_start}{icon}{color_end} {name}: {details}")
    else:
        print(f"  {color_start}{icon}{color_end} {name}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version (vLLM requires 3.8+)."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        return True, version_str
    else:
        return False, f"{version_str} (requires 3.8+)"


def check_vllm() -> Tuple[bool, str]:
    """Check vLLM installation."""
    try:
        import vllm
        version = getattr(vllm, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, f"Not installed ({e})"
    except Exception as e:
        return False, f"Error: {e}"


def check_torch() -> Tuple[bool, str]:
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"

        if cuda_available:
            return True, f"{version} (CUDA {cuda_version})"
        else:
            return False, f"{version} (CUDA not available)"
    except ImportError:
        return False, "Not installed"
    except Exception as e:
        return False, f"Error: {e}"


def check_transformers() -> Tuple[bool, str]:
    """Check transformers installation."""
    try:
        import transformers
        version = transformers.__version__
        # vLLM typically needs transformers >= 4.30.0
        major, minor = map(int, version.split(".")[:2])
        if major >= 4 and minor >= 30:
            return True, version
        else:
            return False, f"{version} (recommend >= 4.30.0)"
    except ImportError:
        return False, "Not installed"
    except Exception as e:
        return False, f"Error: {e}"


def check_gpu_count() -> Tuple[bool, str]:
    """Check number of available GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        count = torch.cuda.device_count()
        if count >= 2:
            return True, f"{count} GPUs available"
        elif count == 1:
            return True, f"1 GPU available (tensor_parallel_size must be 1)"
        else:
            return False, "No GPUs found"
    except Exception as e:
        return False, f"Error: {e}"


def check_gpu_memory() -> Tuple[bool, str]:
    """Check GPU memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            gpu_info.append(f"GPU {i}: {props.name} ({mem_gb:.1f}GB)")

        return True, "; ".join(gpu_info)
    except Exception as e:
        return False, f"Error: {e}"


def check_flash_attention() -> Tuple[bool, str]:
    """Check Flash Attention availability (optional but recommended)."""
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        return True, f"v{version}"
    except ImportError:
        return True, "Not installed (optional, vLLM will use alternative)"
    except Exception as e:
        return True, f"Check skipped: {e}"


def check_sentence_transformers() -> Tuple[bool, str]:
    """Check sentence-transformers for embedding model."""
    try:
        import sentence_transformers
        version = sentence_transformers.__version__
        return True, version
    except ImportError:
        return False, "Not installed (needed for NLI)"
    except Exception as e:
        return False, f"Error: {e}"


def check_triton() -> Tuple[bool, str]:
    """Check Triton (optional, for some vLLM optimizations)."""
    try:
        import triton
        version = getattr(triton, "__version__", "unknown")
        return True, version
    except ImportError:
        return True, "Not installed (optional)"
    except Exception as e:
        return True, f"Check skipped: {e}"


def check_vllm_sampling_params() -> Tuple[bool, str]:
    """Check vLLM SamplingParams API."""
    try:
        from vllm import SamplingParams

        # Test creating sampling params
        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
        return True, "SamplingParams API works"
    except ImportError:
        return False, "vLLM not installed"
    except Exception as e:
        return False, f"API error: {e}"


def check_vllm_llm_class() -> Tuple[bool, str]:
    """Check vLLM LLM class is available."""
    try:
        from vllm import LLM
        return True, "LLM class available"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_model_loading(model_path: str) -> Tuple[bool, str]:
    """Test actually loading a model with vLLM."""
    try:
        from vllm import LLM, SamplingParams
        import torch

        if not os.path.exists(model_path):
            return False, f"Model path not found: {model_path}"

        gpu_count = torch.cuda.device_count()
        tp_size = min(2, gpu_count)

        print(f"\n  Loading model with tensor_parallel_size={tp_size}...")
        print(f"  This may take a few minutes...")

        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.5,  # Use less memory for test
            trust_remote_code=True,
        )

        # Test generation
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=32,
        )

        outputs = llm.generate(["Hello, my name is"], sampling_params)

        if outputs and outputs[0].outputs:
            generated = outputs[0].outputs[0].text[:50]
            return True, f"Model loaded and generated: '{generated}...'"
        else:
            return False, "Model loaded but generation failed"

    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_environment_variables() -> Tuple[bool, str]:
    """Check relevant environment variables."""
    vars_to_check = [
        "CUDA_VISIBLE_DEVICES",
        "VLLM_ATTENTION_BACKEND",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    ]

    found = []
    for var in vars_to_check:
        value = os.environ.get(var)
        if value:
            found.append(f"{var}={value}")

    if found:
        return True, "; ".join(found)
    else:
        return True, "No special env vars set (using defaults)"


def run_all_checks(test_model: Optional[str] = None, quick: bool = False) -> bool:
    """Run all environment checks."""
    all_passed = True

    print_header("Python Environment")

    passed, details = check_python_version()
    print_check("Python version", passed, details)
    all_passed = all_passed and passed

    passed, details = check_environment_variables()
    print_check("Environment variables", passed, details)

    print_header("Core Dependencies")

    passed, details = check_torch()
    print_check("PyTorch", passed, details)
    all_passed = all_passed and passed

    passed, details = check_transformers()
    print_check("Transformers", passed, details)
    all_passed = all_passed and passed

    passed, details = check_vllm()
    print_check("vLLM", passed, details)
    vllm_installed = passed
    all_passed = all_passed and passed

    print_header("GPU Configuration")

    passed, details = check_gpu_count()
    print_check("GPU count", passed, details)
    all_passed = all_passed and passed

    passed, details = check_gpu_memory()
    print_check("GPU memory", passed, details)

    print_header("Optional Dependencies")

    passed, details = check_flash_attention()
    print_check("Flash Attention", passed, details)

    passed, details = check_triton()
    print_check("Triton", passed, details)

    passed, details = check_sentence_transformers()
    print_check("Sentence Transformers", passed, details)

    if vllm_installed:
        print_header("vLLM API Tests")

        passed, details = check_vllm_llm_class()
        print_check("LLM class", passed, details)
        all_passed = all_passed and passed

        passed, details = check_vllm_sampling_params()
        print_check("SamplingParams", passed, details)
        all_passed = all_passed and passed

    if test_model and not quick:
        print_header("Model Loading Test")
        passed, details = test_model_loading(test_model)
        print_check("Model loading", passed, details)
        all_passed = all_passed and passed

    # Summary
    print_header("Summary")
    if all_passed:
        print("  \033[92m✓ All checks passed! Environment is ready for vLLM.\033[0m")
    else:
        print("  \033[91m✗ Some checks failed. See above for details.\033[0m")
        print("\n  To fix common issues:")
        print("    - Install vLLM: pip install vllm")
        print("    - Update transformers: pip install transformers>=4.30.0")
        print("    - Check CUDA: nvidia-smi")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Check vLLM environment compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model to test loading (optional, slow)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow tests (model loading)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  vLLM Environment Verification")
    print("=" * 60)

    success = run_all_checks(
        test_model=args.model,
        quick=args.quick,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
