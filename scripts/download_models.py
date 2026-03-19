#!/usr/bin/env python3
"""
Download all experiment models from HuggingFace.

Usage:
    python scripts/download_models.py [--dry-run] [--model MODEL_NAME]

Requirements:
    pip install huggingface_hub

Authentication:
    hf auth login
    (Required for gated models like Llama)
"""

import argparse
import os
from pathlib import Path

# Try importing huggingface_hub
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    exit(1)


def get_token():
    """Get HuggingFace token from cached login (preferred) or environment."""
    # Prefer cached token file (from 'hf auth login')
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    # Fall back to environment variable
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# Model directory (override with --model-dir or MODEL_BASE_DIR env var)
MODEL_DIR = Path(os.environ.get("MODEL_BASE_DIR", "//storage/ice-shared/bmed-sp-wang/Models"))

# Model mapping: local_name -> huggingface_id
# Verified against HuggingFace on 2026-01-27
MODELS = {
    # === Small Models (verified) ===
    "gemma-3-270m-it": "google/gemma-3-270m-it", # not
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct", # done
    "gemma-3-1b-it": "google/gemma-3-1b-it", # done
    "gemma-3-4b-it": "google/gemma-3-4b-it", # do
    "medgemma-4b-it": "google/medgemma-4b-it", # not
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct", # do
    "Llama3-Med42-8B": "m42-health/Llama3-Med42-8B", # not
    "gemma-3-12b-it": "google/gemma-3-12b-it", # do

    # === Medium Models (verified) ===
    "gemma-3-27b-it": "google/gemma-3-27b-it", # do
    "medgemma-27b-it": "google/medgemma-27b-it", # not
    "gpt-oss-20b": "openai/gpt-oss-20b", # do

    # === Large Models (verified) ===
    "Llama3-Med42-70B": "m42-health/Llama3-Med42-70B", # not
    "gpt-oss-120b": "openai/gpt-oss-120b", # not
    "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct", # not
    "Llama-4-Maverick-17B-128E-Instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct", # not
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct", # not
    "Llama-3.1-405B": "meta-llama/Llama-3.1-405B", #big
    "Med42-v2-70B": "m42-health/Llama3-Med42-70B", # alternative judge
}

# Categorize models by size (all verified on HuggingFace)
SMALL_MODELS = [
    "gemma-3-270m-it",
    "Llama-3.2-1B-Instruct",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "medgemma-4b-it",
    "Llama3-Med42-8B",
    "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it",
]

MEDIUM_MODELS = [
    "gemma-3-27b-it",
    "medgemma-27b-it",
    "gpt-oss-20b",
]

LARGE_MODELS = [
    "Llama3-Med42-70B",
    "Med42-v2-70B",
    "Llama-4-Scout-17B-16E-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct",
    "Llama-3.3-70B-Instruct",
    "gpt-oss-120b",
    "Llama-3.1-405B",
]


def is_download_complete(local_path: Path) -> bool:
    """Check if a model download is complete (has weight files)."""
    if not local_path.exists():
        return False
    # Check for common model weight file patterns
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth", "model*.safetensors"]
    for pattern in weight_patterns:
        if list(local_path.glob(pattern)):
            return True
    return False


def download_model(local_name: str, hf_id: str, token: str = None, dry_run: bool = False) -> str:
    """Download a single model from HuggingFace.

    Returns: "skipped", "success", or "failed"
    """
    local_path = MODEL_DIR / local_name

    print(f"\n{'='*60}")
    print(f"Model: {local_name}")
    print(f"HuggingFace ID: {hf_id}")
    print(f"Local path: {local_path}")

    # Check if already downloaded (with weight files, not just metadata)
    if is_download_complete(local_path):
        print(f"  [SKIP] Already exists at {local_path}")
        return "skipped"

    if dry_run:
        print(f"  [DRY-RUN] Would download to {local_path}")
        return "success"

    try:
        print(f"  [DOWNLOADING] Starting download...")
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(local_path),
            token=token,
        )
        print(f"  [SUCCESS] Downloaded to {local_path}")
        return "success"

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"  [ERROR] Authentication required. Run: hf auth login")
            print(f"          Make sure you have access to {hf_id}")
        elif "404" in str(e):
            print(f"  [ERROR] Model not found: {hf_id}")
            print(f"          Please verify the HuggingFace model ID")
        else:
            print(f"  [ERROR] HTTP error: {e}")
        return "failed"

    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        return "failed"


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Download only a specific model (by local name)"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Download only models of a specific size category"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to download models to (default: MODEL_BASE_DIR env var or //storage/ice-shared/bmed-sp-wang/Models)"
    )
    args = parser.parse_args()

    # Override MODEL_DIR if --model-dir is specified
    global MODEL_DIR
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)

    # List models and exit
    if args.list:
        print("\n=== Available Models ===\n")
        print("SMALL:")
        for name in SMALL_MODELS:
            print(f"  {name} -> {MODELS[name]}")
        print("\nMEDIUM:")
        for name in MEDIUM_MODELS:
            print(f"  {name} -> {MODELS[name]}")
        print("\nLARGE:")
        for name in LARGE_MODELS:
            print(f"  {name} -> {MODELS[name]}")
        return

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODEL_DIR}")

    # Get authentication token
    token = get_token()
    if token:
        print("Authentication: Token found")
    else:
        print("Authentication: No token found (may fail for gated models)")

    # Determine which models to download
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: Unknown model '{args.model}'")
            print(f"Available models: {list(MODELS.keys())}")
            return
        models_to_download = {args.model: MODELS[args.model]}
    elif args.size == "small":
        models_to_download = {k: MODELS[k] for k in SMALL_MODELS}
    elif args.size == "medium":
        models_to_download = {k: MODELS[k] for k in MEDIUM_MODELS}
    elif args.size == "large":
        models_to_download = {k: MODELS[k] for k in LARGE_MODELS}
    else:
        models_to_download = MODELS

    print(f"\nModels to download: {len(models_to_download)}")
    if args.dry_run:
        print("[DRY-RUN MODE - No actual downloads]")

    # Download models
    results = {"success": [], "failed": [], "skipped": []}

    for local_name, hf_id in models_to_download.items():
        status = download_model(local_name, hf_id, token=token, dry_run=args.dry_run)
        results[status].append(local_name)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(models_to_download)}")
    print(f"Downloaded:   {len(results['success'])}")
    print(f"Skipped:      {len(results['skipped'])} (already exist)")
    print(f"Failed:       {len(results['failed'])}")

    if results["failed"]:
        print(f"\nFailed models:")
        for name in results["failed"]:
            print(f"  - {name} ({MODELS[name]})")
        print("\nTips:")
        print("  1. Run 'hf auth login' to authenticate")
        print("  2. Verify model IDs in the MODELS dict at the top of this script")
        print("  3. Check if you have access to gated models (Llama, etc.)")


if __name__ == "__main__":
    main()
