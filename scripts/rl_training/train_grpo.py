#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training script for EvidenceRL.

Trains a base model with LoRA using online RL with three reward signals:
  R_format:    Valid JSON format compliance
  R_accuracy:  Clinical diagnosis correctness (BioLORD-2023 similarity)
  R_grounding: NLI-based evidence grounding (PubMedBERT-MNLI-MedNLI)

Uses TRL GRPOTrainer with vLLM for fast generation.

Usage:
    python train_grpo.py --config grpo_config.yaml
    python train_grpo.py \
        --base-model /path/to/model \
        --dataset /path/to/grpo_norag_prompts.json \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for EvidenceRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base-model", type=str, help="Path to base model")
    parser.add_argument("--dataset", type=str, help="Path to GRPO dataset JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (QLoRA, for large models like 70B)")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM, use transformers for generation")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (for logging only, accelerate handles distribution)")
    parser.add_argument("--vllm-tp", type=int, default=None, help="vLLM tensor parallel size (default: 1, set >1 for large models)")
    parser.add_argument("--vllm-gpu-util", type=float, default=None, help="vLLM GPU memory utilization (default: 0.4)")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max model context length (default: 10000)")
    parser.add_argument("--num-iterations", type=int, default=None, help="Policy update iterations per generation batch (default: 2)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--reward-weight-grounding", type=float, default=None,
                        help="Override grounding reward weight w_g (default: 2.0). "
                             "Set to 0.0 for accuracy-only ablation.")
    parser.add_argument("--no-sentence-level", action="store_true",
                        help="Use grounding_max instead of grounding_avg for grounding reward")

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config = _default_config()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = _merge_config(config, yaml_config)

    # CLI overrides
    if args.base_model:
        config['model']['base_model'] = args.base_model
    if args.dataset:
        config['data']['dataset_path'] = args.dataset
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.run_name:
        config['training']['run_name'] = args.run_name
    if args.epochs is not None:
        config['training']['num_train_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['training']['per_device_train_batch_size'] = args.batch_size
    if args.grad_accum is not None:
        config['training']['gradient_accumulation_steps'] = args.grad_accum
    if args.num_generations is not None:
        config['grpo']['num_generations'] = args.num_generations
    if args.max_completion_length is not None:
        config['grpo']['max_completion_length'] = args.max_completion_length
    if args.temperature is not None:
        config['grpo']['temperature'] = args.temperature
    if args.beta is not None:
        config['grpo']['beta'] = args.beta
    if args.lora_r is not None:
        config['lora']['r'] = args.lora_r
    if args.lora_alpha is not None:
        config['lora']['lora_alpha'] = args.lora_alpha
    if args.load_in_4bit:
        config['model']['load_in_4bit'] = True
    if args.no_vllm:
        config['vllm']['use_vllm'] = False
    if args.vllm_tp is not None:
        config['vllm']['tensor_parallel_size'] = args.vllm_tp
    if args.vllm_gpu_util is not None:
        config['vllm']['gpu_memory_utilization'] = args.vllm_gpu_util
    if args.max_model_len is not None:
        config['vllm']['max_model_len'] = args.max_model_len
    if args.num_iterations is not None:
        config['grpo']['num_iterations'] = args.num_iterations
    if args.resume_from_checkpoint:
        config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint
    if args.reward_weight_grounding is not None:
        config['grpo']['reward_weights'][2] = args.reward_weight_grounding

    # Validate required fields
    if not config['model']['base_model']:
        parser.error("--base-model or config model.base_model is required")
    if not config['data']['dataset_path']:
        parser.error("--dataset or config data.dataset_path is required")
    if not config['training']['output_dir']:
        parser.error("--output-dir or config training.output_dir is required")

    # ── Print config ─────────────────────────────────────────────────
    print("=" * 70)
    print("GRPO TRAINING CONFIGURATION")
    print("=" * 70)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 70)

    # ── Imports (heavy, do after config validation) ──────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    # ── Pin each process to its own GPU early ─────────────────────────
    # accelerate sets LOCAL_RANK env var. Without early pinning, all
    # processes load models onto cuda:0, causing OOM/device-busy errors
    # for large models like gpt-oss-20b (~40GB bf16 × N processes).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"  [init] Process pinned to cuda:{local_rank}")

    # ── Patch vLLM for BitsAndBytes 4-bit compatibility in colocate mode ──
    # BitsAndBytes sets bnb_quant_state on weight tensors during 4-bit loading.
    # vLLM's set_weight_attrs asserts no existing attributes, causing a crash.
    # This patch skips attributes that already exist (safe: BnB already set them).
    if config['model'].get('load_in_4bit', False) and config['vllm'].get('use_vllm', False):
        import vllm.model_executor.utils as _vllm_utils
        _orig_set_weight_attrs = _vllm_utils.set_weight_attrs

        def _patched_set_weight_attrs(weight, weight_attrs):
            if weight_attrs is None:
                return
            # Filter out attributes already set by BitsAndBytes
            filtered = {k: v for k, v in weight_attrs.items()
                        if not hasattr(weight, k)}
            if filtered:
                _orig_set_weight_attrs(weight, filtered)

        _vllm_utils.set_weight_attrs = _patched_set_weight_attrs
        print("  [patch] vLLM set_weight_attrs patched for BnB 4-bit compatibility")

    # Import reward functions
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grpo_reward import (
        format_reward,
        accuracy_reward,
        grounding_reward,
        initialize_reward_models,
        set_sentence_level,
    )

    # Set grounding reward mode
    if args.no_sentence_level:
        set_sentence_level(False)

    # ── Load dataset ─────────────────────────────────────────────────
    print("\nLoading GRPO dataset...")
    dataset_path = config['data']['dataset_path']
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)

    prompts_list = raw_data['prompts']
    print(f"  Total prompts: {len(prompts_list)}")
    print(f"  Dataset config: {raw_data.get('config', {})}")

    # Convert to HF Dataset
    # GRPOTrainer expects a "prompt" column (list of chat messages)
    # Additional columns are passed to reward functions as kwargs
    hf_data = Dataset.from_list([
        {
            "prompt": entry["prompt"],
            "ground_truth_diagnoses": entry["ground_truth_diagnoses"],
            "patient_context": entry["patient_context"],
            "pre_evidence": entry["pre_evidence"],
        }
        for entry in prompts_list
    ])

    print(f"  Dataset columns: {hf_data.column_names}")
    print(f"  Dataset size: {len(hf_data)}")

    # ── Load tokenizer ───────────────────────────────────────────────
    print("\nLoading tokenizer...")
    base_model_path = config['model']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=config['model']['trust_remote_code'],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    # ── Load model (base only, no SFT merge) ─────────────────────────
    print(f"\nLoading base model: {base_model_path}")
    model_kwargs = {
        "torch_dtype": getattr(torch, config['model']['torch_dtype']),
        "trust_remote_code": config['model']['trust_remote_code'],
    }

    if config['model']['attn_implementation']:
        model_kwargs["attn_implementation"] = config['model']['attn_implementation']

    if config['model'].get('load_in_4bit', False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # Pin each process to its own GPU so BitsAndBytes doesn't load all on GPU 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model_kwargs["device_map"] = {"": local_rank}
        print(f"  Using 4-bit quantization (QLoRA) on device {local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )

    # ── LoRA config ──────────────────────────────────────────────────
    print("\nConfiguring LoRA...")
    lora_cfg = config['lora']
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        bias=lora_cfg['bias'],
        task_type="CAUSAL_LM",
    )
    print(f"  LoRA rank: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")

    # ── Initialize reward models (pre-load on CPU) ───────────────────
    print("\nInitializing reward models...")
    initialize_reward_models()

    # ── Build GRPOConfig ─────────────────────────────────────────────
    tc = config['training']
    gc = config['grpo']
    vc = config['vllm']
    output_dir = tc['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    grpo_config_kwargs = dict(
        output_dir=output_dir,
        run_name=tc.get('run_name') or Path(output_dir).name,
        num_train_epochs=tc['num_train_epochs'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        gradient_checkpointing=tc.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=tc['learning_rate'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        warmup_ratio=tc['warmup_ratio'],
        max_grad_norm=tc['max_grad_norm'],
        bf16=tc['bf16'],
        fp16=tc['fp16'],
        # GRPO algorithm parameters
        num_generations=gc['num_generations'],
        max_completion_length=gc['max_completion_length'],
        max_prompt_length=gc['max_prompt_length'],
        temperature=gc['temperature'],
        beta=gc['beta'],
        loss_type=gc['loss_type'],
        scale_rewards=gc['scale_rewards'],
        epsilon=gc['epsilon'],
        # Multi-objective reward
        reward_weights=gc['reward_weights'],
        # Reuse generations for multiple policy updates (free gradient updates)
        num_iterations=gc.get('num_iterations', 2),
        # Liger kernel for fused ops (faster forward/backward)
        use_liger_kernel=True,
        # Checkpointing
        save_strategy="steps",
        save_steps=tc['save_steps'],
        save_total_limit=tc['save_total_limit'],
        logging_steps=tc['logging_steps'],
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=tc['report_to'],
        seed=tc['seed'],
        log_completions=True,
        num_completions_to_print=2,
    )

    # Add vLLM config if enabled
    if vc['use_vllm']:
        grpo_config_kwargs.update(
            use_vllm=True,
            vllm_mode=vc['mode'],
            vllm_tensor_parallel_size=vc['tensor_parallel_size'],
            vllm_gpu_memory_utilization=vc['gpu_memory_utilization'],
            vllm_enable_sleep_mode=vc.get('enable_sleep_mode', False),
        )
        if vc.get('max_model_len') is not None:
            grpo_config_kwargs['vllm_max_model_length'] = vc['max_model_len']
        print(f"\n  vLLM enabled: mode={vc['mode']}, tp={vc['tensor_parallel_size']}, "
              f"gpu_util={vc['gpu_memory_utilization']}, "
              f"max_model_len={vc.get('max_model_len', 'auto')}, "
              f"sleep={vc.get('enable_sleep_mode', False)}")
    else:
        print("\n  vLLM disabled, using transformers for generation")

    training_args = GRPOConfig(**grpo_config_kwargs)

    # ── Initialize GRPOTrainer ───────────────────────────────────────
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_data,
        reward_funcs=[format_reward, accuracy_reward, grounding_reward],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Print training stats
    n_train = len(hf_data)
    effective_batch = tc['per_device_train_batch_size'] * tc['gradient_accumulation_steps'] * max(torch.cuda.device_count(), 1)
    steps_per_epoch = n_train // effective_batch
    total_steps = steps_per_epoch * tc['num_train_epochs']
    num_iters = gc.get('num_iterations', 2)
    print(f"\n  Training prompts:       {n_train}")
    print(f"  Generations per prompt: {gc['num_generations']}")
    print(f"  Effective batch size:   {effective_batch} prompts")
    print(f"  Steps per epoch:        {steps_per_epoch}")
    print(f"  Total training steps:   {total_steps}")
    print(f"  Num iterations/gen:     {num_iters} (policy updates per generation batch)")
    print(f"  Liger kernel:           enabled")
    print(f"  Reward weights:         format={gc['reward_weights'][0]}, "
          f"accuracy={gc['reward_weights'][1]}, grounding={gc['reward_weights'][2]}")

    # ── Resolve checkpoint resumption ─────────────────────────────────
    resume = tc.get('resume_from_checkpoint')
    if resume == "auto":
        import glob
        ckpt_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda d: int(os.path.basename(d).split("-")[1]),
        )
        if ckpt_dirs:
            resume = ckpt_dirs[-1]
            print(f"\n  Auto-resume: resuming from {resume}")
        else:
            resume = None
            print("\n  Auto-resume: no checkpoints found, starting from scratch")

    # ── Train ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING GRPO TRAINING")
    print("=" * 70 + "\n")

    trainer.train(resume_from_checkpoint=resume)

    # ── Save final model ─────────────────────────────────────────────
    print("\nSaving final model...")
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training log
    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"  Training log saved to: {log_path}")

    print("\n" + "=" * 70)
    print("GRPO TRAINING COMPLETE")
    print(f"  Final model: {final_dir}")
    print(f"  Training log: {log_path}")
    print("=" * 70)


def _default_config():
    """Default GRPO training configuration."""
    return {
        'model': {
            'base_model': None,
            'torch_dtype': 'bfloat16',
            'attn_implementation': None,
            'trust_remote_code': False,
            'load_in_4bit': False,
        },
        'lora': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            'bias': 'none',
        },
        'data': {
            'dataset_path': None,
        },
        'grpo': {
            'num_generations': 8,
            'max_completion_length': 4096,
            'max_prompt_length': 6144,
            'temperature': 0.7,
            'beta': 0.04,
            'loss_type': 'grpo',
            'scale_rewards': 'group',
            'epsilon': 0.2,
            'reward_weights': [1.0, 1.0, 2.0],
            'num_iterations': 2,
        },
        'vllm': {
            'use_vllm': True,
            'mode': 'colocate',
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.4,
            'max_model_len': None,  # None = vLLM auto-detects; set explicitly for large models
            'enable_sleep_mode': True,
        },
        'training': {
            'num_train_epochs': 2,
            'learning_rate': 5e-6,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'gradient_checkpointing': True,
            'warmup_ratio': 0.05,
            'lr_scheduler_type': 'cosine',
            'max_grad_norm': 1.0,
            'bf16': True,
            'fp16': False,
            'logging_steps': 5,
            'save_steps': 50,
            'save_total_limit': 3,
            'seed': 42,
            'report_to': 'none',
            'output_dir': None,
            'run_name': None,
            'resume_from_checkpoint': None,
        },
    }


def _merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override into base config."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
