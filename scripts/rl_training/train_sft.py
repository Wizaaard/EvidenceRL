#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) training script for EvidenceRL baseline.

Fine-tunes a base model on Evidence-Based outputs using LoRA + TRL SFTTrainer.
Matches the DPO training configuration (LoRA r=16, alpha=32, same target modules).

Usage:
    python train_sft.py --config sft_config.yaml
    python train_sft.py \
        --base-model /path/to/model \
        --dataset /path/to/sft_rag_eb2_cap2.json \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="SFT training for EvidenceRL baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base-model", type=str, help="Path to base model")
    parser.add_argument("--dataset", type=str, help="Path to SFT dataset JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

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
    if args.max_seq_length is not None:
        config['training']['max_seq_length'] = args.max_seq_length
    if args.lora_r is not None:
        config['lora']['r'] = args.lora_r
    if args.lora_alpha is not None:
        config['lora']['lora_alpha'] = args.lora_alpha
    if args.val_split is not None:
        config['data']['val_split'] = args.val_split
    if args.resume_from_checkpoint:
        config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint

    # Validate required fields
    if not config['model']['base_model']:
        parser.error("--base-model or config model.base_model is required")
    if not config['data']['dataset_path']:
        parser.error("--dataset or config data.dataset_path is required")
    if not config['training']['output_dir']:
        parser.error("--output-dir or config training.output_dir is required")

    # ── Print config ─────────────────────────────────────────────────
    print("=" * 70)
    print("SFT TRAINING CONFIGURATION")
    print("=" * 70)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 70)

    # ── Imports (heavy, do after config validation) ──────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    # ── Load dataset ─────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset_path = config['data']['dataset_path']
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)

    examples = raw_data['examples']
    print(f"  Total examples: {len(examples)}")
    print(f"  Dataset config: {raw_data.get('config', {})}")

    # Convert to HF Dataset format using prompt/completion split.
    # We use the prompt-completion format instead of "messages" because TRL's
    # assistant_only_loss with "messages" requires {% generation %} Jinja tags
    # in the chat template, which most model tokenizers don't have.
    # The prompt-completion path creates a completion_mask from token lengths
    # instead, achieving the same result without template modifications.
    hf_data = Dataset.from_list([
        {
            "prompt": [msg for msg in ex["messages"] if msg["role"] != "assistant"],
            "completion": [msg for msg in ex["messages"] if msg["role"] == "assistant"],
        }
        for ex in examples
    ])

    # Train/val split
    val_split = config['data']['val_split']
    if val_split > 0:
        split = hf_data.train_test_split(
            test_size=val_split,
            seed=config['training']['seed'],
        )
        train_dataset = split['train']
        eval_dataset = split['test']
        print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = hf_data
        eval_dataset = None
        print(f"  Train: {len(train_dataset)}, Val: None")

    # ── Load tokenizer ───────────────────────────────────────────────
    print("\nLoading tokenizer...")
    base_model_path = config['model']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=config['model']['trust_remote_code'],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model: {base_model_path}")
    model_kwargs = {
        "torch_dtype": getattr(torch, config['model']['torch_dtype']),
        "trust_remote_code": config['model']['trust_remote_code'],
    }

    if config['model']['attn_implementation']:
        model_kwargs["attn_implementation"] = config['model']['attn_implementation']

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        **model_kwargs,
    )

    # Disable KV cache (required for gradient checkpointing)
    model.config.use_cache = False

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

    trainable_params, total_params = _count_params(model, peft_config)
    print(f"  LoRA rank: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    print(f"  Target modules: {lora_cfg['target_modules']}")

    # ── Training arguments ───────────────────────────────────────────
    tc = config['training']
    output_dir = tc['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=tc.get('run_name') or Path(output_dir).name,
        num_train_epochs=tc['num_train_epochs'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        per_device_eval_batch_size=tc['per_device_eval_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        # gpt-oss models are MoE (32 experts, 4 active per token). MoE routing
        # is non-deterministic, breaking both gradient checkpointing modes:
        #   use_reentrant=False → CheckpointError (shape mismatch)
        #   use_reentrant=True  → silently wrong gradients (loss explosion)
        # Disable gradient checkpointing entirely for MoE models. They fit in
        # 2x H200 memory since only 4/32 experts are active per forward pass.
        gradient_checkpointing="gpt-oss" not in base_model_path and tc['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=tc['learning_rate'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        max_grad_norm=tc['max_grad_norm'],
        max_length=tc['max_seq_length'],
        # completion_only_loss is auto-enabled by TRL when the dataset has
        # "prompt" and "completion" keys (prompt-completion format).
        # This computes loss ONLY on completion tokens, not the prompt.
        # We use prompt-completion format instead of "messages" format with
        # assistant_only_loss because the latter requires {% generation %}
        # Jinja tags in the tokenizer's chat template, which most models lack.
        bf16=tc['bf16'],
        fp16=tc['fp16'],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=tc['logging_steps'],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=tc['eval_steps'] if eval_dataset else None,
        save_strategy="steps",
        save_steps=tc['save_steps'],
        save_total_limit=tc['save_total_limit'],
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        seed=tc['seed'],
        dataloader_num_workers=tc['dataloader_num_workers'],
        report_to=tc['report_to'],
        remove_unused_columns=False,
    )

    # ── Initialize trainer ───────────────────────────────────────────
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Print training stats
    n_train = len(train_dataset)
    effective_batch = tc['per_device_train_batch_size'] * tc['gradient_accumulation_steps'] * max(torch.cuda.device_count(), 1)
    steps_per_epoch = n_train // effective_batch
    total_steps = steps_per_epoch * tc['num_train_epochs']
    print(f"\n  Training examples:     {n_train}")
    print(f"  Effective batch size:  {effective_batch}")
    print(f"  Steps per epoch:       {steps_per_epoch}")
    print(f"  Total training steps:  {total_steps}")
    print(f"  Eval every:            {tc['eval_steps']} steps")
    print(f"  Save every:            {tc['save_steps']} steps")

    # ── Resolve checkpoint resumption ─────────────────────────────────
    resume = tc.get('resume_from_checkpoint')
    if resume == "auto":
        # Find the latest checkpoint-* directory in output_dir
        import glob
        ckpt_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda d: int(os.path.basename(d).split("-")[1]),
        )
        if ckpt_dirs:
            resume = ckpt_dirs[-1]
            print(f"\n  Auto-resume: found {len(ckpt_dirs)} checkpoint(s), resuming from {resume}")
        else:
            resume = None
            print("\n  Auto-resume: no checkpoints found, starting from scratch")

    # ── Train ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    trainer.train(resume_from_checkpoint=resume)

    # ── Save final model ─────────────────────────────────────────────
    print("\nSaving final model...")
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training log history
    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"  Training log saved to: {log_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Final model: {final_dir}")
    print(f"  Training log: {log_path}")
    print("=" * 70)


def _default_config():
    """Default SFT training configuration (matches DPO setup)."""
    return {
        'model': {
            'base_model': None,
            'torch_dtype': 'bfloat16',
            'attn_implementation': None,
            'trust_remote_code': False,
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
            'val_split': 0.1,
        },
        'training': {
            'num_train_epochs': 3,
            'learning_rate': 2e-5,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': 16,
            'gradient_checkpointing': True,
            'warmup_ratio': 0.1,
            'lr_scheduler_type': 'cosine',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'max_seq_length': 8192,
            'bf16': True,
            'fp16': False,
            'logging_steps': 10,
            'eval_steps': 50,
            'save_steps': 100,
            'save_total_limit': 3,
            'seed': 42,
            'dataloader_num_workers': 4,
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


def _count_params(model, peft_config):
    """Estimate trainable params with LoRA (before applying)."""
    total = sum(p.numel() for p in model.parameters())
    # Rough estimate: LoRA adds r * (in + out) per target module
    trainable_est = 0
    for name, module in model.named_modules():
        for target in peft_config.target_modules:
            if target in name and hasattr(module, 'weight'):
                in_f = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                out_f = module.weight.shape[0]
                trainable_est += peft_config.r * (in_f + out_f)
                break
    print(f"  Total params:      {total:,}")
    print(f"  LoRA params (est): {trainable_est:,} ({100*trainable_est/total:.2f}%)")
    return trainable_est, total


if __name__ == "__main__":
    main()
