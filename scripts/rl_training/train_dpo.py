#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training script for Faithfulness-DPO baseline.

Trains a base model on preference pairs where:
  chosen  = evidence-grounded reasoning (high NLI grounding score)
  rejected = evidence-ignoring reasoning (low NLI grounding score)

This is a PURE faithfulness signal — no correctness requirement on chosen.
Used as the Faithfulness-DPO baseline in the EvidenceRL ablation study.

Dataset format (from build_faithfulness_dpo_dataset.py):
  {"pairs": [{"prompt": [...], "chosen": [...], "rejected": [...]}, ...]}

Usage:
    python train_dpo.py --base-model /path/to/model \\
                        --dataset training_data/dpo/faithfulness_dpo.json \\
                        --output-dir ./dpo_trained_models/gemma-3-12b_faithfulness_dpo
"""

import argparse
import json
import os
import sys
import yaml
import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Faithfulness-DPO training for EvidenceRL baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base-model", type=str, help="Path to base model")
    parser.add_argument("--dataset", type=str, help="Path to Faithfulness-DPO dataset JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--run-name", type=str, default=None)

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None,
                        help="DPO beta (KL penalty, default: 0.1)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max total sequence length (prompt + chosen/rejected)")
    parser.add_argument("--max-prompt-length", type=int, default=None,
                        help="Max prompt length (truncated if longer)")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config = _default_config()
    if args.config:
        with open(args.config) as f:
            config = _merge_config(config, yaml.safe_load(f))

    if args.base_model:       config['model']['base_model'] = args.base_model
    if args.dataset:          config['data']['dataset_path'] = args.dataset
    if args.output_dir:       config['training']['output_dir'] = args.output_dir
    if args.run_name:         config['training']['run_name'] = args.run_name
    if args.epochs is not None:     config['training']['num_train_epochs'] = args.epochs
    if args.lr is not None:         config['training']['learning_rate'] = args.lr
    if args.batch_size is not None: config['training']['per_device_train_batch_size'] = args.batch_size
    if args.grad_accum is not None: config['training']['gradient_accumulation_steps'] = args.grad_accum
    if args.beta is not None:       config['dpo']['beta'] = args.beta
    if args.max_length is not None:        config['dpo']['max_length'] = args.max_length
    if args.max_prompt_length is not None: config['dpo']['max_prompt_length'] = args.max_prompt_length
    if args.lora_r is not None:     config['lora']['r'] = args.lora_r
    if args.lora_alpha is not None: config['lora']['lora_alpha'] = args.lora_alpha
    if args.load_in_4bit:           config['model']['load_in_4bit'] = True
    if args.resume_from_checkpoint: config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint

    if not config['model']['base_model']:
        parser.error("--base-model is required")
    if not config['data']['dataset_path']:
        parser.error("--dataset is required")
    if not config['training']['output_dir']:
        parser.error("--output-dir is required")

    print("=" * 70)
    print("FAITHFULNESS-DPO TRAINING CONFIGURATION")
    print("=" * 70)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 70)

    # ── Heavy imports ─────────────────────────────────────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    # ── Load dataset ──────────────────────────────────────────────────
    print("\nLoading Faithfulness-DPO dataset...")
    with open(config['data']['dataset_path']) as f:
        raw_data = json.load(f)

    pairs = raw_data['pairs']
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Dataset stats: {raw_data.get('statistics', {})}")

    # DPOTrainer expects columns: prompt, chosen, rejected
    # Our dataset stores these as chat lists (TRL conversational format)
    hf_data = Dataset.from_list([
        {
            "prompt":   p["prompt"],
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    # Train/val split
    val_split = config['data']['val_split']
    if val_split > 0:
        split = hf_data.train_test_split(test_size=val_split, seed=config['training']['seed'])
        train_dataset = split['train']
        eval_dataset  = split['test']
        print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = hf_data
        eval_dataset  = None
        print(f"  Train: {len(train_dataset)}, Val: None")

    # ── Load tokenizer / processor ───────────────────────────────────
    print("\nLoading tokenizer...")
    base_model_path = config['model']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=config['model']['trust_remote_code'],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO needs left padding

    # We always use text-only DPO (no images), so just use the tokenizer
    processing_class = tokenizer

    # ── Load model ────────────────────────────────────────────────────
    print(f"\nLoading model: {base_model_path}")
    model_kwargs = {
        "torch_dtype": getattr(torch, config['model']['torch_dtype']),
        "trust_remote_code": config['model']['trust_remote_code'],
        "attn_implementation": config['model']['attn_implementation'] or "eager",
    }

    if config['model'].get('load_in_4bit', False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model_kwargs["device_map"] = {"": local_rank}
        print(f"  4-bit quantization (QLoRA) on device {local_rank}")
    # Note: do NOT set device_map="auto" for non-4bit — accelerate handles placement

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    model.config.use_cache = False   # required for gradient checkpointing

    # ── LoRA config ───────────────────────────────────────────────────
    lora_cfg = config['lora']
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        bias=lora_cfg['bias'],
        task_type="CAUSAL_LM",
    )
    print(f"\n  LoRA rank: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")

    # ── DPO training config ───────────────────────────────────────────
    tc = config['training']
    dc = config['dpo']
    output_dir = tc['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        run_name=tc.get('run_name') or Path(output_dir).name,
        num_train_epochs=tc['num_train_epochs'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        per_device_eval_batch_size=tc['per_device_eval_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        gradient_checkpointing=tc['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=tc['learning_rate'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        max_grad_norm=tc['max_grad_norm'],
        bf16=tc['bf16'],
        fp16=tc['fp16'],
        # DPO-specific
        beta=dc['beta'],
        loss_type=dc['loss_type'],
        max_length=dc['max_length'],
        max_prompt_length=dc['max_prompt_length'],
        # Logging & checkpointing
        logging_steps=tc['logging_steps'],
        logging_dir=os.path.join(output_dir, "logs"),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=tc['eval_steps'] if eval_dataset else None,
        save_strategy="steps",
        save_steps=tc['save_steps'],
        save_total_limit=tc['save_total_limit'],
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        seed=tc['seed'],
        report_to=tc['report_to'],
        remove_unused_columns=False,
    )

    print(f"\n  DPO beta:            {dc['beta']}")
    print(f"  DPO loss type:       {dc['loss_type']}")
    print(f"  Max length:          {dc['max_length']}")
    print(f"  Max prompt length:   {dc['max_prompt_length']}")

    # ── Initialize DPOTrainer ─────────────────────────────────────────
    # Force text-only mode: TRL classifies some models (e.g. Gemma-3) as VLMs
    # and expects image data in the dataset. We override this since our DPO
    # dataset is text-only.
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    if model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        original_model_type = model.config.model_type
        model.config.model_type = f"_{original_model_type}_text_only"
        print(f"  Overriding model_type to force text-only DPO (was: {original_model_type})")

    print("\nInitializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
    )

    n_train = len(train_dataset)
    effective_batch = (tc['per_device_train_batch_size']
                       * tc['gradient_accumulation_steps']
                       * max(torch.cuda.device_count(), 1))
    steps_per_epoch = n_train // effective_batch
    total_steps = steps_per_epoch * tc['num_train_epochs']
    print(f"\n  Training pairs:       {n_train}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Steps per epoch:      {steps_per_epoch}")
    print(f"  Total steps:          {total_steps}")

    # ── Checkpoint resume ─────────────────────────────────────────────
    resume = tc.get('resume_from_checkpoint')
    if resume == "auto":
        ckpt_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda d: int(os.path.basename(d).split("-")[1]),
        )
        resume = ckpt_dirs[-1] if ckpt_dirs else None
        print(f"\n  Auto-resume: {'resuming from ' + resume if resume else 'starting from scratch'}")

    # ── Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING FAITHFULNESS-DPO TRAINING")
    print("=" * 70 + "\n")

    trainer.train(resume_from_checkpoint=resume)

    # ── Save ──────────────────────────────────────────────────────────
    print("\nSaving final model...")
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(output_dir, "training_log.json"), 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("\n" + "=" * 70)
    print("FAITHFULNESS-DPO TRAINING COMPLETE")
    print(f"  Final model: {final_dir}")
    print("=" * 70)


def _default_config():
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
        'dpo': {
            'beta': 0.1,             # KL penalty (standard DPO default)
            'loss_type': 'sigmoid',  # standard DPO objective
            'max_length': 8192,      # prompt + chosen/rejected combined
            'max_prompt_length': 6144,
        },
        'data': {
            'dataset_path': None,
            'val_split': 0.05,       # smaller val split: DPO has fewer pairs than SFT
        },
        'training': {
            'num_train_epochs': 2,
            'learning_rate': 5e-6,   # lower than SFT (DPO is more sensitive to LR)
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'gradient_checkpointing': True,
            'warmup_ratio': 0.1,
            'lr_scheduler_type': 'cosine',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'bf16': True,
            'fp16': False,
            'logging_steps': 10,
            'eval_steps': 50,
            'save_steps': 100,
            'save_total_limit': 3,
            'seed': 42,
            'report_to': 'none',
            'output_dir': None,
            'run_name': None,
            'resume_from_checkpoint': None,
        },
    }


def _merge_config(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_config(merged[k], v)
        else:
            merged[k] = v
    return merged


if __name__ == "__main__":
    main()
