#!/usr/bin/env python3
"""Verify OLD's checkpoint actually loads into the running model.

Mirrors the load sequence from train.py + setup_utils.py:
  1. setup_model_and_processor → loads Robometer-4B base + heads
  2. setup_peft_model → wraps with PEFT
  3. trainer._load_from_checkpoint(OLD_CKPT) → loads OLD's adapter+heads

After each step, prints the sum of:
  - progress_head.0.weight  (expected: 39.5 in OLD ckpt)
  - success_head.4.weight  (expected: -0.195 in OLD ckpt)
  - One LoRA adapter weight  (lora_B layers.0.q_proj — expected: nonzero from OLD)

If after step 3 the sums don't match OLD's checkpoint, the load is silently failing.
"""
import os, sys, json
import torch
from safetensors import safe_open

CKPT_DIR = '/projects/prjs1958/LoRA_weights/loss2_22244009/robometer_lora_loss2_c51_asymmetric/checkpoint-7500'

# 1. First, just read OLD's checkpoint values directly
print('=== TARGET VALUES (from OLD checkpoint safetensors) ===')
with open(f'{CKPT_DIR}/model.safetensors.index.json') as f:
    idx = json.load(f)

target_keys = [
    'progress_head.0.weight',
    'success_head.4.weight',
    'model.language_model.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight',
]
target_values = {}
for k in target_keys:
    if k not in idx['weight_map']:
        print(f'  {k}: NOT IN CHECKPOINT')
        continue
    shard = idx['weight_map'][k]
    with safe_open(f'{CKPT_DIR}/{shard}', framework='pt') as f:
        t = f.get_tensor(k)
        target_values[k] = (t.sum().item(), t.abs().mean().item(), tuple(t.shape))
        print(f'  {k}:')
        print(f'    shape={tuple(t.shape)}  sum={t.sum().item():.6f}  abs_mean={t.abs().mean().item():.6f}')

# 2. Now load the model the same way eval does, and check after each step
sys.path.insert(0, '/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer')
import hydra
from omegaconf import OmegaConf, DictConfig
from robometer.utils.setup_utils import setup_model_and_processor, setup_peft_model
from robometer.configs.experiment_configs import ExperimentConfig

# Build minimal config (mirroring eval_test_set.job MODEL=loss2 CKPT=checkpoint-7500)
config_dict = OmegaConf.create({
    'mode': 'evaluate',
    'model': {
        'base_model_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'use_peft': True,
        'use_unsloth': True,
        'train_progress_head': True,
        'train_preference_head': False,
        'train_success_head': True,
        'use_per_frame_progress_token': True,
        'use_multi_image': True,
        'progress_head_mode': 'c51',
        'progress_loss_type': 'discrete',
        'progress_discrete_bins': 10,
        'torch_dtype': 'bfloat16',
        'quantization': False,
        'frame_pooling': 'mean',
        'average_temporal_patches': True,
        'frame_pooling_attn_temperature': 1.0,
        'peft_vision_encoder': False,
        'train_language_model': True,
        'train_vision_encoder': False,
        'trust_remote_code': True,
        'model_type': 'default',
        'rewind': None,
    },
    'peft': {
        'r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'bias': 'none',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'peft_vision_encoder': False,
    },
    'training': {
        'output_dir': '/tmp/verify_ckpt',
        'load_from_checkpoint': 'robometer/Robometer-4B',
        'resume_from_checkpoint': CKPT_DIR,
        'per_device_eval_batch_size': 4,
        'bf16': True,
        'fp16': False,
        'num_gpus': 1,
        'gradient_checkpointing': True,
        'max_seq_length': 1024,
    },
})

print('\n=== STEP 1: setup_model_and_processor (loads Robometer-4B base) ===')
tokenizer, processor, rbm_model = setup_model_and_processor(config_dict, eval_only=True)

def dump_state(name):
    print(f'\n--- {name} ---')
    sd = rbm_model.state_dict()
    for k in target_keys:
        if k in sd:
            t = sd[k]
            print(f'  {k}: shape={tuple(t.shape)} sum={t.sum().item():.6f} abs_mean={t.abs().mean().item():.6f}')
        else:
            # Try alternate keys (PEFT might have wrapped the path)
            matches = [k2 for k2 in sd.keys() if k.split('.')[-2] in k2 and k.split('.')[-1] in k2 and 'lora_B' not in k2.replace('lora_B.default', '__')]
            print(f'  {k}: NOT FOUND in state_dict. matches: {matches[:3] if matches else "[]"}')

dump_state('AFTER setup_model_and_processor')

print('\n=== STEP 2: PEFT wrap ===')
rbm_model = setup_peft_model(rbm_model, config_dict)
dump_state('AFTER setup_peft_model')

print('\n=== STEP 3: Manually load OLD checkpoint (like trainer._load_from_checkpoint) ===')
# Mimic what HF Trainer does
from transformers import Trainer
trainer = Trainer.__new__(Trainer)
trainer.model = rbm_model
trainer._signature_columns = []
trainer.args = type('FakeArgs', (), {'output_dir': '/tmp', 'use_legacy_prediction_loop': False})()
# Use the same _load_from_checkpoint path
trainer._load_from_checkpoint(CKPT_DIR)
dump_state('AFTER trainer._load_from_checkpoint')
