#!/bin/bash

# ============================================================================
# Train and Inference Examples for MoE Model
# ============================================================================

# --- Option 1: Synthetic Data (Full Training + Inference) ---
# conda run -n moe-gpu python moe_train_infer.py \
# 	--dataset synthetic \
# 	--epochs 10 \
# 	--batch-size 256 \
# 	--num-experts 4 \
# 	--top-k 2 \
# 	--num-test-samples 16 \
# 	--checkpoint-path checkpoints/moe_synthetic.pt

# --- Option 2: MNIST - Training with Inference ---
echo "=== Training MNIST MoE Model with Inference ==="
conda run -n moe-gpu python moe_train_infer.py \
	--dataset mnist \
	--epochs 5 \
	--batch-size 512 \
	--num-experts 10 \
	--top-k 2 \
	--max-train-samples 8192 \
	--num-test-samples 16 \
	--checkpoint-path checkpoints/moe_mnist.pt

echo ""

# --- Option 3: MNIST - Inference Only (from saved checkpoint) ---
echo "=== Inference with Different Sample Count ==="
conda run -n moe-gpu python moe_train_infer.py \
	--dataset mnist \
	--inference-only \
	--num-test-samples 32 \
	--checkpoint-path checkpoints/moe_mnist.pt
