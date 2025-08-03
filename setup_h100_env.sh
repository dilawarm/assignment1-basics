#!/bin/bash
# Setup script for H100 FP8 training environment

echo "Setting up H100 environment for FP8 training..."

# Critical environment variables for H100 + Transformer Engine
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_BIAS_GELU_NVFUSION=0
export NVTE_MASKED_SOFTMAX_FUSION=0

# Additional cuBLAS-specific settings for H100
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_DETERMINISTIC=1
export CUDA_MODULE_LOADING=LAZY
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

# Transformer Engine specific settings
export NVTE_FP8_DPA=0  # Disable FP8 dot product attention
export NVTE_FP8_MHA=0  # Disable FP8 multi-head attention
export NVTE_FP8_BACKEND=0  # Use legacy backend

echo "Environment variables set."
echo "Run your training script with: source setup_h100_env.sh && python train_h100.py"