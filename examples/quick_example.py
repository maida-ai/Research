#!/usr/bin/env python3
"""Quick example showing how to use models from the efficient_longctx.models subpackage."""

import torch

from efficient_longctx.models import LongCtxModel, create_model

# Example 1: Create a small model directly
print("=== Direct Model Creation ===")
model = LongCtxModel(
    vocab_size=1000,
    d_model=64,
    n_layers=2,
    n_heads=8,
    block_type="vanilla",
    block_kwargs={},
)
print(f"Small model: {model.get_num_params():,} parameters")

# Example 2: Use the factory function
print("\n=== Factory Function ===")
large_model = create_model(
    vocab_size=1000,
    num_params="150m",
    block_type="dpassm",
    block_kwargs={"window_size": 32, "ssm_state_dim": 64},
)
print(f"Large DP-ASSM model: {large_model.get_num_params():,} parameters")

# Example 3: Forward pass
print("\n=== Forward Pass ===")
input_ids = torch.randint(0, 1000, (1, 10))
logits = model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")

print("\n✅ Models are working correctly from the models/ directory!")
