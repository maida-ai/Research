#!/usr/bin/env python3
"""Example script demonstrating independent model usage.

This script shows how the models can be used independently of training scripts,
making it easy to experiment with different architectures and configurations.
"""

import torch

from efficient_longctx.models import (
    LongCtxModel,
    create_model,
    load_model_from_checkpoint,
)


def demonstrate_model_creation():
    """Demonstrate creating models with different configurations."""
    print("=== Model Creation Examples ===")

    # Create a small vanilla model
    vanilla_model = LongCtxModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=8,
        block_type="vanilla",
        block_kwargs={},
    )
    print(f"Vanilla model: {vanilla_model.get_num_params():,} parameters")

    # Create a DP-ASSM model
    dpassm_model = LongCtxModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=8,
        block_type="dpassm",
        block_kwargs={"window_size": 16, "ssm_state_dim": 32},
    )
    print(f"DP-ASSM model: {dpassm_model.get_num_params():,} parameters")

    # Create a BLADE model
    blade_model = LongCtxModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=8,
        block_type="blade",
        block_kwargs={"chunk_size": 8, "state_dim": 16},
    )
    print(f"BLADE model: {blade_model.get_num_params():,} parameters")


def demonstrate_model_factory():
    """Demonstrate using the model factory function."""
    print("\n=== Model Factory Examples ===")

    # Create different sized models using the factory
    for params in ["150m", "250m", "350m"]:
        model = create_model(
            vocab_size=1000,
            params=params,
            block_type="vanilla",
            block_kwargs={},
        )
        print(f"{params} model: {model.get_num_params():,} parameters")


def demonstrate_forward_pass():
    """Demonstrate forward pass through different models."""
    print("\n=== Forward Pass Examples ===")

    # Create a small model for testing
    model = LongCtxModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=8,
        block_type="vanilla",
        block_kwargs={},
    )

    # Test with different sequence lengths
    for seq_len in [1, 5, 10, 20]:
        input_ids = torch.randint(0, 1000, (2, seq_len))
        logits = model(input_ids)
        print(
            f"Sequence length {seq_len}: input {input_ids.shape} -> output {logits.shape}"
        )


def demonstrate_model_config():
    """Demonstrate model configuration and serialization."""
    print("\n=== Model Configuration Examples ===")

    model = create_model(
        vocab_size=1000,
        params="150m",
        block_type="dpassm",
        block_kwargs={"window_size": 32, "ssm_state_dim": 64},
    )

    config = model.get_model_config()
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def demonstrate_checkpoint_saving():
    """Demonstrate saving and loading model checkpoints."""
    print("\n=== Checkpoint Examples ===")

    # Create a model
    model = LongCtxModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=8,
        block_type="vanilla",
        block_kwargs={},
    )

    # Create a dummy checkpoint
    checkpoint = {
        "step": 100,
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "model_config": model.get_model_config(),
    }

    # Save checkpoint
    checkpoint_path = "/tmp/example_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Load checkpoint
    loaded_model = load_model_from_checkpoint(checkpoint_path)
    print(f"Loaded model with {loaded_model.get_num_params():,} parameters")

    # Verify forward pass works
    input_ids = torch.randint(0, 1000, (1, 5))
    logits = loaded_model(input_ids)
    print(f"Forward pass successful: {logits.shape}")


if __name__ == "__main__":
    print("Long-Context Model Usage Examples")
    print("=" * 40)

    demonstrate_model_creation()
    demonstrate_model_factory()
    demonstrate_forward_pass()
    demonstrate_model_config()
    demonstrate_checkpoint_saving()

    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("\nThe models are now independent of training scripts and can be:")
    print("- Used for inference without training dependencies")
    print("- Easily integrated into different applications")
    print("- Tested independently of the training pipeline")
    print("- Modified without affecting training code")
