#!/usr/bin/env python3
"""
Demo script for DPASSM (Dual-Path Attention + State Space Model) block.

This script demonstrates how to:
1. Initialize a DPASSM block
2. Run forward passes with state persistence
3. Show the benefits of the dual-path architecture
"""

import torch

from efficient_longctx.blocks import DPASSMBlock


def demo_dpassm_block():
    """Demonstrate DPASSM block functionality."""
    print("🚀 DPASSM (Dual-Path Attention + State Space Model) Demo")
    print("=" * 60)

    # Model parameters
    d_model = 128
    n_heads = 8
    window_size = 16
    ssm_state_dim = 32
    dropout = 0.1

    print("📊 Model Configuration:")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of heads: {n_heads}")
    print(f"  - Attention window: {window_size}")
    print(f"  - SSM state dimension: {ssm_state_dim}")
    print(f"  - Dropout rate: {dropout}")
    print()

    # Create DPASSM block
    dpassm_block = DPASSMBlock(
        d_model=d_model,
        n_heads=n_heads,
        window_size=window_size,
        ssm_state_dim=ssm_state_dim,
        dropout=dropout,
    )

    # Print model architecture
    print("🏗️  Model Architecture:")
    total_params = sum(p.numel() for p in dpassm_block.parameters())
    trainable_params = sum(
        p.numel() for p in dpassm_block.parameters() if p.requires_grad
    )
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print()

    # Example usage
    batch_size = 2
    seq_len = 20

    print("📝 Example Usage:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print()

    # Generate sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"📥 Input shape: {x.shape}")

    # Run forward pass
    print("🔄 Running forward pass...")
    dpassm_block.eval()
    with torch.no_grad():
        output, new_state = dpassm_block(x)

    print(f"📤 Output shape: {output.shape}")
    print(f"🧠 SSM state shape: {new_state.shape}")
    print()

    # Demonstrate state persistence
    print("🔗 State Persistence Demo:")
    initial_state = torch.randn(batch_size, ssm_state_dim)

    with torch.no_grad():
        _, state1 = dpassm_block(x, initial_state)
        _, state2 = dpassm_block(x, state1)

    state_diff = torch.norm(state1 - state2).item()
    print(f"  - State evolution magnitude: {state_diff:.6f}")
    print()

    # Show computational complexity benefits
    print("⚡ Computational Complexity:")
    print("  - Traditional attention: O(L²)")
    print(f"  - DPASSM attention: O(L×W) where W={window_size}")

    seq_len_examples = [100, 1000, 10000]
    for L in seq_len_examples:
        traditional_cost = L * L
        dpassm_cost = L * window_size
        speedup = traditional_cost / dpassm_cost
        print(f"  - L={L}: {speedup:.1f}x speedup")
    print()

    # Demonstrate different sequence lengths
    print("📏 Variable Sequence Length Support:")
    test_lengths = [1, 4, 8, 16, 32, 64]

    for length in test_lengths:
        test_input = torch.randn(batch_size, length, d_model)
        with torch.no_grad():
            test_output, test_state = dpassm_block(test_input)
        print(f"  - Length {length:2d}: Output {list(test_output.shape)} ✅")


def demo_windowed_attention():
    """Demonstrate the windowed attention mechanism."""
    print("\n🪟 Windowed Attention Demo:")
    print("-" * 40)

    # Create a small model for visualization
    dpassm = DPASSMBlock(d_model=64, n_heads=4, window_size=8, ssm_state_dim=16)

    seq_len = 16
    mask = dpassm._build_window_mask(seq_len, 8, torch.device("cpu"), torch.float32)

    print(f"Sequence length: {seq_len}")
    print("Window size: 8")
    print("Attention mask (0 = allowed, -inf = blocked):")

    # Convert mask to readable format
    readable_mask = torch.zeros_like(mask)
    readable_mask[mask == float("-inf")] = 1.0

    # Print first few rows for visualization
    for i in range(min(8, seq_len)):
        row_str = " ".join("●" if x == 1 else "○" for x in readable_mask[i, :8].int())
        print(f"  Position {i:2d}: {row_str}")

    print("  Legend: ○ = attended, ● = masked")


def demo_ssm_dynamics():
    """Demonstrate SSM state evolution."""
    print("\n🌀 SSM Dynamics Demo:")
    print("-" * 40)

    dpassm = DPASSMBlock(d_model=64, n_heads=4, window_size=8, ssm_state_dim=16)

    batch_size = 1
    seq_len = 10
    d_model = 64

    # Create input sequence
    torch.manual_seed(42)  # For reproducible results
    x = torch.randn(batch_size, seq_len, d_model)
    initial_state = torch.zeros(batch_size, 16)

    print("SSM State Evolution:")
    current_state = initial_state.clone()

    dpassm.eval()
    with torch.no_grad():
        # Process the sequence step by step
        for i in range(seq_len):
            # Take only the first i+1 tokens
            input_step = x[:, : i + 1, :]
            _, current_state = dpassm._compute_ssm(input_step, current_state)

            # Print state norm (shows evolution)
            state_norm = torch.norm(current_state).item()
            print(f"  Step {i + 1:2d}: state norm = {state_norm:.4f}")


if __name__ == "__main__":
    try:
        demo_dpassm_block()
        demo_windowed_attention()
        demo_ssm_dynamics()

        print("\n✨ Demo completed successfully!")
        print("\n📚 Key Benefits:")
        print("  ✅ Linear complexity for attention window")
        print("  ✅ Recurrent SSM state for long-term memory")
        print("  ✅ Learnable fusion of attention and SSM")
        print("  ✅ Variable sequence length support")
        print("  ✅ Stable gradient flow")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise
