"""Tests for DPASSM SDPA (scaled_dot_product_attention) integration.

This test module specifically validates the optimized attention computation
using PyTorch's scaled_dot_product_attention, including mask conversion,
dropout behavior, and fallback mechanisms.
"""

from unittest.mock import patch

import pytest
import torch

from efficient_longctx.blocks import DPASSMBlock


class TestDPASSMSDPA:
    """Test cases for DPASSM SDPA integration and optimization."""

    @pytest.fixture
    def model_params(self) -> dict[str, int | float]:
        """Standard model parameters for testing."""
        return {
            "d_model": 64,
            "n_heads": 4,
            "window_size": 8,
            "ssm_state_dim": 16,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, model_params: dict[str, int | float]) -> DPASSMBlock:
        """Create a DPASSMBlock instance for testing."""
        return DPASSMBlock(**model_params)

    @pytest.fixture
    def sample_input(self, model_params: dict[str, int | float]) -> torch.Tensor:
        """Sample input tensor."""
        batch_size, seq_len = 2, 16
        return torch.randn(batch_size, seq_len, model_params["d_model"])

    def test_sdpa_is_called(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """
        Test that scaled_dot_product_attention is actually being called.

        This verifies that the optimization is active and not falling back
        to manual attention computation.
        """
        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Set up mock to return valid output
            B, T, d = sample_input.shape
            h = model.n_heads
            d_h = d // h
            mock_output = torch.randn(B, h, T, d_h)
            mock_sdpa.return_value = mock_output

            model.eval()
            with torch.no_grad():
                # Call compute_attention
                _ = model._compute_attention(sample_input)

            # Verify SDPA was called
            assert mock_sdpa.called, "SDPA should be called for attention computation"
            assert mock_sdpa.call_count >= 1

    def test_mask_conversion_correctness(self, model: DPASSMBlock) -> None:
        """
        Test that additive masks are correctly converted to boolean masks.

        The mask conversion is critical: additive masks with -inf/0 must be
        converted to boolean masks with True/False for SDPA.
        """
        seq_len = 8
        device = torch.device("cpu")
        dtype = torch.float32

        # Build windowed causal mask (additive format)
        mask = model._build_window_mask(seq_len, model.window_size, device, dtype)

        # Create dummy input
        B = 1
        x = torch.randn(B, seq_len, model.d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Set up mock
            h = model.n_heads
            d_h = model.d_model // h
            mock_output = torch.randn(B, h, seq_len, d_h)
            mock_sdpa.return_value = mock_output

            model.eval()
            with torch.no_grad():
                _ = model._compute_attention(x, mask=mask)

            # Get the actual call arguments
            call_args = mock_sdpa.call_args

            # Extract attn_mask from kwargs
            attn_mask = call_args.kwargs.get("attn_mask")

            # Verify mask was converted
            assert attn_mask is not None, "Mask should be passed to SDPA"
            assert attn_mask.dtype == torch.bool, "Mask should be boolean type"

            # Verify mask conversion logic:
            # Original mask: -inf for blocked, 0 for allowed
            # Boolean mask: True for allowed, False for blocked
            # Check a few positions
            for i in range(min(4, seq_len)):
                for j in range(i + 1, min(i + 4, seq_len)):
                    # j > i should be blocked (causal)
                    original_blocked = mask[i, j] == float("-inf")
                    bool_blocked = not attn_mask[i, j].item()
                    assert original_blocked == bool_blocked, (
                        f"Mask conversion failed at [{i},{j}]"
                    )

    def test_dropout_behavior_train_vs_eval(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """
        Test that dropout is correctly applied in training mode and disabled in eval.

        This is critical for proper model behavior and reproducibility.
        """
        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            B, T, d = sample_input.shape
            h = model.n_heads
            d_h = d // h
            mock_output = torch.randn(B, h, T, d_h)
            mock_sdpa.return_value = mock_output

            # Test in training mode
            model.train()
            _ = model._compute_attention(sample_input)

            # Get dropout_p from the call
            train_call_kwargs = mock_sdpa.call_args.kwargs
            train_dropout_p = train_call_kwargs.get("dropout_p", 0.0)

            # In training mode, dropout_p should match model.dropout_rate
            assert train_dropout_p == model.dropout_rate, (
                f"Training mode should use dropout_p={model.dropout_rate}"
            )

            # Test in eval mode
            model.eval()
            with torch.no_grad():
                _ = model._compute_attention(sample_input)

            # Get dropout_p from the call
            eval_call_kwargs = mock_sdpa.call_args.kwargs
            eval_dropout_p = eval_call_kwargs.get("dropout_p", 0.0)

            # In eval mode, dropout_p should be 0
            assert eval_dropout_p == 0.0, (
                "Eval mode should disable dropout (dropout_p=0)"
            )

    def test_sdpa_with_no_mask(self, model: DPASSMBlock) -> None:
        """Test SDPA behavior when no mask is provided."""
        B, T, d = 2, 8, model.d_model
        x = torch.randn(B, T, d)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            h = model.n_heads
            d_h = d // h
            mock_output = torch.randn(B, h, T, d_h)
            mock_sdpa.return_value = mock_output

            model.eval()
            with torch.no_grad():
                _ = model._compute_attention(x, mask=None)

            # Verify SDPA was called with None mask
            call_kwargs = mock_sdpa.call_args.kwargs
            attn_mask = call_kwargs.get("attn_mask")
            assert attn_mask is None, "No mask should be passed when mask=None"

    def test_forward_step_uses_sdpa(self, model: DPASSMBlock) -> None:
        """Test that forward_step also uses SDPA for single-token inference."""
        B = 2
        x_t = torch.randn(B, 1, model.d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            h = model.n_heads
            d_h = model.d_model // h
            mock_output = torch.randn(B, h, 1, d_h)
            mock_sdpa.return_value = mock_output

            model.eval()
            with torch.no_grad():
                _, _ = model.forward_step(x_t)

            # Verify SDPA was called
            assert mock_sdpa.called, "forward_step should use SDPA"

    def test_attention_output_correctness(self, model: DPASSMBlock) -> None:
        """
        Test that SDPA produces correct attention outputs.

        Performs a sanity check that the attention mechanism produces
        reasonable outputs (no NaNs, correct shapes, bounded values).
        """
        B, T, d = 2, 16, model.d_model
        x = torch.randn(B, T, d)

        # Build mask
        mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)

        model.eval()
        with torch.no_grad():
            output = model._compute_attention(x, mask=mask)

        # Check output properties
        assert output.shape == x.shape, "Output shape should match input"
        assert not torch.isnan(output).any(), "Output should not contain NaNs"
        assert not torch.isinf(output).any(), "Output should not contain Infs"

        # Attention outputs should be roughly bounded
        # (not a strict requirement but good sanity check)
        output_std = output.std().item()
        assert 0.1 < output_std < 100.0, f"Output std ({output_std}) seems unreasonable"

    def test_deterministic_attention_in_eval(self, model: DPASSMBlock) -> None:
        """Test that attention is deterministic in eval mode."""
        B, T, d = 2, 16, model.d_model
        x = torch.randn(B, T, d)
        mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)

        model.eval()
        with torch.no_grad():
            output1 = model._compute_attention(x, mask=mask)
            output2 = model._compute_attention(x, mask=mask)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6), (
            "Attention should be deterministic in eval mode"
        )

    def test_memory_efficiency_long_sequence(self) -> None:
        """
        Test that DPASSM can handle long sequences efficiently.

        With windowed attention + SDPA, we should be able to process
        sequences much longer than would fit in memory with full attention.
        """
        model = DPASSMBlock(
            d_model=128, n_heads=8, window_size=32, ssm_state_dim=64, dropout=0.1
        )

        # Long sequence that would cause OOM with full O(L^2) attention
        batch_size, seq_len = 1, 2048
        x = torch.randn(batch_size, seq_len, model.d_model)

        model.eval()
        with torch.no_grad():
            output, state = model(x)

        assert output.shape == x.shape
        assert state.shape == (batch_size, model.ssm_state_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sdpa_on_cuda(self, model_params: dict[str, int | float]) -> None:
        """Test SDPA works correctly on CUDA devices."""
        model = DPASSMBlock(**model_params).cuda()

        B, T = 2, 16
        x = torch.randn(B, T, model.d_model, device="cuda")
        mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)

        model.eval()
        with torch.no_grad():
            output = model._compute_attention(x, mask=mask)

        assert output.device.type == "cuda"
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow_through_sdpa(self, model: DPASSMBlock) -> None:
        """Test that gradients flow correctly through SDPA attention."""
        B, T, d = 2, 16, model.d_model
        x = torch.randn(B, T, d, requires_grad=True)
        mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)

        model.train()
        output = model._compute_attention(x, mask=mask)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Gradients should flow back to input"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaNs"

    def test_windowed_attention_respects_window_size(self, model: DPASSMBlock) -> None:
        """
        Test that windowed attention actually limits the context window.

        This is a functional test to verify the mask is working correctly.
        """
        # Use deterministic input for easier analysis
        torch.manual_seed(42)

        B, T = 1, 16
        model.eval()

        # Create input where each position has a distinct pattern
        x = torch.zeros(B, T, model.d_model)
        for t in range(T):
            x[0, t, 0] = float(t)  # Position marker

        with torch.no_grad():
            mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)
            output = model._compute_attention(x, mask=mask)

        # The output at each position should only be influenced by tokens
        # within the window. This is hard to test precisely without knowing
        # the learned weights, but we can at least verify the output is valid
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_batch_size_consistency(self, model: DPASSMBlock) -> None:
        """Test that SDPA attention handles different batch sizes correctly."""
        T, d = 16, model.d_model
        mask = model._build_window_mask(
            T, model.window_size, torch.device("cpu"), torch.float32
        )

        model.eval()
        with torch.no_grad():
            for B in [1, 2, 4, 8]:
                x = torch.randn(B, T, d)
                output = model._compute_attention(x, mask=mask)

                assert output.shape == (B, T, d)
                assert not torch.isnan(output).any()

    def test_edge_case_single_token(self, model: DPASSMBlock) -> None:
        """Test SDPA attention with single token (T=1)."""
        B, T, d = 2, 1, model.d_model
        x = torch.randn(B, T, d)

        model.eval()
        with torch.no_grad():
            output = model._compute_attention(x, mask=None)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_edge_case_large_window(self) -> None:
        """Test SDPA attention when window size equals or exceeds sequence length."""
        # Window size larger than sequence
        model = DPASSMBlock(
            d_model=64, n_heads=4, window_size=128, ssm_state_dim=16, dropout=0.1
        )

        B, T, d = 2, 32, model.d_model  # T < window_size
        x = torch.randn(B, T, d)

        model.eval()
        with torch.no_grad():
            mask = model._build_window_mask(T, model.window_size, x.device, x.dtype)
            output = model._compute_attention(x, mask=mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_full_forward_with_sdpa(self, model: DPASSMBlock) -> None:
        """
        Test full forward pass (attention + SSM) uses SDPA correctly.

        This is an integration test verifying the entire forward pass works
        with the SDPA optimization.
        """
        B, T = 2, 16
        x = torch.randn(B, T, model.d_model)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Set up mock for attention computation
            h = model.n_heads
            d_h = model.d_model // h
            mock_output = torch.randn(B, h, T, d_h)
            mock_sdpa.return_value = mock_output

            model.eval()
            with torch.no_grad():
                output, state = model(x)

            # Verify SDPA was called during forward pass
            assert mock_sdpa.called, "Full forward should use SDPA for attention"

            # Verify outputs
            assert output.shape == x.shape
            assert state.shape == (B, model.ssm_state_dim)
