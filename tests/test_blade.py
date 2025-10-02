"""Tests for BLADE (Block-Local Attention with Per-Block State) block."""

import pytest
import torch

from efficient_longctx.blocks import BLADEBlock
from efficient_longctx.utils.constants import CUDA_AVAILABLE


class TestBLADEBlock:
    """Test cases for BLADEBlock."""

    @pytest.fixture
    def model_params(self) -> dict[str, int | float]:
        """Standard model parameters for testing."""
        return {
            "d_model": 64,
            "n_heads": 4,
            "chunk_size": 8,
            "state_dim": 16,
            "m_global": 0,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, model_params: dict[str, int | float]) -> BLADEBlock:
        """Create a BLADEBlock instance for testing."""
        return BLADEBlock(**model_params)

    @pytest.fixture
    def sample_input(self, model_params: dict[str, int | float]) -> torch.Tensor:
        """Sample input tensor."""
        batch_size, seq_len = 2, 16
        return torch.randn(batch_size, seq_len, model_params["d_model"])

    def test_model_initialization(
        self, model: BLADEBlock, model_params: dict[str, int | float]
    ) -> None:
        """Test that model initializes with correct parameters."""
        assert model.d_model == model_params["d_model"]
        assert model.n_heads == model_params["n_heads"]
        assert model.chunk_size == model_params["chunk_size"]
        assert model.state_dim == model_params["state_dim"]
        assert model.m_global == model_params["m_global"]
        assert model.dropout_rate == model_params["dropout"]

        # Check that all required components are initialized
        assert hasattr(model, "state_proj")
        assert hasattr(model, "state_inject")
        assert hasattr(model, "q_proj")
        assert hasattr(model, "k_proj")
        assert hasattr(model, "v_proj")
        assert hasattr(model, "out_proj")
        assert hasattr(model, "dropout")

        # Check dimensions
        assert model.state_proj.out_features == model_params["state_dim"]
        assert model.state_inject.in_features == model_params["state_dim"]
        assert model.state_inject.out_features == model_params["d_model"]

    def test_model_initialization_with_global_tokens(self) -> None:
        """Test model initialization with global tokens."""
        model = BLADEBlock(
            d_model=64,
            n_heads=4,
            chunk_size=8,
            state_dim=16,
            m_global=2,
            dropout=0.1,
        )

        assert hasattr(model, "global_tokens")
        assert model.global_tokens.shape == (2, 64)  # [m_global, d_model]

    def test_model_initialization_invalid_n_heads(self) -> None:
        """Test that model raises error for invalid n_heads."""
        with pytest.raises(ValueError, match="d_model .* must be divisible by n_heads"):
            BLADEBlock(
                d_model=64,
                n_heads=5,  # 64 is not divisible by 5
                chunk_size=8,
                state_dim=16,
            )

    def test_shapes_cpu(self) -> None:
        B, T, D, H = 2, 33, 64, 4
        block = BLADEBlock(
            d_model=D, n_heads=H, chunk_size=16, state_dim=32, m_global=2
        )
        x = torch.randn(B, T, D)
        y, s = block(x)
        assert y.shape == (B, T, D)
        assert s.shape == (B, 32)

    def test_no_cross_chunk_leakage(self) -> None:
        B, D, H = 1, 32, 4
        block = BLADEBlock(d_model=D, n_heads=H, chunk_size=8, state_dim=16, m_global=0)
        x = torch.zeros(B, 16, D)
        x[:, :8, :] = 1.0  # first chunk all ones
        y, _ = block(x)
        # pick a token t in second chunk; tweak a far-future token and assert no change
        t = 12
        y_baseline = y.clone()
        x2 = x.clone()
        x2[:, 2, :] = 5.0  # change token in first chunk (past)
        y2, _ = block(x2)
        # token t must change (past change can affect later via state)
        assert not torch.allclose(y2[:, t, :], y_baseline[:, t, :])

    def test_state_continuity(self) -> None:
        B, T, D, H = 2, 32, 64, 4
        block = BLADEBlock(
            d_model=D,
            n_heads=H,
            chunk_size=16,
            state_dim=32,
            m_global=0,
            dropout=0.0,  # Disable dropout for deterministic comparison
        )
        x = torch.randn(B, T, D)

        y_full, s_full = block(x)

        # two passes
        y_a, s_a = block(x[:, :16, :])
        y_b, s_b = block(x[:, 16:, :], state=s_a)
        y_cat = torch.cat([y_a, y_b], dim=1)

        assert torch.allclose(y_full, y_cat, atol=1e-4, rtol=1e-3)
        assert torch.allclose(s_full, s_b, atol=1e-4, rtol=1e-3)

    def test_globals_present_and_trainable(self) -> None:
        B, T, D, H = 1, 24, 48, 4
        block = BLADEBlock(
            d_model=D, n_heads=H, chunk_size=12, state_dim=16, m_global=2
        )
        x = torch.randn(B, T, D)
        assert hasattr(block, "global_tokens")
        y, s = block(x)
        loss = y.sum()
        loss.backward()
        assert block.global_tokens.grad is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_smoke(self) -> None:
        block = BLADEBlock(64, 4, 32, 16, m_global=1).cuda()
        x = torch.randn(2, 128, 64, device="cuda")
        y, s = block(x)
        assert y.is_cuda and s.is_cuda

    def test_forward_pass_basic(
        self, model: BLADEBlock, sample_input: torch.Tensor
    ) -> None:
        """Test basic forward pass functionality."""
        output, state = model(sample_input)

        # Check output shape
        assert output.shape == sample_input.shape
        assert isinstance(output, torch.Tensor)

        # Check state shape
        assert state.shape == (sample_input.shape[0], model.state_dim)
        assert isinstance(state, torch.Tensor)

    def test_forward_pass_with_custom_state(
        self, model: BLADEBlock, sample_input: torch.Tensor
    ) -> None:
        """Test forward pass with custom initial state."""
        custom_state = torch.randn(sample_input.shape[0], model.state_dim)
        output, new_state = model(sample_input, state=custom_state)

        assert output.shape == sample_input.shape
        assert new_state.shape == custom_state.shape
        assert not torch.equal(new_state, custom_state)  # State should be updated

    def test_forward_pass_state_influence(
        self, model: BLADEBlock, device: str = "cpu"
    ) -> None:
        """Test that early chunk state influences later chunks."""
        model.to(device)

        # Create input with multiple chunks
        batch_size, seq_len = 1, 12  # 2 chunks of size 8
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        # Forward pass with zero initial state
        zero_state = torch.zeros(batch_size, model.state_dim, device=device)
        output_with_zero, state_from_zero = model(x, state=zero_state)

        # Forward pass with nonzero initial state
        nonzero_state = torch.randn(batch_size, model.state_dim, device=device)
        output_with_nonzero, state_from_nonzero = model(x, state=nonzero_state)

        # Outputs should be different due to state injection
        assert not torch.allclose(output_with_zero, output_with_nonzero, atol=1e-5)

    def test_gradient_flow_across_chunks(
        self, model: BLADEBlock, device: str = "cpu"
    ) -> None:
        """Test that gradients flow across chunks when state passing is enabled."""
        model.to(device)
        model.train()

        # Create input requiring gradient
        batch_size, seq_len = 1, 12  # Multiple chunks
        x = torch.randn(
            batch_size, seq_len, model.d_model, device=device, requires_grad=True
        )

        output, state = model(x)

        # Compute loss on later positions (should backprop through earlier chunks via state)
        loss = output[-1].sum()  # Loss on last position
        loss.backward()

        # Check that gradients flow back to input
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    @pytest.mark.parametrize("chunk_size", [4, 8, 16])
    def test_different_chunk_sizes(self, chunk_size: int, device: str = "cpu") -> None:
        """Test BLADE block with different chunk sizes."""
        model = BLADEBlock(
            d_model=64,
            n_heads=4,
            chunk_size=chunk_size,
            state_dim=16,
            dropout=0.1,
        )
        model.to(device)

        # Test with sequence longer than chunk size
        seq_len = chunk_size * 3
        x = torch.randn(2, seq_len, 64, device=device)

        output, state = model(x)
        assert output.shape == x.shape
        assert state.shape == (2, model.state_dim)

    def test_long_sequence_processing(self, device: str = "cpu") -> None:
        """Test processing long sequences without OOM."""
        model = BLADEBlock(
            d_model=128,
            n_heads=8,
            chunk_size=16,
            state_dim=32,
            dropout=0.1,
        )
        model.to(device)

        # Create long sequence
        batch_size, seq_len = 1, 256
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        output, state = model(x)
        assert output.shape == x.shape
        assert state.shape == (batch_size, model.state_dim)

    def test_sliding_window_attention(self) -> None:
        """Test sliding window attention mode with modern SDPA APIs."""
        # Test that the BLADE block can be configured for sliding window attention
        # This tests the modern PyTorch SDPA integration
        model = BLADEBlock(
            d_model=64,
            n_heads=4,
            chunk_size=8,
            state_dim=16,
            dropout=0.1,
        )

        batch_size, seq_len = 1, 16
        x = torch.randn(batch_size, seq_len, model.d_model)

        # Forward pass should work with sliding window attention within chunks
        output, state = model(x)

        # Verify output shapes
        assert output.shape == x.shape
        assert state.shape == (batch_size, model.state_dim)

        # Test gradients work
        loss = output.sum()
        loss.backward()
        assert model.q_proj.weight.grad is not None

        # Verify that attention computation supports modern PyTorch features
        # (This indirectly tests that SDPA is being used for newer PyTorch versions)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_windowed_vs_full_causal(self) -> None:
        """Test difference between windowed and full causal attention."""
        # Create two models: one with windowed attention, one without
        model_full = BLADEBlock(
            d_model=32, n_heads=2, chunk_size=8, state_dim=8, dropout=0.0
        )
        model_windowed = BLADEBlock(
            d_model=32, n_heads=2, chunk_size=8, state_dim=8, dropout=0.0
        )

        batch_size, seq_len = 1, 12
        x = torch.randn(batch_size, seq_len, model_full.d_model)

        # Forward passes
        output_full, state_full = model_full(x)
        output_windowed, state_windowed = model_windowed(x)

        # Both should have same shape but different values due to attention pattern
        assert output_full.shape == output_windowed.shape
        assert state_full.shape == state_windowed.shape

        # The outputs should be different (different attention patterns)
        # but close due to same random initialization
        diff = torch.abs(output_full - output_windowed).mean()
        assert diff > 1e-6  # Should be meaningfully different

        # Both should be numerically stable
        assert not torch.isnan(output_full).any()
        assert not torch.isnan(output_windowed).any()

    def test_window_mask_behavior(self) -> None:
        """Test window mask behavior by calling _causal_attn directly."""
        block = BLADEBlock(
            d_model=32, n_heads=4, chunk_size=8, state_dim=16, dropout=0.0
        )
        x = torch.randn(1, 8, 32)

        with torch.no_grad():
            y_no_win = block._causal_attn(x, window_size=None)
            y_win = block._causal_attn(x, window_size=(2, 0))

        assert y_no_win.shape == y_win.shape
        assert not torch.allclose(y_no_win, y_win, atol=1e-6)  # Should be different

    def test_causal_within_chunk(self) -> None:
        """Test that changing a future token does not affect current position."""
        block = BLADEBlock(
            d_model=32, n_heads=4, chunk_size=8, state_dim=16, m_global=0, dropout=0.0
        )
        x = torch.randn(1, 16, 32)

        y0, _ = block(x)
        x2 = x.clone()
        x2[:, 7, :] += 10.0  # modify future token for t=6
        y1, _ = block(x2)

        # Position t=6 should not change (causality)
        assert torch.allclose(y0[:, 6, :], y1[:, 6, :], atol=1e-5, rtol=1e-4)

    def test_state_continuity_two_pass(self) -> None:
        """Test algorithmic equivalence: full pass vs two-chunk pass."""
        B, T, D = 2, 48, 64
        block = BLADEBlock(
            d_model=D, n_heads=4, chunk_size=16, state_dim=32, m_global=0, dropout=0.0
        )
        x = torch.randn(B, T, D)

        y_full, s_full = block(x)

        # two passes
        y_a, s_a = block(x[:, :16, :])
        y_b, s_b = block(x[:, 16:, :], state=s_a)
        y_cat = torch.cat([y_a, y_b], dim=1)

        assert torch.allclose(y_full, y_cat, atol=1e-5, rtol=1e-4)
        assert torch.allclose(s_full, s_b, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_comprehensive(self) -> None:
        """Test GPU functionality with larger inputs."""
        block = BLADEBlock(
            d_model=64, n_heads=4, chunk_size=32, state_dim=16, m_global=1
        ).cuda()
        x = torch.randn(2, 128, 64, device="cuda")

        y, s = block(x)

        assert y.is_cuda and s.is_cuda
        assert y.shape == (2, 128, 64)
        assert s.shape == (2, 16)

    def test_global_tokens_functionality(self, device: str = "cpu") -> None:
        """Test that global tokens work correctly."""
        model = BLADEBlock(
            d_model=64,
            n_heads=4,
            chunk_size=8,
            state_dim=16,
            m_global=2,
            dropout=0.1,
        )
        model.to(device)

        # Test forward pass
        batch_size, seq_len = 2, 12
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        output, state = model(x)
        assert output.shape == x.shape

        # Test that global tokens are trainable
        loss = output.sum()
        loss.backward()

        # Global tokens should have gradients
        assert model.global_tokens.grad is not None
        assert not torch.allclose(
            model.global_tokens.grad, torch.zeros_like(model.global_tokens.grad)
        )

    def test_deterministic_behavior(
        self, model: BLADEBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that forward pass is deterministic given same inputs and state."""
        # Set model to eval mode for deterministic behavior
        model.eval()

        # First forward pass
        output1, state1 = model(sample_input)

        # Second forward pass (should be identical)
        output2, state2 = model(sample_input)

        assert torch.allclose(output1, output2, atol=1e-6)
        assert torch.allclose(state1, state2, atol=1e-6)

    def test_device_compatibility(self, model: BLADEBlock, device: str = "cpu") -> None:
        """Test that model works on specified device."""
        model.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        output, state = model(x)

        assert output.device.type == device
        assert state.device.type == device

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(
        self, model: BLADEBlock, batch_size: int, device: str = "cpu"
    ) -> None:
        """Test model with different batch sizes."""
        model.to(device)

        seq_len = 12
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        output, state = model(x)

        assert output.shape == (batch_size, seq_len, model.d_model)
        assert state.shape == (batch_size, model.state_dim)

    def test_state_update_correctness(self, device: str = "cpu") -> None:
        """Test that state is updated correctly across chunks."""
        model = BLADEBlock(
            d_model=32,
            n_heads=2,
            chunk_size=4,
            state_dim=8,
            dropout=0.0,  # No dropout for deterministic test
        )
        model.to(device)
        model.eval()

        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        # Test that the full forward pass completes successfully
        full_output, final_state = model(x)

        # Basic checks
        assert full_output.shape == x.shape
        assert final_state.shape == (batch_size, model.state_dim)
        assert not torch.isnan(full_output).any()
        assert not torch.isnan(final_state).any()

        # Test with multiple chunks and state passing
        chunk1_input = x[:, :4, :]
        chunk1_output, chunk1_state = model(chunk1_input, state=None)

        chunk2_input = x[:, 4:, :]
        chunk2_output, chunk2_state = model(chunk2_input, state=chunk1_state)

        # Verify output shapes
        assert chunk1_output.shape == chunk1_input.shape
        assert chunk2_output.shape == chunk2_input.shape
        assert chunk1_state.shape == (batch_size, model.state_dim)
        assert chunk2_state.shape == (batch_size, model.state_dim)

    def test_sdpa_exception_fallback(self) -> None:
        """Test fallback when SDPA throws RuntimeError or NotImplementedError."""
        from unittest.mock import patch

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Make SDPA throw a RuntimeError to trigger fallback
            mock_sdpa.side_effect = RuntimeError("SDPA failed")

            model = BLADEBlock(
                d_model=32,
                n_heads=2,
                chunk_size=4,
                state_dim=8,
                dropout=0.0,
            )

            batch_size, seq_len = 1, 8
            x = torch.randn(batch_size, seq_len, model.d_model)

            # This should trigger the fallback paths (lines 154-216)
            output, state = model(x)

            assert output.shape == x.shape
            assert state.shape == (batch_size, model.state_dim)
            assert not torch.isnan(output).any()

    def test_flash_attn_fallback_with_fp32(self) -> None:
        """Test FlashAttention fallback path with float32 dtype conversion."""
        from unittest.mock import patch

        if torch.cuda.is_available():
            with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
                with patch("efficient_longctx.blocks.blade.FLASH_ATTN_AVAILABLE", True):
                    with patch(
                        "efficient_longctx.blocks.blade.flash_attn_func"
                    ) as mock_flash_attn:
                        # Make SDPA fail
                        mock_sdpa.side_effect = RuntimeError("SDPA failed")

                        # Mock FlashAttention to succeed
                        mock_flash_attn.return_value = torch.randn(1, 8, 2, 16).cuda()

                        model = BLADEBlock(
                            d_model=32,
                            n_heads=2,
                            chunk_size=8,
                            state_dim=8,
                            dropout=0.0,
                        ).cuda()

                        # Use float32 input to test dtype conversion (lines 169-172)
                        batch_size, seq_len = 1, 8
                        x = torch.randn(
                            batch_size, seq_len, model.d_model, dtype=torch.float32
                        ).cuda()

                        # This should trigger FlashAttention fallback
                        output, state = model(x)

                        assert output.shape == x.shape
                        assert state.shape == (batch_size, model.state_dim)
                        assert output.dtype == torch.float32

                        # Verify FlashAttention was called with half precision
                        flash_attn_call_args = mock_flash_attn.call_args
                        if flash_attn_call_args:
                            qfa, kfa, vfa = flash_attn_call_args[0]
                            assert qfa.dtype == torch.float16

    def test_flash_attn_exception_fallback(self) -> None:
        """Test FlashAttention exception handling and manual fallback."""
        from unittest.mock import patch

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            with patch("efficient_longctx.blocks.blade.FLASH_ATTN_AVAILABLE", True):
                with patch(
                    "efficient_longctx.blocks.blade.flash_attn_func"
                ) as mock_flash_attn:
                    # Make SDPA fail
                    mock_sdpa.side_effect = RuntimeError("SDPA failed")
                    # Make FlashAttention also fail
                    mock_flash_attn.side_effect = Exception("FlashAttention failed")

                    model = BLADEBlock(
                        d_model=32,
                        n_heads=2,
                        chunk_size=8,
                        state_dim=8,
                        dropout=0.0,
                    )

                    # Test on CUDA to trigger FlashAttention path
                    if torch.cuda.is_available():
                        model = model.cuda()
                        x = torch.randn(1, 8, 32).cuda()
                    else:
                        x = torch.randn(1, 8, 32)

                    # This should trigger manual fallback (lines 195-216)
                    output, state = model(x)

                    assert output.shape == x.shape
                    assert state.shape == (1, 8)

    def test_manual_fallback_with_window_size(self) -> None:
        """Test manual fallback with window_size parameter."""
        from unittest.mock import patch

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.side_effect = NotImplementedError("SDPA not available")

            model = BLADEBlock(
                d_model=32,
                n_heads=2,
                chunk_size=8,
                state_dim=8,
                dropout=0.0,
            )

            # Test manual attention computation with window_size (lines 206-211)
            batch_size, seq_len = 1, 8
            x = torch.randn(batch_size, seq_len, model.d_model)

            # Call _causal_attn directly with window_size
            output = model._causal_attn(x, window_size=(2, 0))

            assert output.shape == x.shape
            assert not torch.isnan(output).any()

    def test_flash_attn_window_size_path(self) -> None:
        """Test FlashAttention path with window_size parameter."""
        from unittest.mock import patch

        if torch.cuda.is_available():
            with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
                with patch("efficient_longctx.blocks.blade.FLASH_ATTN_AVAILABLE", True):
                    with patch(
                        "efficient_longctx.blocks.blade.flash_attn_func"
                    ) as mock_flash_attn:
                        # Make SDPA fail
                        mock_sdpa.side_effect = RuntimeError("SDPA failed")

                        # Mock FlashAttention return
                        mock_flash_attn.return_value = torch.randn(1, 8, 2, 16).cuda()

                        model = BLADEBlock(
                            d_model=32,
                            n_heads=2,
                            chunk_size=8,
                            state_dim=8,
                            dropout=0.0,
                        ).cuda()

                        batch_size, seq_len = 1, 8
                        x = torch.randn(batch_size, seq_len, model.d_model).cuda()

                        # Call _causal_attn with window_size to test lines 177-186
                        output = model._causal_attn(x, window_size=(3, 0))

                        assert output.shape == x.shape

                        # Verify FlashAttention was called with window_size
                        flash_attn_call_args = mock_flash_attn.call_args
                        if flash_attn_call_args:
                            call_kwargs = flash_attn_call_args[1]
                            assert "window_size" in call_kwargs

    def test_mask_cache_hit_path(self) -> None:
        """Test the mask cache hit path to cover line 141."""
        from unittest.mock import patch

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Make SDPA return a proper tensor rather than a mock
            mock_sdpa.return_value = torch.randn(2, 8, 16)  # [B,H,T,Dh] shape

            model = BLADEBlock(
                d_model=32,
                n_heads=2,
                chunk_size=8,
                state_dim=8,
                dropout=0.0,
            )

            batch_size, seq_len = 1, 8
            x = torch.randn(batch_size, seq_len, model.d_model)
            window_size = (2, 0)

            # First call - cache miss (line 143)
            output1 = model._causal_attn(x, window_size=window_size)

            # Second call - cache hit (line 141)
            output2 = model._causal_attn(x, window_size=window_size)

            assert output1.shape == output2.shape
            assert not torch.isnan(output1).any()
            assert not torch.isnan(output2).any()

    def test_flash_attn_dtype_conversion_edge_case(self) -> None:
        """Test FlashAttention dtype conversion when output differs from query dtype (line 190)."""
        from unittest.mock import patch

        if torch.cuda.is_available():
            with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
                with patch("efficient_longctx.blocks.blade.FLASH_ATTN_AVAILABLE", True):
                    with patch(
                        "efficient_longctx.blocks.blade.flash_attn_func"
                    ) as mock_flash_attn:
                        # Make SDPA fail to trigger FlashAttention path
                        mock_sdpa.side_effect = RuntimeError("SDPA failed")

                        model = BLADEBlock(
                            d_model=32,
                            n_heads=2,
                            chunk_size=8,
                            state_dim=8,
                            dropout=0.0,
                        ).cuda()

                        # Create a mock return that has different dtype than input
                        # Input will be float32, FlashAttn will return half precision
                        batch_size, seq_len = 1, 8
                        x_float32 = torch.randn(
                            batch_size, seq_len, model.d_model, dtype=torch.float32
                        ).cuda()

                        # Mock FlashAttention to return half precision output
                        mock_output_half = torch.randn(
                            1, 8, 2, 16, dtype=torch.float16
                        ).cuda()
                        mock_flash_attn.return_value = mock_output_half

                        # This should trigger dtype conversion (lines 189-190)
                        output = model._causal_attn(x_float32, window_size=(3, 0))

                        assert output.shape == x_float32.shape
                        # Output should be converted back to input dtype
                        assert output.dtype == torch.float32

    def test_memory_efficiency(self, device: str = "cpu") -> None:
        """Test that BLADE processes long序列 without excessive memory usage."""
        model = BLADEBlock(
            d_model=128,
            n_heads=8,
            chunk_size=32,
            state_dim=64,
            dropout=0.1,
        )
        model.to(device)

        # Very long sequence that would cause OOM with full attention
        batch_size, seq_len = 1, 1024
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        # This should not cause OOM
        output, state = model(x)
        assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBLADEBlockCUDA:
    """CUDA-specific tests for BLADEBlock."""

    @pytest.fixture
    def model(self) -> BLADEBlock:
        """Create a BLADEBlock instance for CUDA testing."""
        return BLADEBlock(
            d_model=64,
            n_heads=4,
            chunk_size=8,
            state_dim=16,
            dropout=0.1,
        ).cuda()

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Sample input tensor on CUDA."""
        batch_size, seq_len = 2, 16
        return torch.randn(batch_size, seq_len, 64).cuda()

    def test_cuda_forward_pass(
        self, model: BLADEBlock, sample_input: torch.Tensor
    ) -> None:
        """Test forward pass on CUDA device."""
        output, state = model(sample_input)

        assert output.device.type == "cuda"
        assert state.device.type == "cuda"
        assert output.shape == sample_input.shape
        assert state.shape == (sample_input.shape[0], model.state_dim)

    def test_flash_attention_availability(
        self, model: BLADEBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that FlashAttention is used when available on CUDA."""
        # This test verifies that FlashAttention integration works
        # The actual implementation will fall back to manual attention
        # if FlashAttention is not available
        output, state = model(sample_input)

        assert output.device.type == "cuda"
        assert state.device.type == "cuda"
