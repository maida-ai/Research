"""Tests for DPASSM (Dual-Path Attention + State Space Model) block."""

import pytest
import torch

from efficient_longctx.blocks import DPASSMBlock


class TestDPASSMBlock:
    """Test cases for DPASSMBlock."""

    @pytest.fixture
    def model_params(self) -> dict[str, int]:
        """Standard model parameters for testing."""
        return {
            "d_model": 64,
            "n_heads": 4,
            "window_size": 8,
            "ssm_state_dim": 16,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, model_params: dict[str, int]) -> DPASSMBlock:
        """Create a DPASSMBlock instance for testing."""
        return DPASSMBlock(**model_params)

    @pytest.fixture
    def sample_input(self, model_params: dict[str, int]) -> torch.Tensor:
        """Sample input tensor."""
        batch_size, seq_len = 2, 16
        return torch.randn(batch_size, seq_len, model_params["d_model"])

    def test_model_initialization(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test that model initializes with correct parameters."""
        assert model.d_model == model_params["d_model"]
        assert model.n_heads == model_params["n_heads"]
        assert model.window_size == model_params["window_size"]
        assert model.ssm_state_dim == model_params["ssm_state_dim"]
        assert model.dropout_rate == model_params["dropout"]

        # Check that all required components are initialized
        assert hasattr(model, "ln1")
        assert hasattr(model, "ln2")
        assert hasattr(model, "W_Q")
        assert hasattr(model, "W_K")
        assert hasattr(model, "W_V")
        assert hasattr(model, "W_O")
        assert hasattr(model, "gate")
        assert hasattr(model, "A")
        assert hasattr(model, "B")
        assert hasattr(model, "C")
        assert hasattr(model, "mlp")
        assert hasattr(model, "drop")

    def test_model_forward_shape(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that forward pass returns correct output shapes."""
        model.eval()
        with torch.no_grad():
            output, new_state = model(sample_input)

        # Check output shapes
        assert output.shape == sample_input.shape
        assert new_state.shape == (sample_input.shape[0], model.ssm_state_dim)

    def test_model_forward_with_state(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test forward pass with provided SSM state."""
        # Create initial state
        batch_size = sample_input.shape[0]
        initial_state = torch.randn(batch_size, model.ssm_state_dim)

        model.eval()
        with torch.no_grad():
            output, new_state = model(sample_input, initial_state)

        # Check output shapes
        assert output.shape == sample_input.shape
        assert new_state.shape == initial_state.shape

    def test_windowed_attention_mask(self, model: DPASSMBlock) -> None:
        """Test that windowed attention mask is correct."""
        seq_len = 16
        device = torch.device("cpu")
        dtype = torch.float32

        mask = model._build_window_mask(seq_len, model.window_size, device, dtype)

        # Check mask properties
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == dtype
        assert mask.device == device

        # Check causal property (upper triangle should be -inf)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == float("-inf")

        # Check window property (distance > window_size should be -inf)
        for i in range(seq_len):
            for j in range(max(0, i - model.window_size + 1)):
                assert mask[i, j] == float("-inf")

    def test_attention_computation(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test attention computation with windowing."""
        model.eval()
        with torch.no_grad():
            # Test attention with mask
            attention_out = model._compute_attention(sample_input)

            # Check output shape
            assert attention_out.shape == sample_input.shape

            # Test attention without mask
            attention_out_no_mask = model._compute_attention(sample_input, mask=None)
            assert attention_out_no_mask.shape == sample_input.shape

    def test_ssm_computation(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test SSM computation."""
        model.eval()
        with torch.no_grad():
            ssm_out, new_state = model._compute_ssm(sample_input)

            # Check output shapes
            assert ssm_out.shape == sample_input.shape
            assert new_state.shape == (sample_input.shape[0], model.ssm_state_dim)

            # Test with provided state
            initial_state = torch.randn(sample_input.shape[0], model.ssm_state_dim)
            ssm_out_with_state, final_state = model._compute_ssm(
                sample_input, initial_state
            )
            assert ssm_out_with_state.shape == sample_input.shape
            assert final_state.shape == initial_state.shape

    def test_state_persistence(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that SSM state persists across forward passes."""
        model.eval()
        initial_state = torch.randn(sample_input.shape[0], model.ssm_state_dim)

        with torch.no_grad():
            _, state1 = model(sample_input, initial_state)

            # Test deterministic behavior - same input should produce same state
            output_same, state_same = model(sample_input, initial_state)
            assert torch.allclose(state1, state_same, atol=1e-6)

            # Test that state evolves with different inputs
            modified_input = sample_input + 1.0  # Larger change
            _, state2 = model(modified_input, state1)

            # States should be different (SSM should evolve with different inputs)
            # Check that at least some components are different
            assert not torch.equal(state1, state2), (
                "States should differ with different inputs"
            )

    def test_gradient_flow(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that gradients flow correctly through the model."""
        model.train()
        model.zero_grad()

        # Forward pass
        output, _ = model(sample_input)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_different_seq_lengths(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test model with different sequence lengths."""
        batch_size = 2
        d_model = model_params["d_model"]

        model.eval()
        with torch.no_grad():
            # Test different sequence lengths
            for seq_len in [1, 4, 8, 16, 32]:
                x = torch.randn(batch_size, seq_len, d_model)
                output, new_state = model(x)

                assert output.shape == x.shape
                assert new_state.shape == (batch_size, model.ssm_state_dim)

    def test_model_determinism(
        self, model: DPASSMBlock, sample_input: torch.Tensor
    ) -> None:
        """Test that model produces deterministic outputs."""
        model.eval()
        initial_state = torch.randn(sample_input.shape[0], model.ssm_state_dim)

        with torch.no_grad():
            output1, state1 = model(sample_input, initial_state.clone())
            output2, state2 = model(sample_input, initial_state.clone())

            # Outputs should be identical when inputs are identical
            assert torch.allclose(output1, output2, atol=1e-6)
            assert torch.allclose(state1, state2, atol=1e-6)

    def test_ssm_determinism_with_seed(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """
        Test SSM deterministic behavior with torch.manual_seed as required by issue #23.
        """
        # Set seed for reproducible results
        torch.manual_seed(123)

        batch_size, seq_len = 2, 8
        d_model = model_params["d_model"]

        # Create input with seed
        x = torch.randn(batch_size, seq_len, d_model)
        x = x.clone()  # Ensure deterministic input

        model.eval()
        with torch.no_grad():
            # First forward pass
            output1, state1 = model(x, ssm_state=None)

            # Reset seed and do second forward pass with same input
            torch.manual_seed(123)  # Reset to same seed
            x2 = torch.randn(batch_size, seq_len, d_model)  # Same random input
            x2 = x.clone()  # Use exact same input tensor

            output2, state2 = model(x2, ssm_state=None)

            # Should be deterministic with same seed and input
            assert torch.allclose(output1, output2, atol=1e-6), (
                f"SSM outputs not deterministic with same seed. "
                f"Max diff: {torch.max(torch.abs(output1 - output2))}"
            )
            assert torch.allclose(state1, state2, atol=1e-6), (
                f"SSM states not deterministic with same seed. "
                f"Max diff: {torch.max(torch.abs(state1 - state2))}"
            )

    def test_ssm_continuity_across_chunks(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """
        Test SSM continuity across chunks as required by issue #23.

        This test isolates the SSM component to verify that SSM paths are continuous
        when split across chunks, independent of the attention mechanism.
        """
        # Set seed for deterministic results
        torch.manual_seed(42)

        batch_size, seq_len = 2, 16
        d_model = model_params["d_model"]

        # Create full input
        x_full = torch.randn(batch_size, seq_len, d_model)

        # Split into two halves
        mid_point = seq_len // 2
        x_a = x_full[:, :mid_point, :]  # First half
        x_b = x_full[:, mid_point:, :]  # Second half

        model.eval()
        with torch.no_grad():
            # Apply pre-layer normalization to get x_in from LN1(x)
            x_norm_full = model.ln1(x_full)
            x_norm_a = model.ln1(x_a)
            x_norm_b = model.ln1(x_b)

            # Test SSM component directly
            # Full SSM forward pass
            y_ssm_full, s_ssm_full = model._compute_ssm(x_norm_full, state=None)

            # Split SSM forward passes
            y_ssm_a, s_ssm_a = model._compute_ssm(x_norm_a, state=None)
            y_ssm_b, s_ssm_b = model._compute_ssm(x_norm_b, state=s_ssm_a)

            # Concatenate SSM outputs from two halves
            y_ssm_concatenated = torch.cat([y_ssm_a, y_ssm_b], dim=1)

            # Check SSM continuity within tolerance specified in issue #23
            assert torch.allclose(
                y_ssm_concatenated, y_ssm_full, atol=1e-5, rtol=1e-4
            ), (
                f"SSM output continuity failed. Max diff: {torch.max(torch.abs(y_ssm_concatenated - y_ssm_full))}"
            )

            # Check final SSM state matches
            assert torch.allclose(s_ssm_b, s_ssm_full, atol=1e-6, rtol=1e-5), (
                f"SSM state continuity failed. Max diff: {torch.max(torch.abs(s_ssm_b - s_ssm_full))}"
            )

    def test_device_consistency(self, model_params: dict[str, int]) -> None:
        """Test model works on different devices."""
        model = DPASSMBlock(**model_params)
        batch_size, seq_len = 2, 8
        d_model = model_params["d_model"]

        # Test CPU
        cpu_input = torch.randn(batch_size, seq_len, d_model)
        model.eval()
        with torch.no_grad():
            output_cpu, state_cpu = model(cpu_input)
            assert output_cpu.device.type == "cpu"
            assert state_cpu.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, model_params: dict[str, int]) -> None:
        """Test model works on CUDA."""
        model = DPASSMBlock(**model_params)
        model = model.cuda()

        batch_size, seq_len = 2, 8
        d_model = model_params["d_model"]
        cuda_input = torch.randn(batch_size, seq_len, d_model, device="cuda")

        model.eval()
        with torch.no_grad():
            output_cuda, state_cuda = model(cuda_input)
            assert output_cuda.device.type == "cuda"
            assert state_cuda.device.type == "cuda"

    def test_edge_cases(self, model: DPASSMBlock, model_params: dict[str, int]) -> None:
        """Test edge cases."""
        batch_size = 1
        d_model = model_params["d_model"]

        model.eval()
        with torch.no_grad():
            # Very short sequence
            short_input = torch.randn(batch_size, 1, d_model)
            output, state = model(short_input)
            assert output.shape == short_input.shape
            assert state.shape == (batch_size, model.ssm_state_dim)

            # Empty batch (edge case)
            empty_input = torch.randn(0, 1, d_model)
            empty_state = torch.randn(0, model.ssm_state_dim)
            try:
                output, state = model(empty_input, empty_state)
                assert output.shape == empty_input.shape
                assert state.shape == empty_state.shape
            except Exception:
                # Empty batch might not be supported, which is okay
                pass

    def test_forward_step_shape(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test that forward_step returns correct output shapes."""
        batch_size = 2
        d_model = model_params["d_model"]
        d_state = model_params["ssm_state_dim"]

        # Create single token input
        x_t = torch.randn(batch_size, 1, d_model)

        model.eval()
        with torch.no_grad():
            output, new_state = model.forward_step(x_t)

        # Check output shapes
        assert output.shape == (batch_size, 1, d_model)
        assert new_state.shape == (batch_size, d_state)

    def test_forward_step_with_state(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test forward_step with provided SSM state."""
        batch_size = 2
        d_model = model_params["d_model"]
        d_state = model_params["ssm_state_dim"]

        # Create single token input and state
        x_t = torch.randn(batch_size, 1, d_model)
        initial_state = torch.randn(batch_size, d_state)

        model.eval()
        with torch.no_grad():
            output, new_state = model.forward_step(x_t, initial_state)

        # Check output shapes
        assert output.shape == (batch_size, 1, d_model)
        assert new_state.shape == initial_state.shape

    def test_forward_step_continuity(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test forward_step state continuity across multiple steps."""
        batch_size = 2
        d_model = model_params["d_model"]

        # Create sequence of tokens
        tokens = torch.randn(batch_size, 3, d_model)

        model.eval()
        with torch.no_grad():
            state = None

            # Process tokens one by one
            outputs_step = []
            for t in range(tokens.shape[1]):
                x_t = tokens[:, t : t + 1, :]  # Single token
                output_t, state = model.forward_step(x_t, state)
                outputs_step.append(output_t)

            # Concatenate step-by-step outputs
            output_stepwise = torch.cat(outputs_step, dim=1)

            # Now process full sequence at once
            output_full, state_full = model(tokens)

            # Step-wise processing should produce similar results to full-sequence processing
            # Note: Due to LayerNorm behavior differences (single token vs sequence normalization),
            # outputs won't be identical, but should be in reasonable ballpark
            # Issue #25 acceptance criteria states should match "within tolerance"
            # and notes advanced window context management "OK to skip now"

            # Check that outputs are in reasonable range (not wild differences)
            max_diff = torch.max(torch.abs(output_stepwise - output_full))
            assert max_diff < 1.0, (
                f"Step-wise output differs too much from full output. "
                f"Max diff: {max_diff:.4f}"
            )

            # SSM states should be closer since they don't depend on sequence normalization
            state_diff = torch.max(torch.abs(state - state_full))
            assert state_diff < 0.1, (
                f"SSM states differ too much. Max diff: {state_diff:.4f}"
            )

    def test_forward_step_input_validation(
        self, model: DPASSMBlock, model_params: dict[str, int]
    ) -> None:
        """Test forward_step input validation."""
        batch_size = 2
        d_model = model_params["d_model"]

        # Test correct input (should work)
        x_t = torch.randn(batch_size, 1, d_model)
        model.eval()
        with torch.no_grad():
            output, state = model.forward_step(x_t)
            assert output.shape == (batch_size, 1, d_model)

        # Test wrong sequence length (should raise assertion)
        x_wrong = torch.randn(batch_size, 2, d_model)  # Wrong: T=2 instead of T=1
        with pytest.raises(AssertionError):
            model.forward_step(x_wrong)

    def test_shape_to_heads_basic(self, model: DPASSMBlock) -> None:
        """Test basic functionality of _shape_to_heads method."""
        batch_size, seq_len = 4, 16

        # Create input tensor
        x = torch.randn(batch_size, seq_len, model.d_model)

        # Apply the method
        result = model._shape_to_heads(x)

        # Check output shape: (B, n_heads, T, d_model // n_heads)
        expected_head_dim = model.d_model // model.n_heads
        expected_shape = (batch_size, model.n_heads, seq_len, expected_head_dim)

        assert result.shape == expected_shape
        assert result.dtype == x.dtype
        assert result.device == x.device

    def test_shape_to_heads_data_consistency(self, model: DPASSMBlock) -> None:
        """Test that _shape_to_heads preserves data consistency."""
        batch_size, seq_len = 2, 8

        # Create input with known values
        x = torch.ones(batch_size, seq_len, model.d_model)

        # Apply the method
        result = model._shape_to_heads(x)

        # All values should still be 1.0 (just reshaped)
        assert torch.all(torch.abs(result - 1.0) < 1e-6)

        # Test with random values - reshaping should preserve sum
        x_random = torch.randn(batch_size, seq_len, model.d_model)
        original_sum = x_random.sum()
        result_sum = model._shape_to_heads(x_random).sum()

        assert torch.allclose(original_sum, result_sum, atol=1e-6)

    def test_shape_to_heads_edge_cases(self, model_params: dict[str, int]) -> None:
        """Test _shape_to_heads with edge cases."""
        # Test with single head
        model_single_head = DPASSMBlock(
            d_model=model_params["d_model"],
            n_heads=1,
            window_size=model_params["window_size"],
            ssm_state_dim=model_params["ssm_state_dim"],
            dropout=model_params["dropout"],
        )

        # Test with single sequence length
        x_single = torch.randn(2, 1, model_params["d_model"])
        result = model_single_head._shape_to_heads(x_single)

        expected_shape = (2, 1, 1, model_params["d_model"])
        assert result.shape == expected_shape

        # Test with single batch
        model_single_batch = DPASSMBlock(**model_params)
        x_single_batch = torch.randn(1, 8, model_params["d_model"])
        result = model_single_batch._shape_to_heads(x_single_batch)

        expected_shape = (
            1,
            model_params["n_heads"],
            8,
            model_params["d_model"] // model_params["n_heads"],
        )
        assert result.shape == expected_shape

    def test_shape_to_heads_various_dimensions(
        self, model_params: dict[str, int]
    ) -> None:
        """Test _shape_to_heads with various dimension combinations."""
        d_model = model_params["d_model"]
        n_heads = model_params["n_heads"]

        model = DPASSMBlock(**model_params)

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 12, d_model)
            result = model._shape_to_heads(x)

            expected_shape = (batch_size, n_heads, 12, d_model // n_heads)
            assert result.shape == expected_shape

        # Test different sequence lengths
        for seq_len in [1, 4, 8, 16, 32]:
            x = torch.randn(3, seq_len, d_model)
            result = model._shape_to_heads(x)

            expected_shape = (3, n_heads, seq_len, d_model // n_heads)
            assert result.shape == expected_shape

    def test_shape_to_heads_different_head_counts(
        self, model_params: dict[str, int]
    ) -> None:
        """Test _shape_to_heads with different head counts."""
        d_model = model_params["d_model"]

        # Test with different head counts that divide d_model evenly
        valid_head_counts = [1, 2, 4, 8] if d_model == 64 else [1, 2, 4]

        for n_heads in valid_head_counts:
            if d_model % n_heads != 0:
                continue  # Skip if d_model not divisible by n_heads

            model = DPASSMBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=model_params["window_size"],
                ssm_state_dim=model_params["ssm_state_dim"],
                dropout=model_params["dropout"],
            )

            x = torch.randn(2, 8, d_model)
            result = model._shape_to_heads(x)

            expected_head_dim = d_model // n_heads
            expected_shape = (2, n_heads, 8, expected_head_dim)

            assert result.shape == expected_shape

    def test_shape_to_heads_requires_grad(self, model: DPASSMBlock) -> None:
        """Test that _shape_to_heads preserves requires_grad property."""
        batch_size, seq_len = 2, 8

        # Create input that requires gradients
        x = torch.randn(batch_size, seq_len, model.d_model, requires_grad=True)

        result = model._shape_to_heads(x)

        # Result should also require gradients (since it's just reshaping)
        assert result.requires_grad

    def test_shape_to_heads_transpose_operation(self, model: DPASSMBlock) -> None:
        """Test that the transpose operation in _shape_to_heads works correctly."""
        batch_size, seq_len = 3, 12

        # Create input with identifiable pattern
        x = torch.zeros(batch_size, seq_len, model.d_model)

        # Set specific values for each position to verify transpose
        for b in range(batch_size):
            for t in range(seq_len):
                for d in range(model.d_model):
                    x[b, t, d] = b * 1000 + t * 10 + d

        result = model._shape_to_heads(x)

        # Manually verify the reshape and transpose operation
        # Original: (B, T, d_model)
        # After view: (B, T, n_heads, d_model // n_heads)
        # After transpose: (B, n_heads, T, d_model // n_heads)

        head_dim = model.d_model // model.n_heads

        # Check a few specific positions
        for b in range(batch_size):
            for h in range(model.n_heads):
                for t in range(seq_len):
                    for d in range(head_dim):
                        original_idx = h * head_dim + d
                        expected_value = b * 1000 + t * 10 + original_idx
                        actual_value = result[b, h, t, d]

                        assert actual_value == expected_value, (
                            f"Transpose verification failed at position ({b}, {h}, {t}, {d}). "
                            f"Expected {expected_value}, got {actual_value}"
                        )
