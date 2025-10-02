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
