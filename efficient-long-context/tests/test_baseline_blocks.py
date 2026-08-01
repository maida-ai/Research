"""Tests for baseline attention blocks (Longformer and BigBird)."""

import torch

from efficient_longctx.blocks.bigbird import BigBirdBlock
from efficient_longctx.blocks.longformer import LongformerBlock


class TestLongformerBlock:
    """Test Longformer baseline block."""

    def test_longformer_initialization(self):
        """Test Longformer block initialization."""
        block = LongformerBlock(
            d_model=768,
            n_heads=12,
            window_size=128,
            n_global_tokens=2,
            dropout=0.1,
        )

        assert block.d_model == 768
        assert block.n_heads == 12
        assert block.window_size == 128
        assert block.n_global_tokens == 2
        assert block.head_dim == 768 // 12

    def test_longformer_forward_shape(self):
        """Test Longformer forward pass shape preservation."""
        block = LongformerBlock(
            d_model=256,
            n_heads=8,
            window_size=64,
            n_global_tokens=2,
        )

        batch_size, seq_len, d_model = 2, 128, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is None
        assert output.dtype == x.dtype

    def test_longformer_attention_mask(self):
        """Test Longformer attention mask creation."""
        block = LongformerBlock(
            d_model=128,
            n_heads=4,
            window_size=32,
            n_global_tokens=2,
        )

        seq_len = 64
        device = torch.device("cpu")
        mask = block._create_attention_mask(seq_len, device)

        # Check mask shape
        assert mask.shape == (1, 1, seq_len, seq_len)

        # Check that mask has finite values where attention is allowed
        assert torch.isfinite(mask).any()
        assert torch.isinf(mask).any()

    def test_longformer_gradient_flow(self):
        """Test that gradients flow through Longformer block."""
        block = LongformerBlock(
            d_model=128,
            n_heads=4,
            window_size=32,
        )

        x = torch.randn(1, 64, 128, requires_grad=True)
        output, state = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_longformer_different_window_sizes(self):
        """Test Longformer with different window sizes."""
        d_model, n_heads = 128, 4
        batch_size, seq_len = 2, 64

        for window_size in [16, 32, 64]:
            block = LongformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
            )

            x = torch.randn(batch_size, seq_len, d_model)
            output, state = block(x)

            assert output.shape == x.shape
            assert state is None

    def test_longformer_different_global_tokens(self):
        """Test Longformer with different numbers of global tokens."""
        d_model, n_heads = 128, 4
        batch_size, seq_len = 2, 64

        for n_global in [0, 1, 2]:
            block = LongformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=32,
                n_global_tokens=n_global,
            )

            x = torch.randn(batch_size, seq_len, d_model)
            output, state = block(x)

            assert output.shape == x.shape
            assert state is None


class TestBigBirdBlock:
    """Test BigBird baseline block."""

    def test_bigbird_initialization(self):
        """Test BigBird block initialization."""
        block = BigBirdBlock(
            d_model=768,
            n_heads=12,
            window_size=128,
            n_random_tokens=4,
            n_global_tokens=2,
            dropout=0.1,
        )

        assert block.d_model == 768
        assert block.n_heads == 12
        assert block.window_size == 128
        assert block.n_random_tokens == 4
        assert block.n_global_tokens == 2
        assert block.head_dim == 768 // 12

    def test_bigbird_forward_shape(self):
        """Test BigBird forward pass shape preservation."""
        block = BigBirdBlock(
            d_model=256,
            n_heads=8,
            window_size=64,
            n_random_tokens=2,
            n_global_tokens=2,
        )

        batch_size, seq_len, d_model = 2, 128, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is None
        assert output.dtype == x.dtype

    def test_bigbird_attention_mask(self):
        """Test BigBird attention mask creation."""
        block = BigBirdBlock(
            d_model=128,
            n_heads=4,
            window_size=32,
            n_random_tokens=2,
            n_global_tokens=2,
        )

        seq_len = 64
        device = torch.device("cpu")
        mask = block._create_attention_mask(seq_len, device)

        # Check mask shape
        assert mask.shape == (1, 1, seq_len, seq_len)

        # Check that mask has finite values where attention is allowed
        assert torch.isfinite(mask).any()
        assert torch.isinf(mask).any()

    def test_bigbird_gradient_flow(self):
        """Test that gradients flow through BigBird block."""
        block = BigBirdBlock(
            d_model=128,
            n_heads=4,
            window_size=32,
            n_random_tokens=2,
        )

        x = torch.randn(1, 64, 128, requires_grad=True)
        output, state = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_bigbird_different_window_sizes(self):
        """Test BigBird with different window sizes."""
        d_model, n_heads = 128, 4
        batch_size, seq_len = 2, 64

        for window_size in [16, 32, 64]:
            block = BigBirdBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                n_random_tokens=2,
            )

            x = torch.randn(batch_size, seq_len, d_model)
            output, state = block(x)

            assert output.shape == x.shape
            assert state is None

    def test_bigbird_different_random_tokens(self):
        """Test BigBird with different numbers of random tokens."""
        d_model, n_heads = 128, 4
        batch_size, seq_len = 2, 64

        for n_random in [0, 2, 4]:
            block = BigBirdBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=32,
                n_random_tokens=n_random,
            )

            x = torch.randn(batch_size, seq_len, d_model)
            output, state = block(x)

            assert output.shape == x.shape
            assert state is None

    def test_bigbird_different_global_tokens(self):
        """Test BigBird with different numbers of global tokens."""
        d_model, n_heads = 128, 4
        batch_size, seq_len = 2, 64

        for n_global in [0, 1, 2]:
            block = BigBirdBlock(
                d_model=d_model,
                n_heads=n_heads,
                window_size=32,
                n_random_tokens=2,
                n_global_tokens=n_global,
            )

            x = torch.randn(batch_size, seq_len, d_model)
            output, state = block(x)

            assert output.shape == x.shape
            assert state is None


class TestBaselineBlocksIntegration:
    """Integration tests for baseline blocks."""

    def test_baseline_blocks_in_model_registry(self):
        """Test that baseline blocks are registered in model registry."""
        from efficient_longctx.models.models import get_layer

        # Check that baseline blocks are in the registry
        layers = get_layer()
        assert "baseline_longformer" in layers
        assert "baseline_bigbird" in layers

        # Check that we can instantiate them
        longformer_cls = layers["baseline_longformer"]
        bigbird_cls = layers["baseline_bigbird"]

        assert longformer_cls == LongformerBlock
        assert bigbird_cls == BigBirdBlock

    def test_baseline_blocks_with_model_creation(self):
        """Test creating models with baseline blocks."""
        from efficient_longctx.models.models import create_model

        # Test Longformer model
        model = create_model(
            vocab_size=1000,
            num_params="150m",
            block_type="baseline_longformer",
            block_kwargs={"window_size": 32, "n_global_tokens": 2},
        )

        assert model is not None
        assert len(model.layers) == 12

        # Test BigBird model
        model = create_model(
            vocab_size=1000,
            num_params="150m",
            block_type="baseline_bigbird",
            block_kwargs={
                "window_size": 32,
                "n_random_tokens": 2,
                "n_global_tokens": 2,
            },
        )

        assert model is not None
        assert len(model.layers) == 12

    def test_baseline_blocks_forward_pass(self):
        """Test forward pass through models with baseline blocks."""
        from efficient_longctx.models.models import create_model

        batch_size, seq_len = 2, 64
        vocab_size, _ = 1000, 128

        # Test Longformer model forward pass
        model = create_model(
            vocab_size=vocab_size,
            num_params="150m",
            block_type="baseline_longformer",
            block_kwargs={"window_size": 32},
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

        # Test BigBird model forward pass
        model = create_model(
            vocab_size=vocab_size,
            num_params="150m",
            block_type="baseline_bigbird",
            block_kwargs={"window_size": 32, "n_random_tokens": 2},
        )

        logits = model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_baseline_blocks_in_evaluation(self):
        """Test that baseline blocks work in evaluation script."""
        from efficient_longctx.evals.synthetic import SimpleModel

        # Test Longformer evaluator
        evaluator = SimpleModel(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            block_type="baseline_longformer",
            block_kwargs={"window_size": 32},
        )

        batch_size, seq_len = 2, 64
        x = torch.randint(0, 1000, (batch_size, seq_len))
        logits = evaluator(x)

        assert logits.shape == (batch_size, seq_len, 1000)

        # Test BigBird evaluator
        evaluator = SimpleModel(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            block_type="baseline_bigbird",
            block_kwargs={"window_size": 32, "n_random_tokens": 2},
        )

        logits = evaluator(x)

        assert logits.shape == (batch_size, seq_len, 1000)
