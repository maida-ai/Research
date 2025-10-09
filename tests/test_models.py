"""Tests for the models module."""

import itertools
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from efficient_longctx.models import (
    LongCtxLightningModule,
    LongCtxModel,
    VanillaAttentionBlock,
    create_model,
    get_config_params,
    get_layer,
    load_model_from_checkpoint,
)


@pytest.fixture(scope="session")
def small_model_config():
    """Small model configuration for fast testing."""
    return {
        "vocab_size": 1000,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 8,
    }


@pytest.fixture(scope="session")
def test_models(small_model_config):
    """Pre-created test models for reuse across tests."""
    models = {}

    # Create small test models for each block type
    for block_type in ["vanilla", "dpassm", "blade"]:
        if block_type == "dpassm":
            block_kwargs = {"window_size": 16, "ssm_state_dim": 32}
        elif block_type == "blade":
            block_kwargs = {"chunk_size": 8, "state_dim": 16}
        else:
            block_kwargs = {}

        models[block_type] = LongCtxModel(
            vocab_size=small_model_config["vocab_size"],
            d_model=small_model_config["d_model"],
            n_layers=small_model_config["n_layers"],
            n_heads=small_model_config["n_heads"],
            block_type=block_type,
            block_kwargs=block_kwargs,
        )

    return models


class TestVanillaAttentionBlock:
    """Test the vanilla attention block."""

    def test_forward_shape(self):
        """Test that forward pass maintains correct shapes."""
        block = VanillaAttentionBlock(d_model=64, n_heads=8)
        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        block = VanillaAttentionBlock(d_model=64, n_heads=8)

        # Test with different sequence lengths
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, 64)
            output = block(x)
            assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        block = VanillaAttentionBlock(d_model=64, n_heads=8)
        x = torch.randn(2, 10, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        for param in block.parameters():
            assert param.grad is not None


class TestLongCtxModel:
    """Test the LongCtxModel class."""

    def test_model_creation_dpassm(self):
        """Test creating model with DP-ASSM blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="dpassm",
            block_kwargs={"window_size": 16, "ssm_state_dim": 32},
        )

        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.n_layers == 2
        assert model.n_heads == 8
        assert model.block_type == "dpassm"
        assert len(model.layers) == 2

    def test_model_creation_blade(self):
        """Test creating model with BLADE blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="blade",
            block_kwargs={"chunk_size": 8, "state_dim": 16},
        )

        assert model.block_type == "blade"
        assert len(model.layers) == 2

    def test_model_creation_vanilla(self):
        """Test creating model with vanilla attention blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        assert model.block_type == "vanilla"
        assert len(model.layers) == 2

    def test_model_creation_invalid_block_type(self):
        """Test that invalid block type raises error."""
        with pytest.raises(ValueError, match="Unknown block type"):
            LongCtxModel(
                vocab_size=1000,
                d_model=64,
                n_layers=2,
                n_heads=8,
                block_type="invalid",
                block_kwargs={},
            )

    def test_forward_pass_dpassm(self):
        """Test forward pass with DP-ASSM blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="dpassm",
            block_kwargs={"window_size": 16, "ssm_state_dim": 32},
        )

        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids)

        assert logits.shape == (2, 10, 1000)

    def test_forward_pass_blade(self):
        """Test forward pass with BLADE blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="blade",
            block_kwargs={"chunk_size": 8, "state_dim": 16},
        )

        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids)

        assert logits.shape == (2, 10, 1000)

    def test_forward_pass_vanilla(self):
        """Test forward pass with vanilla attention blocks."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids)

        assert logits.shape == (2, 10, 1000)

    def test_forward_pass_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        for seq_len in [1, 5, 10, 20]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            logits = model(input_ids)
            assert logits.shape == (2, seq_len, 1000)

    def test_get_num_params(self):
        """Test parameter counting."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        num_params = model.get_num_params()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_get_model_config(self):
        """Test model configuration retrieval."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        config = model.get_model_config()
        assert config["vocab_size"] == 1000
        assert config["d_model"] == 64
        assert config["n_layers"] == 2
        assert config["n_heads"] == 8
        assert config["block_type"] == "vanilla"
        assert config["max_seq_len"] == 2048

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )

        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestGetConfigParams:
    """Test the get_config_params function."""

    def test_get_config_params_none(self):
        """Test getting all configs when num_params is None."""
        configs = get_config_params(None)
        assert isinstance(configs, dict)
        assert "150m" in configs
        assert "250m" in configs
        assert "350m" in configs

    def test_get_config_params_int(self):
        """Test getting config when num_params is an integer."""
        config = get_config_params(150)
        assert config["d_model"] == 768
        assert config["n_layers"] == 12
        assert config["n_heads"] == 12

    def test_get_config_params_string(self):
        """Test getting config when num_params is a string."""
        config = get_config_params("250m")
        assert config["d_model"] == 1024
        assert config["n_layers"] == 16
        assert config["n_heads"] == 16

    def test_get_config_params_invalid(self):
        """Test that invalid parameter count raises error."""
        with pytest.raises(ValueError, match="Unknown parameter count"):
            get_config_params("500m")


class TestGetLayer:
    """Test the get_layer function."""

    def test_get_layer_dpassm(self):
        """Test getting DP-ASSM layer."""
        layer_cls = get_layer("dpassm")
        assert layer_cls is not None
        # Test that we can instantiate it
        layer = layer_cls(d_model=64, n_heads=8, window_size=16, ssm_state_dim=32)
        assert layer is not None

    def test_get_layer_blade(self):
        """Test getting BLADE layer."""
        layer_cls = get_layer("blade")
        assert layer_cls is not None
        # Test that we can instantiate it
        layer = layer_cls(d_model=64, n_heads=8, chunk_size=8, state_dim=16)
        assert layer is not None

    def test_get_layer_vanilla(self):
        """Test getting vanilla attention layer."""
        layer_cls = get_layer("vanilla")
        assert layer_cls is not None
        # Test that we can instantiate it
        layer = layer_cls(d_model=64, n_heads=8)
        assert layer is not None

    def test_get_layer_invalid(self):
        """Test that invalid block type raises error."""
        with pytest.raises(ValueError, match="Unknown block type"):
            get_layer("invalid")

    def test_get_layer_none(self):
        """Test getting all available layers."""
        layers = get_layer(None)
        assert isinstance(layers, dict)
        assert "dpassm" in layers
        assert "blade" in layers
        assert "vanilla" in layers


class TestCreateModel:
    """Test the create_model function."""

    @pytest.mark.parametrize(
        "num_params, block_type",
        itertools.product(
            ["150m"], ["vanilla", "dpassm", "blade"]
        ),  # Only test 150m for speed
    )
    def test_create_model(self, num_params, block_type):
        """Test creating model with different parameter counts."""
        model = create_model(
            vocab_size=1000,
            params=num_params,
            block_type=block_type,
            block_kwargs={
                "window_size": 16,
                "ssm_state_dim": 32,
                "chunk_size": 8,
                "state_dim": 16,
            },
        )

        assert model.d_model == get_config_params(num_params)["d_model"]
        assert model.n_layers == get_config_params(num_params)["n_layers"]
        assert model.n_heads == get_config_params(num_params)["n_heads"]
        assert model.block_type == block_type

    def test_create_model_invalid_params(self):
        """Test that invalid parameter count raises error."""
        with pytest.raises(ValueError, match="Unknown parameter count"):
            create_model(
                vocab_size=1000,
                params="500m",
                block_type="vanilla",
                block_kwargs={},
            )


class TestLoadModelFromCheckpoint:
    """Test loading models from checkpoints."""

    @pytest.mark.integration
    @pytest.mark.parametrize("block_type", ["vanilla", "dpassm", "blade"])
    def test_load_model_from_checkpoint(self, block_type):
        """Test loading a model from a checkpoint."""
        # Create a model and save it
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type=block_type,
            block_kwargs={
                "window_size": 16,
                "ssm_state_dim": 32,
                "chunk_size": 8,
                "state_dim": 16,
            },
        )

        # Create a dummy checkpoint
        checkpoint = {
            "step": 100,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "model_config": model.get_model_config(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            # Load the model
            loaded_model = load_model_from_checkpoint(str(checkpoint_path))

            # Check that the loaded model has the same configuration
            assert loaded_model.vocab_size == model.vocab_size
            assert loaded_model.d_model == model.d_model
            assert loaded_model.n_layers == model.n_layers
            assert loaded_model.n_heads == model.n_heads
            assert loaded_model.block_type == model.block_type

            # Test that forward pass works
            input_ids = torch.randint(0, 1000, (2, 10))
            logits = loaded_model(input_ids)
            assert logits.shape == (2, 10, 1000)


class TestLongCtxLightningModule:
    """Test the LongCtxLightningModule class."""

    def test_lightning_module_creation(self):
        """Test creating Lightning module."""
        module = LongCtxLightningModule()
        assert module.params == "150m"
        assert module.block == "dpassm"
        assert module.window_size == 2048
        assert module.ssm_state_dim == 256
        assert module.chunk_size == 512
        assert module.state_dim == 128
        assert module.learning_rate == 3e-4
        assert module.weight_decay == 0.01
        assert module.warmup_steps == 1000
        assert module.max_steps == 100000
        assert module.tokenizer_name == "gpt2"

    def test_lightning_module_custom_params(self):
        """Test creating Lightning module with custom parameters."""
        module = LongCtxLightningModule(
            params="250m",
            block="blade",
            window_size=1024,
            ssm_state_dim=128,
            chunk_size=256,
            state_dim=64,
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=500,
            max_steps=50000,
            tokenizer_name="custom",
        )
        assert module.params == "250m"
        assert module.block == "blade"
        assert module.window_size == 1024
        assert module.ssm_state_dim == 128
        assert module.chunk_size == 256
        assert module.state_dim == 64
        assert module.learning_rate == 1e-4
        assert module.weight_decay == 0.1
        assert module.warmup_steps == 500
        assert module.max_steps == 50000
        assert module.tokenizer_name == "custom"

    def test_set_model(self):
        """Test setting a pre-created model."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)
        assert module.model is model

    def test_forward_without_model(self):
        """Test that forward raises error when model is not set."""
        module = LongCtxLightningModule()
        input_ids = torch.randint(0, 1000, (2, 10))
        with pytest.raises(RuntimeError, match="Model not initialized"):
            module(input_ids)

    def test_get_model_config_without_model(self):
        """Test getting model config when model is not set."""
        module = LongCtxLightningModule()
        config = module.get_model_config()
        assert config == {}

    def test_get_num_params_without_model(self):
        """Test getting number of parameters when model is not set."""
        module = LongCtxLightningModule()
        num_params = module.get_num_params()
        assert num_params == 0

    def test_get_model_config_with_model(self):
        """Test getting model config when model is set."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)
        config = module.get_model_config()
        assert config["vocab_size"] == 1000
        assert config["d_model"] == 64
        assert config["n_layers"] == 2
        assert config["n_heads"] == 8
        assert config["block_type"] == "vanilla"

    def test_get_num_params_with_model(self):
        """Test getting number of parameters when model is set."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)
        num_params = module.get_num_params()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)

        optimizers = module.configure_optimizers()
        assert "optimizer" in optimizers
        assert "lr_scheduler" in optimizers
        assert optimizers["lr_scheduler"]["interval"] == "step"
        assert optimizers["lr_scheduler"]["frequency"] == 1

    @pytest.mark.integration
    def test_training_step(self):
        """Test training step."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)

        # Create dummy batch
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        batch = (input_ids, labels)

        # Run training step without logging to avoid warnings
        with patch.object(module, "log"):
            loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    @pytest.mark.integration
    def test_validation_step(self):
        """Test validation step."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)

        # Create dummy batch
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        batch = (input_ids, labels)

        # Run validation step without logging to avoid warnings
        with patch.object(module, "log"):
            loss = module.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.integration
    def test_math_import_usage(self):
        """Test that math import is used correctly in training/validation steps."""
        module = LongCtxLightningModule()
        model = LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type="vanilla",
            block_kwargs={},
        )
        module.set_model(model)

        # Create dummy batch
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        batch = (input_ids, labels)

        # Test training step (uses math.exp for perplexity calculation)
        with patch.object(module, "log"):
            loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

        # Test validation step (uses math.exp for perplexity calculation)
        with patch.object(module, "log"):
            loss = module.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.integration
    def test_logging_import_usage(self):
        """Test that logging import is used correctly in setup method."""
        from unittest.mock import patch

        module = LongCtxLightningModule()

        # Mock the setup_tokenizer to avoid actual tokenizer setup
        with patch(
            "efficient_longctx.models.models.setup_tokenizer"
        ) as mock_setup_tokenizer:
            mock_tokenizer = type("MockTokenizer", (), {"__len__": lambda self: 1000})()
            mock_setup_tokenizer.return_value = mock_tokenizer

            # Mock logging.info to capture the log message
            with patch("logging.info") as mock_logging:
                module.setup("fit")

                # Verify that logging.info was called
                mock_logging.assert_called_once()
                log_message = mock_logging.call_args[0][0]
                assert "Created" in log_message
                assert "model with" in log_message
                assert "parameters" in log_message

    @pytest.mark.integration
    def test_setup_blade_block(self):
        """Test setup method with blade block configuration."""
        from unittest.mock import patch

        module = LongCtxLightningModule(block="blade")

        # Mock the setup_tokenizer to avoid actual tokenizer setup
        with patch(
            "efficient_longctx.models.models.setup_tokenizer"
        ) as mock_setup_tokenizer:
            mock_tokenizer = type("MockTokenizer", (), {"__len__": lambda self: 1000})()
            mock_setup_tokenizer.return_value = mock_tokenizer

            # Mock logging.info to capture the log message
            with patch("logging.info") as mock_logging:
                module.setup("fit")

                # Verify that logging.info was called
                mock_logging.assert_called_once()
                log_message = mock_logging.call_args[0][0]
                assert "Created" in log_message
                assert "model with" in log_message
                assert "parameters" in log_message

                # Verify the model was created with blade configuration
                assert module.model is not None
                assert module.model.block_type == "blade"

    @pytest.mark.integration
    def test_setup_vanilla_block(self):
        """Test setup method with vanilla block configuration."""
        from unittest.mock import patch

        module = LongCtxLightningModule(block="vanilla")

        # Mock the setup_tokenizer to avoid actual tokenizer setup
        with patch(
            "efficient_longctx.models.models.setup_tokenizer"
        ) as mock_setup_tokenizer:
            mock_tokenizer = type("MockTokenizer", (), {"__len__": lambda self: 1000})()
            mock_setup_tokenizer.return_value = mock_tokenizer

            # Mock logging.info to capture the log message
            with patch("logging.info") as mock_logging:
                module.setup("fit")

                # Verify that logging.info was called
                mock_logging.assert_called_once()
                log_message = mock_logging.call_args[0][0]
                assert "Created" in log_message
                assert "model with" in log_message
                assert "parameters" in log_message

                # Verify the model was created with vanilla configuration
                assert module.model is not None
                assert module.model.block_type == "vanilla"


class TestIntegration:
    """Integration tests for the models module."""

    @pytest.mark.integration
    @pytest.mark.parametrize("block_type", ["dpassm", "blade", "vanilla"])
    def test_model_with_different_blocks(self, block_type):
        """Test that models with different blocks work correctly."""
        vocab_size = 1000
        d_model = 64
        n_layers = 2
        n_heads = 8

        model = LongCtxModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            block_type=block_type,
            block_kwargs={
                "window_size": 16,
                "ssm_state_dim": 32,
                "chunk_size": 8,
                "state_dim": 16,
            },
        )

        # Test forward pass for all models
        input_ids = torch.randint(0, vocab_size, (2, 10))

        logits = model(input_ids)

        assert logits.shape == (2, 10, vocab_size)
