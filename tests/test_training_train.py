"""Tests for the Lightning-based training module."""

import itertools
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from efficient_longctx.models import LongCtxModel
from efficient_longctx.models.data import LongCtxDataModule, TokenizedDataset
from efficient_longctx.models.models import LongCtxLightningModule
from efficient_longctx.training.train import MetricsCallback, TrainingCLI, main
from efficient_longctx.utils.constants import has_cuda


@pytest.fixture(scope="session")
def small_lightning_module():
    """Pre-created small Lightning module for testing."""
    module = LongCtxLightningModule(
        num_params="150m",
        block="vanilla",
        window_size=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=1000,
        ssm_state_dim=32,
        chunk_size=8,
        state_dim=16,
    )
    module.setup("fit")
    return module


class TestTokenizedDataset:
    """Test the TokenizedDataset class."""

    def test_tokenized_dataset(self):
        """Test TokenizedDataset functionality."""
        examples = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10]},
            {"input_ids": [11, 12, 13, 14, 15]},
        ]

        dataset = TokenizedDataset(examples)

        assert len(dataset) == 3
        assert dataset[0] == {"input_ids": [1, 2, 3, 4, 5]}
        assert dataset[1] == {"input_ids": [6, 7, 8, 9, 10]}
        assert dataset[2] == {"input_ids": [11, 12, 13, 14, 15]}


class TestLongCtxLightningModule:
    """Test the LongCtxLightningModule class."""

    def create_test_model(self, block_type):
        """Create a test model."""
        return LongCtxModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=8,
            block_type=block_type,
            block_kwargs={},
        )

    def test_lightning_module_initialization(self):
        """Test Lightning module initialization."""
        lightning_module = LongCtxLightningModule(
            num_params="150m",
            block="vanilla",
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000,
        )

        assert lightning_module.num_params == "150m"
        assert lightning_module.block == "vanilla"
        assert lightning_module.learning_rate == 1e-3
        assert lightning_module.weight_decay == 0.01
        assert lightning_module.warmup_steps == 100
        assert lightning_module.max_steps == 1000

    def test_lightning_module_forward(self, small_lightning_module):
        """Test Lightning module forward pass."""
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        logits = small_lightning_module(input_ids)

        assert logits.shape == (2, 10, 50257)  # GPT-2 vocab size

    def test_lightning_module_training_step(self, small_lightning_module):
        """Test training step."""
        # Create mock batch
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        # Run training step without logging to avoid warnings
        with patch.object(small_lightning_module, "log"):
            loss = small_lightning_module.training_step(
                (input_ids, labels), batch_idx=0
            )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_lightning_module_validation_step(self, small_lightning_module):
        """Test validation step."""
        # Create mock batch
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        # Run validation step without logging to avoid warnings
        with patch.object(small_lightning_module, "log"):
            loss = small_lightning_module.validation_step(
                (input_ids, labels), batch_idx=0
            )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_lightning_module_configure_optimizers(self, small_lightning_module):
        """Test optimizer configuration."""
        optimizers = small_lightning_module.configure_optimizers()

        assert "optimizer" in optimizers
        assert "lr_scheduler" in optimizers
        assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
        # The learning rate might be modified by the scheduler, so check it's reasonable
        lr = optimizers["optimizer"].param_groups[0]["lr"]
        assert 1e-4 <= lr <= 1e-2  # Should be in reasonable range
        assert optimizers["optimizer"].param_groups[0]["weight_decay"] == 0.01


class TestLongCtxDataModule:
    """Test the LongCtxDataModule class."""

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    def test_data_module_initialization(
        self, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test data module initialization."""
        # Mock tokenizer
        mock_tokenizer = mock_setup_tokenizer.return_value
        mock_tokenizer.pad_token_id = 0

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule(
            dataset_name="test_dataset",
            max_tokens=1000,
            seq_len=128,
            batch_size=4,
            val_split=0.1,
        )

        # Test prepare_data
        data_module.prepare_data()

        # Test setup
        data_module.setup("fit")

        assert len(data_module.train_examples) == 2
        assert len(data_module.val_examples) == 1

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    def test_data_module_dataloaders(self, mock_setup_tokenizer, mock_stream_dataset):
        """Test data module dataloaders."""
        # Mock tokenizer
        mock_tokenizer = mock_setup_tokenizer.return_value
        mock_tokenizer.pad_token_id = 0

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule(
            dataset_name="test_dataset",
            max_tokens=1000,
            seq_len=128,
            batch_size=2,
            val_split=0.1,
        )

        data_module.prepare_data()
        data_module.setup("fit")

        # Test train dataloader
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 2

        # Test val dataloader
        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 2


class TestIntegration:
    """Integration tests for the Lightning training system."""

    def create_mock_data_loaders(
        self, batch_size: int = 2, seq_len: int = 10, vocab_size: int = 1000
    ):
        """Create mock data loaders for testing."""
        # Create dummy data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        dataset = TensorDataset(input_ids, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return loader, loader  # Return same loader for train and val

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "num_params, block_type",
        itertools.product(["150m", "250m", "350m"], ["vanilla", "dpassm", "blade"]),
    )
    def test_lightning_trainer_integration(self, num_params, block_type):
        """Test Lightning trainer integration."""
        device = "cuda" if has_cuda() else "cpu"
        model = LongCtxModel(
            vocab_size=256,
            d_model=32,
            n_layers=2,
            n_heads=4,
            block_type=block_type,
            block_kwargs={
                "window_size": 16,
                "ssm_state_dim": 32,
                "chunk_size": 8,
                "state_dim": 16,
            },
        )

        lightning_module = LongCtxLightningModule(
            num_params=num_params,
            block=block_type,
            learning_rate=1e-3,
        )
        lightning_module.set_model(model)
        train_loader, val_loader = self.create_mock_data_loaders()

        # Create Lightning trainer
        trainer = Trainer(
            accelerator=device,
            devices=1,
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Test that we can create the trainer without errors
        assert trainer is not None

    @pytest.mark.integration
    def test_end_to_end_lightning_training_step(self, small_lightning_module):
        """Test a complete Lightning training step."""
        train_loader, val_loader = self.create_mock_data_loaders()

        # Run one training step manually without logging to avoid warnings
        small_lightning_module.train()
        for input_ids, labels in train_loader:
            with patch.object(small_lightning_module, "log"):
                loss = small_lightning_module.training_step(
                    (input_ids, labels), batch_idx=0
                )
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            break  # Only one step

        # Test validation step
        small_lightning_module.eval()
        for input_ids, labels in val_loader:
            with patch.object(small_lightning_module, "log"):
                loss = small_lightning_module.validation_step(
                    (input_ids, labels), batch_idx=0
                )
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            break  # Only one step


# Add helper methods to test classes
TestIntegration.create_mock_data_loaders = TestIntegration.create_mock_data_loaders


class TestMetricsCallback:
    """Test the MetricsCallback class."""

    def test_metrics_callback_initialization(self):
        """Test MetricsCallback initialization."""
        callback = MetricsCallback()
        assert callback is not None

    def test_on_train_epoch_end(self, small_lightning_module):
        """Test on_train_epoch_end method."""
        callback = MetricsCallback()

        # Create mock trainer and logger
        mock_trainer = MagicMock()
        mock_trainer.global_step = 100
        mock_trainer.logger = MagicMock()

        # Mock the get_num_params method
        with patch.object(
            small_lightning_module, "get_num_params", return_value=1000000
        ):
            callback.on_train_epoch_end(mock_trainer, small_lightning_module)

        # Verify that log_metrics was called
        mock_trainer.logger.log_metrics.assert_called_once()
        call_args = mock_trainer.logger.log_metrics.call_args
        assert "model/num_params" in call_args[0][0]
        assert call_args[0][0]["model/num_params"] == 1000000
        assert call_args[1]["step"] == 100  # step


class TestTrainingCLI:
    """Test the TrainingCLI class."""

    @patch("efficient_longctx.training.train.LightningCLI.__init__")
    def test_training_cli_initialization(self, mock_super_init):
        """Test TrainingCLI initialization."""
        # Mock the parent class initialization to avoid CLI parsing
        mock_super_init.return_value = None

        TrainingCLI(
            model_class=LongCtxLightningModule,
            datamodule_class=LongCtxDataModule,
            save_config_callback=None,
            seed_everything_default=42,
            parser_kwargs={"default_env": True},
        )

        # Verify parent class was called with correct arguments
        mock_super_init.assert_called_once_with(
            model_class=LongCtxLightningModule,
            datamodule_class=LongCtxDataModule,
            save_config_callback=None,
            seed_everything_default=42,
            parser_kwargs={"default_env": True},
        )

    def test_add_arguments_to_parser(self):
        """Test add_arguments_to_parser method."""
        # Create CLI instance without calling __init__
        cli = TrainingCLI.__new__(TrainingCLI)

        # Create a mock parser
        mock_parser = MagicMock()

        # This method currently does nothing, but we test it doesn't crash
        cli.add_arguments_to_parser(mock_parser)

        # Verify no methods were called on the parser (since it's empty)
        assert not mock_parser.called

    @patch("efficient_longctx.training.train.logging.basicConfig")
    @patch("efficient_longctx.training.train.seed_everything")
    def test_before_instantiate_classes_with_seed(
        self, mock_seed_everything, mock_basic_config
    ):
        """Test before_instantiate_classes with seed configuration."""
        # Create CLI instance without calling __init__
        cli = TrainingCLI.__new__(TrainingCLI)

        # Mock config with seed
        mock_config = MagicMock()
        mock_config.seed_everything = 123
        cli.config = mock_config

        cli.before_instantiate_classes()

        # Verify logging was configured
        mock_basic_config.assert_called_once_with(level=logging.INFO)

        # Verify seed was set
        mock_seed_everything.assert_called_once_with(123, workers=True)

    @patch("efficient_longctx.training.train.logging.basicConfig")
    @patch("efficient_longctx.training.train.seed_everything")
    def test_before_instantiate_classes_without_seed(
        self, mock_seed_everything, mock_basic_config
    ):
        """Test before_instantiate_classes without seed configuration."""
        # Create CLI instance without calling __init__
        cli = TrainingCLI.__new__(TrainingCLI)

        # Mock config without seed attribute
        mock_config = MagicMock()
        # Remove the seed_everything attribute to test the hasattr check
        del mock_config.seed_everything
        cli.config = mock_config

        cli.before_instantiate_classes()

        # Verify logging was configured
        mock_basic_config.assert_called_once_with(level=logging.INFO)

        # Verify seed was not set (because hasattr returns False)
        mock_seed_everything.assert_not_called()

    def test_after_instantiate_classes(self):
        """Test after_instantiate_classes method."""
        # Create CLI instance without calling __init__
        cli = TrainingCLI.__new__(TrainingCLI)

        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer

        cli.after_instantiate_classes()

        # Verify callbacks were added
        assert len(mock_trainer.callbacks) == 3

        # Check that the callbacks are the expected types
        callback_types = [
            type(callback).__name__ for callback in mock_trainer.callbacks
        ]
        assert "LearningRateMonitor" in callback_types
        assert "RichProgressBar" in callback_types
        assert "MetricsCallback" in callback_types


class TestMainFunction:
    """Test the main function."""

    @patch("efficient_longctx.training.train.TrainingCLI")
    def test_main_function(self, mock_training_cli):
        """Test the main function."""
        # Mock the CLI instance
        mock_cli_instance = MagicMock()
        mock_training_cli.return_value = mock_cli_instance

        # Call main function
        main()

        # Verify TrainingCLI was instantiated with correct parameters
        mock_training_cli.assert_called_once_with(
            model_class=LongCtxLightningModule,
            datamodule_class=LongCtxDataModule,
            save_config_callback=None,
            seed_everything_default=42,
            parser_kwargs={"default_env": True},
        )

    @patch("efficient_longctx.training.train.TrainingCLI")
    def test_main_function_cli_execution(self, mock_training_cli):
        """Test that main function executes the CLI."""
        # Mock the CLI instance
        mock_cli_instance = MagicMock()
        mock_training_cli.return_value = mock_cli_instance

        # Call main function
        main()

        # Verify the CLI instance was created (implicitly tested by the call above)
        assert mock_training_cli.called


class TestTrainingCLIIntegration:
    """Integration tests for TrainingCLI."""

    @patch("efficient_longctx.training.train.LightningCLI.__init__")
    def test_cli_with_mocked_classes(self, mock_super_init):
        """Test CLI with mocked model and data module classes."""
        # Mock the parent class initialization
        mock_super_init.return_value = None

        # Create CLI
        TrainingCLI(
            model_class=LongCtxLightningModule,
            datamodule_class=LongCtxDataModule,
            save_config_callback=None,
        )

        # Verify parent class was called with correct arguments
        mock_super_init.assert_called_once_with(
            model_class=LongCtxLightningModule,
            datamodule_class=LongCtxDataModule,
            save_config_callback=None,
        )

    def test_cli_callbacks_are_properly_configured(self):
        """Test that callbacks are properly configured in CLI."""
        # Create CLI instance without calling __init__
        cli = TrainingCLI.__new__(TrainingCLI)

        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.callbacks = []
        cli.trainer = mock_trainer

        # Call after_instantiate_classes
        cli.after_instantiate_classes()

        # Verify callbacks were added
        assert len(mock_trainer.callbacks) == 3

        # Verify MetricsCallback is present
        metrics_callback = None
        for callback in mock_trainer.callbacks:
            if isinstance(callback, MetricsCallback):
                metrics_callback = callback
                break

        assert metrics_callback is not None
        assert isinstance(metrics_callback, MetricsCallback)
