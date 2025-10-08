"""Tests for the models.data module."""

# Import the module directly to avoid triggering package-level imports
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from efficient_longctx.models.data import LongCtxDataModule, TokenizedDataset


class TestTokenizedDataset:
    """Test the TokenizedDataset class."""

    def test_init(self):
        """Test TokenizedDataset initialization."""
        examples = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10]},
            {"input_ids": [11, 12, 13, 14, 15]},
        ]

        dataset = TokenizedDataset(examples)
        assert dataset.examples == examples

    def test_len(self):
        """Test TokenizedDataset __len__ method."""
        examples = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6]},
            {"input_ids": [7, 8, 9]},
        ]

        dataset = TokenizedDataset(examples)
        assert len(dataset) == 3

    def test_getitem(self):
        """Test TokenizedDataset __getitem__ method."""
        examples = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10]},
            {"input_ids": [11, 12, 13, 14, 15]},
        ]

        dataset = TokenizedDataset(examples)

        assert dataset[0] == {"input_ids": [1, 2, 3, 4, 5]}
        assert dataset[1] == {"input_ids": [6, 7, 8, 9, 10]}
        assert dataset[2] == {"input_ids": [11, 12, 13, 14, 15]}

    def test_getitem_index_error(self):
        """Test TokenizedDataset __getitem__ with invalid index."""
        examples = [{"input_ids": [1, 2, 3]}]
        dataset = TokenizedDataset(examples)

        with pytest.raises(IndexError):
            dataset[1]

    def test_empty_dataset(self):
        """Test TokenizedDataset with empty examples."""
        dataset = TokenizedDataset([])
        assert len(dataset) == 0

        with pytest.raises(IndexError):
            dataset[0]


class TestLongCtxDataModule:
    """Test the LongCtxDataModule class."""

    def test_init_default_params(self):
        """Test LongCtxDataModule initialization with default parameters."""
        data_module = LongCtxDataModule()

        assert data_module.dataset_name == "openwebtext"
        assert data_module.max_tokens == 1000000
        assert data_module.seq_len == 2048
        assert data_module.batch_size == 16
        assert data_module.val_split == 0.01
        assert data_module.dataset_config is None
        assert data_module.tokenizer_name == "gpt2"
        assert data_module.num_workers == 4
        assert data_module.tokenizer is None
        assert data_module.collator is None

    def test_init_custom_params(self):
        """Test LongCtxDataModule initialization with custom parameters."""
        data_module = LongCtxDataModule(
            dataset_name="custom_dataset",
            max_tokens=500000,
            seq_len=1024,
            batch_size=8,
            val_split=0.05,
            dataset_config="custom_config",
            tokenizer_name="custom_tokenizer",
            num_workers=2,
        )

        assert data_module.dataset_name == "custom_dataset"
        assert data_module.max_tokens == 500000
        assert data_module.seq_len == 1024
        assert data_module.batch_size == 8
        assert data_module.val_split == 0.05
        assert data_module.dataset_config == "custom_config"
        assert data_module.tokenizer_name == "custom_tokenizer"
        assert data_module.num_workers == 2

    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_prepare_data(self, mock_data_collator, mock_setup_tokenizer):
        """Test prepare_data method."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_setup_tokenizer.return_value = mock_tokenizer

        # Mock DataCollator
        mock_collator_instance = MagicMock()
        mock_data_collator.return_value = mock_collator_instance

        data_module = LongCtxDataModule(seq_len=128)
        data_module.prepare_data()

        mock_setup_tokenizer.assert_called_once_with("gpt2")
        mock_data_collator.assert_called_once_with(128, 0)
        assert data_module.tokenizer == mock_tokenizer
        assert data_module.collator == mock_collator_instance

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_setup_fit_with_prepared_data(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test setup method with fit stage when data is already prepared."""
        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule()
        data_module.tokenizer = mock_tokenizer
        data_module.collator = mock_collator_instance

        data_module.setup("fit")

        mock_stream_dataset.assert_called_once_with(
            "openwebtext",
            1000000,
            mock_tokenizer,
            0.01,
            None,
        )
        assert data_module.train_examples == train_examples
        assert data_module.val_examples == val_examples

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_setup_fit_without_prepared_data(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test setup method with fit stage when data is not prepared."""
        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()
        mock_setup_tokenizer.return_value = mock_tokenizer
        mock_data_collator.return_value = mock_collator_instance

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule()
        data_module.setup("fit")

        mock_setup_tokenizer.assert_called_once_with("gpt2")
        mock_data_collator.assert_called_once_with(2048, 0)
        mock_stream_dataset.assert_called_once_with(
            "openwebtext",
            1000000,
            mock_tokenizer,
            0.01,
            None,
        )
        assert data_module.train_examples == train_examples
        assert data_module.val_examples == val_examples

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_setup_non_fit_stage(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test setup method with non-fit stage."""
        data_module = LongCtxDataModule()
        data_module.setup("test")

        # Should not call any setup methods for non-fit stages
        mock_setup_tokenizer.assert_not_called()
        mock_data_collator.assert_not_called()
        mock_stream_dataset.assert_not_called()

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_train_dataloader_success(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test train_dataloader method success case."""
        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()
        mock_setup_tokenizer.return_value = mock_tokenizer
        mock_data_collator.return_value = mock_collator_instance

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule(batch_size=2, num_workers=1)
        data_module.setup("fit")

        train_loader = data_module.train_dataloader()

        assert train_loader.batch_size == 2
        assert train_loader.num_workers == 1
        assert train_loader.collate_fn == mock_collator_instance
        assert train_loader.pin_memory is True

    def test_train_dataloader_error(self):
        """Test train_dataloader method error case."""
        data_module = LongCtxDataModule()

        with pytest.raises(RuntimeError, match="train_examples not found"):
            data_module.train_dataloader()

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_val_dataloader_success(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test val_dataloader method success case."""
        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()
        mock_setup_tokenizer.return_value = mock_tokenizer
        mock_data_collator.return_value = mock_collator_instance

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule(batch_size=2, num_workers=1)
        data_module.setup("fit")

        val_loader = data_module.val_dataloader()

        assert val_loader.batch_size == 2
        assert val_loader.num_workers == 1
        assert val_loader.collate_fn == mock_collator_instance
        assert val_loader.pin_memory is True

    def test_val_dataloader_error(self):
        """Test val_dataloader method error case."""
        data_module = LongCtxDataModule()

        with pytest.raises(RuntimeError, match="val_examples not found"):
            data_module.val_dataloader()

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_custom_parameters_passed_to_stream_dataset(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset
    ):
        """Test that custom parameters are passed correctly to stream_dataset."""
        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()
        mock_setup_tokenizer.return_value = mock_tokenizer
        mock_data_collator.return_value = mock_collator_instance

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}]
        val_examples = [{"input_ids": [4, 5, 6]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule(
            dataset_name="custom_dataset",
            max_tokens=500000,
            val_split=0.05,
            dataset_config="custom_config",
        )
        data_module.setup("fit")

        mock_stream_dataset.assert_called_once_with(
            "custom_dataset",
            500000,
            mock_tokenizer,
            0.05,
            "custom_config",
        )

    @patch("efficient_longctx.models.data.stream_dataset")
    @patch("efficient_longctx.models.data.setup_tokenizer")
    @patch("efficient_longctx.models.data.DataCollator")
    def test_logging_in_setup(
        self, mock_data_collator, mock_setup_tokenizer, mock_stream_dataset, caplog
    ):
        """Test that setup method logs the number of examples."""
        import logging

        # Mock tokenizer and collator
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_collator_instance = MagicMock()
        mock_setup_tokenizer.return_value = mock_tokenizer
        mock_data_collator.return_value = mock_collator_instance

        # Mock dataset streaming
        train_examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        val_examples = [{"input_ids": [7, 8, 9]}]
        mock_stream_dataset.return_value = (iter(train_examples), iter(val_examples))

        data_module = LongCtxDataModule()

        with caplog.at_level(logging.INFO):
            data_module.setup("fit")

        assert "Loaded 2 train examples" in caplog.text
        assert "Loaded 1 val examples" in caplog.text
