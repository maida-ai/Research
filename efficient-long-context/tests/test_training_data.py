"""Tests for data loading functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from efficient_longctx.training.data import (
    DataCollator,
    dry_run_samples,
    save_tokenizer,
    setup_tokenizer,
    stream_dataset,
)


class TestDataCollator:
    """Test DataCollator functionality."""

    def test_collator_basic(self):
        """Test basic collator functionality."""
        collator = DataCollator(seq_len=8, pad_token_id=0)

        # Create mock batch
        batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10, 11, 12]},
        ]

        input_ids, labels = collator(batch)

        # Check shapes
        assert input_ids.shape == (2, 8)
        assert labels.shape == (2, 8)

        # Check padding
        assert input_ids[0, 5:].tolist() == [0, 0, 0]  # Padded with pad_token_id
        assert labels[0, 4:].tolist() == [0, 0, 0, 0]  # Right-shifted and padded

        # Check right-shift
        assert labels[0, :4].tolist() == [2, 3, 4, 5]  # Right-shifted input_ids
        assert labels[1, :6].tolist() == [
            7,
            8,
            9,
            10,
            11,
            12,
        ]  # Right-shifted input_ids

    def test_collator_long_sequence(self):
        """Test collator with sequences longer than seq_len."""
        collator = DataCollator(seq_len=4, pad_token_id=0)

        batch = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9]}]

        input_ids, labels = collator(batch)

        # Should create multiple chunks
        assert input_ids.shape[0] == 3  # 9 tokens / 4 seq_len = 3 chunks
        assert input_ids.shape[1] == 4

        # Check first chunk
        assert input_ids[0].tolist() == [1, 2, 3, 4]
        assert labels[0].tolist() == [2, 3, 4, 0]  # Right-shifted and padded

        # Check second chunk
        assert input_ids[1].tolist() == [5, 6, 7, 8]
        assert labels[1].tolist() == [6, 7, 8, 0]  # Right-shifted and padded

        # Check third chunk (padded)
        assert input_ids[2].tolist() == [9, 0, 0, 0]
        assert labels[2].tolist() == [0, 0, 0, 0]  # Right-shifted and padded


class TestTokenizerSetup:
    """Test tokenizer setup functionality."""

    def test_setup_tokenizer_gpt2(self):
        """Test GPT-2 tokenizer setup."""
        tokenizer = setup_tokenizer("gpt2")

        assert tokenizer.pad_token == tokenizer.eos_token
        assert tokenizer.pad_token_id == tokenizer.eos_token_id
        assert tokenizer.pad_token_id is not None

    def test_setup_tokenizer_custom_model(self):
        """Test custom model tokenizer setup."""
        # Mock tokenizer to avoid downloading
        with patch(
            "efficient_longctx.training.data.AutoTokenizer"
        ) as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.eos_token = "<|endoftext|>"
            mock_tokenizer.eos_token_id = 50256
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            tokenizer = setup_tokenizer("custom-model")

            assert tokenizer.pad_token == "<|endoftext|>"
            mock_tokenizer_class.from_pretrained.assert_called_once_with("custom-model")


class TestSaveTokenizer:
    """Test tokenizer saving functionality."""

    def test_save_tokenizer(self):
        """Test saving tokenizer to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            save_path = Path(temp_dir) / "tokenizer"
            save_tokenizer(mock_tokenizer, str(save_path))

            # Check directory was created
            assert save_path.exists()
            assert save_path.is_dir()

            # Check save_pretrained was called
            mock_tokenizer.save_pretrained.assert_called_once_with(save_path)


class TestDryRunSamples:
    """Test dry run functionality."""

    @patch("efficient_longctx.training.data.load_dataset")
    def test_dry_run_samples_with_config(self, mock_load_dataset):
        """Test dry run samples with dataset configuration."""
        # Mock dataset
        mock_dataset = [
            {"text": "This is a sample text for testing."},
            {"text": "Another sample text with different content."},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        mock_tokenizer.decode.return_value = "decoded text"

        # Should not raise any exceptions
        dry_run_samples(
            "test_dataset", mock_tokenizer, num_samples=2, dataset_config="test-config"
        )

        # Verify dataset was loaded with config
        mock_load_dataset.assert_called_once_with(
            "test_dataset", "test-config", split="train", streaming=True
        )

    @patch("efficient_longctx.training.data.load_dataset")
    def test_dry_run_samples_empty_text_skip(self, mock_load_dataset):
        """Test dry run samples skipping empty text."""
        # Mock dataset with empty text
        mock_dataset = [
            {"text": ""},
            {"text": "   "},  # Whitespace only
            {"text": "Valid text"},
            {"text": ""},  # Another empty text
            {"text": "Another valid text"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        mock_tokenizer.decode.return_value = "decoded text"

        # Should not raise any exceptions and skip empty texts
        dry_run_samples("test_dataset", mock_tokenizer, num_samples=5)

        # Verify dataset was loaded
        mock_load_dataset.assert_called_once_with(
            "test_dataset", split="train", streaming=True
        )

    @patch("efficient_longctx.training.data.load_dataset")
    def test_dry_run_samples_num_samples_limit(self, mock_load_dataset):
        """Test dry run samples with num_samples limit that triggers break."""
        # Mock dataset with many examples
        mock_dataset = [{"text": f"Sample text {i}"} for i in range(10)]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        mock_tokenizer.decode.return_value = "decoded text"

        # Request only 3 samples
        dry_run_samples("test_dataset", mock_tokenizer, num_samples=3)

        # Verify dataset was loaded
        mock_load_dataset.assert_called_once_with(
            "test_dataset", split="train", streaming=True
        )

        # Verify tokenizer was called only 3 times (not for all 10 examples)
        assert mock_tokenizer.call_count == 3

        # Verify decode was called only 3 times
        assert mock_tokenizer.decode.call_count == 3

        # Verify the calls were made with the first 3 samples
        mock_tokenizer.assert_any_call("Sample text 0", truncation=True, max_length=100)
        mock_tokenizer.assert_any_call("Sample text 1", truncation=True, max_length=100)
        mock_tokenizer.assert_any_call("Sample text 2", truncation=True, max_length=100)

        # Ensure it didn't process beyond the limit by checking call count
        # If it processed more than 3, the call count would be higher
        assert mock_tokenizer.call_count == 3


class TestStreamDataset:
    """Test dataset streaming functionality."""

    @patch("efficient_longctx.training.data.load_dataset")
    def test_stream_dataset_basic(self, mock_load_dataset):
        """Test basic dataset streaming."""
        # Mock dataset with limited examples
        mock_dataset = [
            {"text": "Sample text 1"},
            {"text": "Sample text 2"},
            {"text": "Sample text 3"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        train_iter, val_iter = stream_dataset(
            "test_dataset", max_tokens=100, tokenizer=mock_tokenizer, val_split=0.5
        )

        # Convert iterators to lists for testing
        train_examples = list(train_iter)
        val_examples = list(val_iter)

        # Should have processed all examples
        assert len(train_examples) + len(val_examples) == 3

        # Verify tokenizer was called for each example
        assert mock_tokenizer.call_count == 3

    @patch("efficient_longctx.training.data.load_dataset")
    def test_stream_dataset_with_config(self, mock_load_dataset):
        """Test streaming with dataset configuration."""
        # Mock dataset with limited examples
        mock_dataset = [
            {"text": "Sample text 1"},
            {"text": "Sample text 2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        train_iter, val_iter = stream_dataset(
            "test_dataset",
            max_tokens=100,
            tokenizer=mock_tokenizer,
            val_split=0.0,
            dataset_config="test-config",
        )

        # Convert iterators to lists
        train_examples = list(train_iter)
        val_examples = list(val_iter)

        # Should have processed all examples
        assert len(train_examples) == 2
        assert len(val_examples) == 0

        # Verify load_dataset was called with config
        mock_load_dataset.assert_called_once_with(
            "test_dataset", "test-config", split="train", streaming=True
        )

    @patch("efficient_longctx.training.data.load_dataset")
    def test_stream_dataset_max_tokens_limit(self, mock_load_dataset):
        """Test streaming with max_tokens limit."""
        # Mock dataset with many examples
        mock_dataset = [{"text": f"Sample text {i}"} for i in range(100)]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer that returns different token counts
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = [
            {"input_ids": [1] * 10},  # 10 tokens
            {"input_ids": [2] * 20},  # 20 tokens
            {"input_ids": [3] * 15},  # 15 tokens
            {"input_ids": [4] * 5},  # 5 tokens
        ]

        train_iter, val_iter = stream_dataset(
            "test_dataset",
            max_tokens=30,  # Should stop after 2 examples (10 + 20 = 30)
            tokenizer=mock_tokenizer,
            val_split=0.0,
        )

        # Convert iterators to lists
        train_examples = list(train_iter)
        val_examples = list(val_iter)

        # Should have stopped at max_tokens limit
        assert len(train_examples) == 2  # Only first 2 examples processed
        assert len(val_examples) == 0

        # Verify tokenizer was called only twice
        assert mock_tokenizer.call_count == 2

    @patch("efficient_longctx.training.data.load_dataset")
    def test_stream_dataset_empty_text_continue(self, mock_load_dataset):
        """Test streaming with empty/whitespace text that triggers continue."""
        # Mock dataset with various empty text cases
        mock_dataset = [
            {"text": ""},  # Empty string
            {"text": "   "},  # Whitespace only
            {"text": "\t\n\r"},  # Various whitespace characters
            {"text": "Valid text"},  # Valid text
            {"text": ""},  # Another empty string
            {"text": "Another valid text"},  # Another valid text
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        train_iter, val_iter = stream_dataset(
            "test_dataset", max_tokens=100, tokenizer=mock_tokenizer, val_split=0.0
        )

        # Convert iterators to lists
        train_examples = list(train_iter)
        val_examples = list(val_iter)

        # Should only process valid text (2 out of 6 examples)
        assert len(train_examples) == 2
        assert len(val_examples) == 0

        # Verify tokenizer was called only for valid text
        assert mock_tokenizer.call_count == 2

        # Verify the calls were made with valid text
        mock_tokenizer.assert_any_call("Valid text", truncation=True, max_length=2048)
        mock_tokenizer.assert_any_call(
            "Another valid text", truncation=True, max_length=2048
        )


@pytest.mark.integration
class TestIntegration:
    """Integration tests for data loading."""

    def test_end_to_end_collator(self):
        """Test end-to-end collator with real tokenizer."""
        # Use a small, fast tokenizer for testing
        tokenizer = setup_tokenizer("gpt2")
        collator = DataCollator(seq_len=8, pad_token_id=tokenizer.pad_token_id)

        # Create sample batch
        batch = [
            {"input_ids": tokenizer.encode("Hello world!")},
            {"input_ids": tokenizer.encode("This is a test.")},
        ]

        input_ids, labels = collator(batch)

        # Basic shape checks
        assert input_ids.shape[0] == 2
        assert input_ids.shape[1] == 8
        assert labels.shape == input_ids.shape

        # Check that labels are right-shifted
        for i in range(input_ids.shape[0]):
            # Find non-padding positions
            non_pad_mask = input_ids[i] != tokenizer.pad_token_id
            if non_pad_mask.sum() > 1:  # More than 1 token
                # Labels should be right-shifted input_ids
                expected_labels = torch.cat(
                    [
                        input_ids[i][1 : non_pad_mask.sum()],
                        torch.full(
                            (8 - non_pad_mask.sum() + 1,), tokenizer.pad_token_id
                        ),
                    ]
                )
                assert torch.equal(labels[i], expected_labels)
