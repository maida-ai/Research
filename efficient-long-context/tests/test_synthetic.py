"""Tests for synthetic evaluation tasks."""

import pytest
import torch

from efficient_longctx.evals.synthetic import (
    CopyRecallDataset,
    DriftCurveDataset,
    PasskeyDataset,
    SimpleModel,
    evaluate_model,
    pretrain_model,
)


class TestSyntheticDatasets:
    """Test synthetic dataset generators."""

    def test_passkey_dataset(self):
        """Test passkey dataset generation."""
        dataset = PasskeyDataset(vocab_size=100, seed=42)
        input_ids, labels = dataset.generate_batch(batch_size=2, seq_len=10)

        # Check shapes
        assert input_ids.shape == (2, 10)
        assert labels.shape == (2, 10)

        # Check that passkey token is planted
        assert torch.any(input_ids == dataset.passkey_token)

        # Check that query token is at the end
        assert torch.all(input_ids[:, -1] == dataset.query_token)

        # Check that only query position has labels
        assert torch.all(labels[:, :-1] == -100)
        assert torch.all(labels[:, -1] == dataset.passkey_token)

    def test_copy_recall_dataset(self):
        """Test copy/recall dataset generation."""
        dataset = CopyRecallDataset(vocab_size=100, subsequence_len=3, seed=42)
        input_ids, labels = dataset.generate_batch(batch_size=2, seq_len=10)

        # Check shapes
        assert input_ids.shape == (2, 10)
        assert labels.shape == (2, 10)

        # Check that query token is at the end
        assert torch.all(input_ids[:, -1] == dataset.query_token)

        # Check that labels are set after query position
        assert torch.all(labels[:, 0] == -100)  # Query position should be ignored

    def test_drift_curve_dataset(self):
        """Test drift curve dataset generation."""
        dataset = DriftCurveDataset(vocab_size=100, seed=42)
        input_ids, labels = dataset.generate_batch(batch_size=2, seq_len=10)

        # Check shapes
        assert input_ids.shape == (2, 10)
        assert labels.shape == (2, 10)

        # Check that target token is at position 0
        assert torch.all(input_ids[:, 0] == dataset.target_token)

        # Check that query token is at the end
        assert torch.all(input_ids[:, -1] == dataset.query_token)

        # Check that only query position has labels
        assert torch.all(labels[:, :-1] == -100)
        assert torch.all(labels[:, -1] == dataset.target_token)


class TestSimpleModel:
    """Test simple model for synthetic evaluations."""

    @pytest.mark.parametrize(
        "block_type",
        ["dpassm", "blade", "vanilla", "baseline_longformer", "baseline_bigbird"],
    )
    def test_model(self, block_type):
        """Test different model types."""
        # Set up block-specific parameters
        block_kwargs = {
            "ssm_state_dim": 16,
            "window_size": 32,
            "chunk_size": 32,
            "state_dim": 16,
            "n_global_tokens": 2,
            "n_random_tokens": 4,
        }

        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type=block_type,
            block_kwargs=block_kwargs,
        )

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits = model(input_ids)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 100)

    def test_blade_model(self):
        """Test BLADE model."""
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="blade",
            chunk_size=32,
            state_dim=16,
        )

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits = model(input_ids)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 100)

    def test_longformer_model(self):
        """Test Longformer model."""
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="baseline_longformer",
            window_size=32,
            n_global_tokens=2,
        )

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits = model(input_ids)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 100)

    def test_bigbird_model(self):
        """Test BigBird model."""
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="baseline_bigbird",
            window_size=32,
            n_global_tokens=2,
            n_random_tokens=4,
        )

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits = model(input_ids)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 100)

    def test_model_forward_with_state(self):
        """Test model forward pass with state."""
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="dpassm",
            window_size=32,
            ssm_state_dim=16,
        )

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Test with state parameter
        logits = model(input_ids, state=None)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, 100)

    def test_invalid_block_type(self):
        """Test that invalid block type raises error."""
        with pytest.raises(ValueError, match="Unknown block type"):
            SimpleModel(
                vocab_size=100,
                d_model=64,
                n_heads=4,
                n_layers=1,
                block_type="invalid",
            )


class TestEvaluation:
    """Test evaluation functions."""

    def test_evaluate_model_passkey(self):
        """Test model evaluation on passkey task."""
        dataset = PasskeyDataset(vocab_size=100, seed=42)
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="dpassm",
            window_size=32,
            ssm_state_dim=16,
        )

        metrics = evaluate_model(model, dataset, seq_len=10, batch_size=2, device="cpu")

        # Check metrics structure
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

        # Check metric types
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["total_correct"], int)
        assert isinstance(metrics["total_tokens"], int)

        # Check metric ranges
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["total_correct"] >= 0
        assert metrics["total_tokens"] >= 0

    def test_evaluate_model_copy_recall(self):
        """Test model evaluation on copy/recall task."""
        dataset = CopyRecallDataset(vocab_size=100, subsequence_len=3, seed=42)
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="blade",
            chunk_size=32,
            state_dim=16,
        )

        metrics = evaluate_model(model, dataset, seq_len=10, batch_size=2, device="cpu")

        # Check metrics structure
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

        # Check metric ranges
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["total_correct"] >= 0
        assert metrics["total_tokens"] >= 0

    def test_evaluate_model_drift_curve(self):
        """Test model evaluation on drift curve task."""
        dataset = DriftCurveDataset(vocab_size=100, seed=42)
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="dpassm",
            window_size=32,
            ssm_state_dim=16,
        )

        metrics = evaluate_model(model, dataset, seq_len=10, batch_size=2, device="cpu")

        # Check metrics structure
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

        # Check metric ranges
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["total_correct"] >= 0
        assert metrics["total_tokens"] >= 0


class TestPretraining:
    """Test pretraining functionality."""

    def test_pretrain_model(self):
        """Test model pretraining."""
        dataset = PasskeyDataset(vocab_size=100, seed=42)
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="dpassm",
            window_size=32,
            ssm_state_dim=16,
        )

        # Test pretraining
        metrics = pretrain_model(model, dataset, seq_len=10, num_steps=5, device="cpu")

        # Check metrics structure
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

        # Check metric types and ranges
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] >= 0.0

    def test_pretrained_model_loading(self):
        """Test loading pretrained models."""
        # Test with invalid model name (should fallback gracefully)
        model = SimpleModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=1,
            block_type="dpassm",
            pretrained_model="invalid_model_name",
            window_size=32,
            ssm_state_dim=16,
        )

        # Should still work with fallback
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        assert logits.shape == (2, 10, 100)

    def test_plot_drift_curve(self):
        """Test drift curve plotting functionality."""
        import os
        import tempfile
        from pathlib import Path

        from efficient_longctx.evals.synthetic import plot_drift_curve

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create mock results
            results = {
                32: {"accuracy": 0.8},
                64: {"accuracy": 0.7},
                128: {"accuracy": 0.6},
            }

            # Test plotting
            plot_drift_curve(results, "test_task", output_dir)

            # Check that file was created
            expected_file = output_dir / "test_task_drift_curve.png"
            assert expected_file.exists()

            # Clean up
            if expected_file.exists():
                os.remove(expected_file)


class TestIntegration:
    """Integration tests for synthetic evaluation tasks."""

    @pytest.mark.integration
    def test_passkey_task_integration(self):
        """Test passkey task integration."""
        from efficient_longctx.evals.synthetic import run_passkey_task

        results = run_passkey_task(
            seq_len=32,
            max_len=32,
            model_path="",
            block_type="dpassm",
            device="cpu",
            pretrain_steps=0,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            window_size=16,
            ssm_state_dim=16,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert 32 in results

        # Check metrics structure
        metrics = results[32]
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

    @pytest.mark.integration
    def test_passkey_task_with_pretraining(self):
        """Test passkey task with pretraining."""
        from efficient_longctx.evals.synthetic import run_passkey_task

        results = run_passkey_task(
            seq_len=16,
            max_len=16,
            model_path="",
            block_type="dpassm",
            device="cpu",
            pretrain_steps=10,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            window_size=8,
            ssm_state_dim=16,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert 16 in results

        # Check metrics structure
        metrics = results[16]
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

    @pytest.mark.integration
    def test_copy_recall_task_integration(self):
        """Test copy/recall task integration."""
        from efficient_longctx.evals.synthetic import run_copy_recall_task

        results = run_copy_recall_task(
            seq_len=32,
            max_len=32,
            model_path="",
            block_type="blade",
            device="cpu",
            pretrain_steps=0,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            chunk_size=16,
            state_dim=16,
            subsequence_len=3,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert 32 in results

        # Check metrics structure
        metrics = results[32]
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

    @pytest.mark.integration
    def test_copy_recall_task_longformer_integration(self):
        """Test copy/recall task integration with Longformer."""
        from efficient_longctx.evals.synthetic import run_copy_recall_task

        results = run_copy_recall_task(
            seq_len=32,
            max_len=32,
            model_path="",
            block_type="baseline_longformer",
            device="cpu",
            pretrain_steps=0,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            window_size=16,
            n_global_tokens=2,
            subsequence_len=3,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert 32 in results

        # Check metrics structure
        metrics = results[32]
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

    @pytest.mark.integration
    def test_copy_recall_task_bigbird_integration(self):
        """Test copy/recall task integration with BigBird."""
        from efficient_longctx.evals.synthetic import run_copy_recall_task

        results = run_copy_recall_task(
            seq_len=32,
            max_len=32,
            model_path="",
            block_type="baseline_bigbird",
            device="cpu",
            pretrain_steps=0,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            window_size=16,
            n_global_tokens=2,
            n_random_tokens=4,
            subsequence_len=3,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert 32 in results

        # Check metrics structure
        metrics = results[32]
        assert "accuracy" in metrics
        assert "total_correct" in metrics
        assert "total_tokens" in metrics

    @pytest.mark.integration
    def test_drift_curve_task_integration(self):
        """Test drift curve task integration."""
        from efficient_longctx.evals.synthetic import run_drift_curve_task

        results = run_drift_curve_task(
            seq_len=32,
            max_len=64,
            model_path="",
            block_type="dpassm",
            device="cpu",
            pretrain_steps=0,
            pretrained_model=None,
            d_model=64,
            n_heads=4,
            n_layers=1,
            window_size=16,
            ssm_state_dim=16,
        )

        # Check results structure
        assert isinstance(results, dict)
        assert len(results) > 1  # Should have multiple lengths

        # Check metrics structure for each length
        for _length, metrics in results.items():
            assert "accuracy" in metrics
            assert "total_correct" in metrics
            assert "total_tokens" in metrics
