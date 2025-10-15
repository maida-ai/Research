"""Pytest tests for the benchmarking framework."""

import pytest
import torch
import torch.nn as nn
from benchmark.base import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from benchmark.blocks import create_custom_input_generators, create_standard_blocks
from benchmark.utils import get_system_info


class TestSystemInfo:
    """Test system information functionality."""

    def test_get_system_info(self) -> None:
        """Test that system info is retrieved correctly."""
        info = get_system_info()

        # Check required keys
        required_keys = [
            "python_version",
            "torch_version",
            "cuda_available",
            "cpu_count",
            "memory_total_gb",
        ]

        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

        # Check data types
        assert isinstance(info["python_version"], str)
        assert isinstance(info["torch_version"], str)
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["cpu_count"], int)
        assert isinstance(info["memory_total_gb"], float)

        # Check CUDA-specific info if available
        if info["cuda_available"]:
            assert "cuda_version" in info
            assert "cuda_device_count" in info
            assert "cuda_device_name" in info
            assert "cuda_memory_total_gb" in info


class TestBenchmarkConfig:
    """Test BenchmarkConfig functionality."""

    def test_default_config(self) -> None:
        """Test default configuration creation."""
        config = BenchmarkConfig()

        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.batch_size == 1
        assert config.sequence_lengths == [1024, 2048, 4096, 8192]
        assert config.warmup_runs == 3
        assert config.benchmark_runs == 10
        assert config.torch_benchmark is True
        assert config.cudnn_benchmark is True
        assert config.cudnn_deterministic is False
        assert config.profile_memory is True
        assert config.verbose is True
        assert config.save_results is True
        assert config.results_dir == "./reports/benchmark"

    def test_custom_config(self) -> None:
        """Test custom configuration creation."""
        config = BenchmarkConfig(
            d_model=256,
            n_heads=4,
            batch_size=2,
            sequence_lengths=[512, 1024],
            warmup_runs=1,
            benchmark_runs=3,
            device="cpu",
            torch_benchmark=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            profile_memory=False,
            verbose=False,
            save_results=False,
            results_dir="./test_results",
        )

        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.batch_size == 2
        assert config.sequence_lengths == [512, 1024]
        assert config.warmup_runs == 1
        assert config.benchmark_runs == 3
        assert config.device == "cpu"
        assert config.torch_benchmark is False
        assert config.cudnn_benchmark is False
        assert config.cudnn_deterministic is True
        assert config.profile_memory is False
        assert config.verbose is False
        assert config.save_results is False
        assert config.results_dir == "./test_results"

    def test_device_auto_detection(self) -> None:
        """Test automatic device detection."""
        # Test default behavior (auto-detection in field default)
        config = BenchmarkConfig()

        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"

        # Test explicit device setting
        config_cpu = BenchmarkConfig(device="cpu")
        assert config_cpu.device == "cpu"

        config_cuda = BenchmarkConfig(device="cuda")
        if torch.cuda.is_available():
            assert config_cuda.device == "cuda"
        else:
            # Should fallback to CPU if CUDA not available
            assert config_cuda.device == "cpu"


class TestBlockCreation:
    """Test block creation functionality."""

    @pytest.fixture
    def test_config(self) -> BenchmarkConfig:
        """Create a test configuration."""
        return BenchmarkConfig(
            d_model=256,
            n_heads=4,
            sequence_lengths=[512],
            warmup_runs=1,
            benchmark_runs=2,
            verbose=False,
        )

    def test_create_standard_blocks(self, test_config: BenchmarkConfig) -> None:
        """Test creation of standard blocks."""
        blocks = create_standard_blocks(test_config)

        # Check that blocks were created
        assert len(blocks) > 0, "No blocks were created"

        # Check that all blocks are nn.Module instances
        for block_name, block in blocks.items():
            assert isinstance(block, nn.Module), (
                f"Block {block_name} is not an nn.Module"
            )

        # Check specific blocks if available
        expected_blocks = ["BLADE", "DPASSM", "Longformer", "BigBird"]
        for block_name in expected_blocks:
            if block_name in blocks:
                assert isinstance(blocks[block_name], nn.Module)

    def test_create_custom_input_generators(self, test_config: BenchmarkConfig) -> None:
        """Test creation of custom input generators."""
        generators = create_custom_input_generators(test_config)

        # Check that generators were created
        assert len(generators) > 0, "No input generators were created"

        # Test a generator
        for block_name, generator in generators.items():
            assert callable(generator), f"Generator for {block_name} is not callable"

            # Test generator with sample inputs
            inputs = generator(256, 1, 256)
            assert isinstance(inputs, list), (
                f"Generator for {block_name} should return a list"
            )
            assert len(inputs) > 0, f"Generator for {block_name} returned empty list"

            # Check input tensor properties
            input_tensor = inputs[0]
            assert isinstance(input_tensor, torch.Tensor), (
                f"Generator for {block_name} should return tensors"
            )
            assert input_tensor.shape == (1, 256, 256), (
                f"Unexpected tensor shape: {input_tensor.shape}"
            )


class TestBenchmarkRunner:
    """Test BenchmarkRunner functionality."""

    @pytest.fixture
    def test_config(self) -> BenchmarkConfig:
        """Create a test configuration."""
        return BenchmarkConfig(
            d_model=256,
            n_heads=4,
            sequence_lengths=[256],  # Very short for testing
            warmup_runs=1,
            benchmark_runs=2,
            verbose=False,
            save_results=False,
        )

    @pytest.fixture
    def test_block(self) -> nn.Module:
        """Create a simple test block."""
        from efficient_longctx.blocks import BLADEBlock

        return BLADEBlock(
            d_model=256,
            n_heads=4,
            chunk_size=128,
            state_dim=32,
            m_global=0,
            dropout=0.1,
        )

    def test_benchmark_runner_creation(self, test_config: BenchmarkConfig) -> None:
        """Test BenchmarkRunner creation."""
        runner = BenchmarkRunner(test_config)

        assert runner.config == test_config
        assert runner.results == []

    def test_benchmark_single_block(
        self, test_config: BenchmarkConfig, test_block: nn.Module
    ) -> None:
        """Test benchmarking a single block."""
        runner = BenchmarkRunner(test_config)

        # Create input generator
        def input_generator(seq_len: int, batch_size: int, d_model: int):
            x = torch.randn(batch_size, seq_len, d_model, device=test_config.device)
            return [x]

        # Run benchmark
        results = runner.benchmark_block(test_block, "TEST_BLOCK", input_generator)

        # Check results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        result = results[0]
        assert isinstance(result, BenchmarkResult)
        assert result.block_name == "TEST_BLOCK"
        assert result.sequence_length == 256
        assert result.batch_size == test_config.batch_size
        assert result.forward_time_mean > 0
        assert result.forward_time_std >= 0
        assert result.forward_time_min > 0
        assert result.forward_time_max > 0
        assert result.throughput_tokens_per_sec > 0
        assert result.allocated_memory_mb >= 0

    def test_benchmark_multiple_blocks(self, test_config: BenchmarkConfig) -> None:
        """Test benchmarking multiple blocks."""
        from efficient_longctx.blocks import BLADEBlock, DPASSMBlock

        # Create test blocks
        blocks = {
            "BLADE": BLADEBlock(
                d_model=test_config.d_model,
                n_heads=test_config.n_heads,
                chunk_size=128,
                state_dim=32,
                m_global=0,
                dropout=0.1,
            ),
            "DPASSM": DPASSMBlock(
                d_model=test_config.d_model,
                n_heads=test_config.n_heads,
                window_size=64,
                ssm_state_dim=32,
                dropout=0.1,
            ),
        }

        runner = BenchmarkRunner(test_config)

        # Create input generator
        def input_generator(seq_len: int, batch_size: int, d_model: int):
            x = torch.randn(batch_size, seq_len, d_model, device=test_config.device)
            return [x]

        # Run benchmarks
        all_results = []
        for block_name, block in blocks.items():
            results = runner.benchmark_block(block, block_name, input_generator)
            all_results.extend(results)

        # Check results
        assert len(all_results) == 2, f"Expected 2 results, got {len(all_results)}"

        for result in all_results:
            assert isinstance(result, BenchmarkResult)
            assert result.block_name in ["BLADE", "DPASSM"]
            assert result.forward_time_mean > 0
            assert result.throughput_tokens_per_sec > 0


class TestBenchmarkResult:
    """Test BenchmarkResult functionality."""

    def test_benchmark_result_creation(self) -> None:
        """Test BenchmarkResult creation."""
        config = BenchmarkConfig()

        result = BenchmarkResult(
            block_name="TEST_BLOCK",
            sequence_length=1024,
            batch_size=1,
            forward_time_mean=0.1,
            forward_time_std=0.01,
            forward_time_min=0.09,
            forward_time_max=0.11,
            peak_memory_mb=100.0,
            allocated_memory_mb=80.0,
            reserved_memory_mb=90.0,
            throughput_tokens_per_sec=10240,
            memory_efficiency_mb_per_token=0.08,
            config=config,
        )

        assert result.block_name == "TEST_BLOCK"
        assert result.sequence_length == 1024
        assert result.batch_size == 1
        assert result.forward_time_mean == 0.1
        assert result.throughput_tokens_per_sec == 10240
        assert result.config == config

    def test_benchmark_result_to_dict(self) -> None:
        """Test BenchmarkResult serialization."""
        config = BenchmarkConfig()

        result = BenchmarkResult(
            block_name="TEST_BLOCK",
            sequence_length=1024,
            batch_size=1,
            forward_time_mean=0.1,
            forward_time_std=0.01,
            forward_time_min=0.09,
            forward_time_max=0.11,
            peak_memory_mb=100.0,
            allocated_memory_mb=80.0,
            reserved_memory_mb=90.0,
            throughput_tokens_per_sec=10240,
            memory_efficiency_mb_per_token=0.08,
            config=config,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["block_name"] == "TEST_BLOCK"
        assert result_dict["sequence_length"] == 1024
        assert result_dict["forward_time_mean"] == 0.1
        assert result_dict["throughput_tokens_per_sec"] == 10240
        assert "config" in result_dict


class TestIntegration:
    """Integration tests for the complete benchmarking workflow."""

    @pytest.mark.integration
    def test_complete_benchmark_workflow(self) -> None:
        """Test the complete benchmarking workflow."""
        # Create configuration
        config = BenchmarkConfig(
            d_model=256,
            n_heads=4,
            sequence_lengths=[256],
            warmup_runs=1,
            benchmark_runs=2,
            verbose=False,
            save_results=False,
        )

        # Create blocks
        blocks = create_standard_blocks(config)
        assert len(blocks) > 0, "No blocks created"

        # Create input generators
        generators = create_custom_input_generators(config)
        assert len(generators) > 0, "No generators created"

        # Run benchmark
        runner = BenchmarkRunner(config)

        # Test with first available block
        block_name = list(blocks.keys())[0]
        block = blocks[block_name]
        generator = generators[block_name]

        results = runner.benchmark_block(block, block_name, generator)

        # Verify results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        assert result.block_name == block_name
        assert result.forward_time_mean > 0
        assert result.throughput_tokens_per_sec > 0
        assert result.allocated_memory_mb >= 0

    @pytest.mark.integration
    def test_memory_profiling(self) -> None:
        """Test memory profiling functionality."""
        config = BenchmarkConfig(
            d_model=256,
            n_heads=4,
            sequence_lengths=[256],
            warmup_runs=1,
            benchmark_runs=2,
            profile_memory=True,
            verbose=False,
            save_results=False,
        )

        from efficient_longctx.blocks import BLADEBlock

        block = BLADEBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            chunk_size=128,
            state_dim=32,
            m_global=0,
            dropout=0.1,
        )

        runner = BenchmarkRunner(config)

        def input_generator(seq_len: int, batch_size: int, d_model: int):
            x = torch.randn(batch_size, seq_len, d_model, device=config.device)
            return [x]

        results = runner.benchmark_block(block, "MEMORY_TEST", input_generator)

        result = results[0]
        assert result.peak_memory_mb >= 0
        assert result.allocated_memory_mb >= 0
        assert result.reserved_memory_mb >= 0
        assert result.memory_efficiency_mb_per_token >= 0


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_benchmark_performance(self) -> None:
        """Test that benchmarks complete within reasonable time."""
        import time

        config = BenchmarkConfig(
            d_model=256,
            n_heads=4,
            sequence_lengths=[256],
            warmup_runs=1,
            benchmark_runs=3,
            verbose=False,
            save_results=False,
        )

        from efficient_longctx.blocks import BLADEBlock

        block = BLADEBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            chunk_size=128,
            state_dim=32,
            m_global=0,
            dropout=0.1,
        )

        runner = BenchmarkRunner(config)

        def input_generator(seq_len: int, batch_size: int, d_model: int):
            x = torch.randn(batch_size, seq_len, d_model, device=config.device)
            return [x]

        start_time = time.time()
        results = runner.benchmark_block(block, "PERF_TEST", input_generator)
        end_time = time.time()

        # Should complete within 30 seconds
        assert end_time - start_time < 30, "Benchmark took too long"
        assert len(results) == 1
        assert results[0].forward_time_mean > 0
