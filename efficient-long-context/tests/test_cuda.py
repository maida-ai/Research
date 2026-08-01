"""Test CUDA availability."""

import torch


def test_cuda_available():
    """Test that CUDA GPU is available."""
    assert torch.cuda.is_available(), "CUDA GPU not detected"


def test_cuda_device_count():
    """Test that at least one CUDA device is available."""
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0, "No CUDA devices found"
        print(f"Found {torch.cuda.device_count()} CUDA device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available - skipping device count test")
