import torch


# Check if FlashAttention is available
def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


FLASH_ATTN_AVAILABLE = has_flash_attn()


# CUDA available
def has_cuda() -> bool:
    return torch.cuda.is_available()


CUDA_AVAILABLE = has_cuda()
