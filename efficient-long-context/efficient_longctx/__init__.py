"""Efficient long-context blocks for training and benchmarking."""

__version__ = "0.1.0"
__author__ = "Maida.AI Research Team"
__email__ = "research@maida.ai"

# Import submodules
from efficient_longctx import blocks, evals, models, training, utils  # noqa: F401

# Import main model classes for easy access
from efficient_longctx.models import (  # noqa: F401
    LongCtxModel,
    VanillaAttentionBlock,
    create_model,
    load_model_from_checkpoint,
)
