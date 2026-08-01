"""Model definitions for long-context attention blocks.

This module contains the core model classes that can be used independently
of training or inference scripts.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

import efficient_longctx
from efficient_longctx.training.data import setup_tokenizer
from efficient_longctx.utils.gqa_kernels import gqa_attention


def get_config_params(num_params: int | str | None = None) -> dict[str, Any]:
    configs = {
        "150m": {"d_model": 768, "n_layers": 12, "n_heads": 12},
        "250m": {"d_model": 1024, "n_layers": 16, "n_heads": 16},
        "350m": {"d_model": 1280, "n_layers": 20, "n_heads": 20},
    }

    if num_params is None:
        return configs

    if isinstance(num_params, int):
        num_params = f"{num_params}m"

    if num_params not in configs:
        raise ValueError(f"Unknown parameter count: {num_params}")

    return configs[num_params]


class VanillaAttentionBlock(nn.Module):
    """Simple causal attention block for baseline comparison."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        num_kv_heads: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        # GQA setup: default to n_heads for backward compatibility
        if num_kv_heads is None:
            num_kv_heads = n_heads

        # Validate GQA parameters
        if num_kv_heads > n_heads:
            raise ValueError("num_kv_heads cannot be greater than n_heads")
        if n_heads % num_kv_heads != 0:
            raise ValueError("n_heads must be divisible by num_kv_heads for GQA")

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # GQA: K/V projections use num_kv_heads instead of n_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with causal attention."""
        B, T, D = x.shape

        # Pre-LN
        h = self.ln1(x)

        # Attention with proper GQA implementation
        q = (
            self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hq, T, Dh)
        k = (
            self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, Dh)
        v = (
            self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, Dh)

        # Use GQA-aware attention computation (no materialization)
        dropout_p = self.dropout.p if self.training else 0.0
        attn_out = gqa_attention(
            Q=q,
            K=k,
            V=v,
            dropout_p=dropout_p,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = x + self.dropout(attn_out)

        # FFN
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x


class LongCtxModel(nn.Module):
    """GPT-style model with configurable attention blocks.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        block_type: Type of attention block ('dpassm', 'blade', 'vanilla', 'baseline_longformer', 'baseline_bigbird')
        block_kwargs: Additional arguments for the attention block
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        block_type: str,
        block_kwargs: dict[str, Any],
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_type = block_type
        self.max_seq_len = max_seq_len
        self.block_kwargs = block_kwargs

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer_cls = get_layer(block_type)
            layer = layer_cls(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                **block_kwargs,
            )

            self.layers.append(layer)

        # Layer normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, T, D]

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        # Pass through transformer layers
        state = None
        for layer in self.layers:
            if self.block_type in [
                "dpassm",
                "blade",
                "baseline_longformer",
                "baseline_bigbird",
            ]:
                x, state = layer(x, state)
            else:
                x = layer(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "block_type": self.block_type,
            "max_seq_len": self.max_seq_len,
            "block_kwargs": self.block_kwargs,
        }


def get_layer(block_type: str | None = None) -> nn.Module:
    layers = {
        "dpassm": efficient_longctx.blocks.DPASSMBlock,
        "blade": efficient_longctx.blocks.BLADEBlock,
        "vanilla": VanillaAttentionBlock,
        "baseline_longformer": efficient_longctx.blocks.LongformerBlock,
        "baseline_bigbird": efficient_longctx.blocks.BigBirdBlock,
    }

    if block_type is None:
        return layers

    if block_type not in layers:
        raise ValueError(f"Unknown block type: {block_type}")

    return layers[block_type]


def create_model(
    *,  # Force keyword-only arguments
    vocab_size: int,
    num_params: int | str,
    block_type: str,
    block_kwargs: dict[str, Any],
    **kwargs,  # All other args will be passed to the mdoel directly
) -> LongCtxModel:
    """Create a model with specified parameter count.

    Args:
        vocab_size: Vocabulary size
        num_params: Parameter count ('150m', '250m', '350m')
        block_type: Type of attention block
        block_kwargs: Additional arguments for the attention block

    Returns:
        Configured model
    """
    # Model configurations for different parameter counts
    config = get_config_params(num_params)
    config.update(kwargs)  # Override with kwargs

    model = LongCtxModel(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        block_type=block_type,
        block_kwargs=block_kwargs,
        **kwargs,
    )

    return model


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> LongCtxModel:
    """Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Create model with the saved configuration
    model = LongCtxModel(
        **model_config,
    )

    model.to(device)

    return model


class LongCtxLightningModule(LightningModule):
    """Lightning module for long-context model training."""

    def __init__(
        self,
        *,  # Force keyword-only arguments
        # Model configuration
        num_params: int | str = "150m",
        block: str = "dpassm",
        window_size: int = 2048,
        ssm_state_dim: int = 256,
        chunk_size: int = 512,
        state_dim: int = 128,
        n_global_tokens: int = 2,
        n_random_tokens: int = 4,
        # Training configuration
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        # Tokenizer configuration
        tokenizer_name: str = "gpt2",
    ) -> None:
        """Initialize Lightning module.

        Args:
            num_params: Model size ('150m', '250m', '350m')
            block: Attention block type ('dpassm', 'blade', 'vanilla', 'baseline_longformer', 'baseline_bigbird')
            window_size: Window size for DP-ASSM and baseline blocks
            ssm_state_dim: SSM state dimension for DP-ASSM
            chunk_size: Chunk size for BLADE
            state_dim: State dimension for BLADE
            n_global_tokens: Number of global tokens for baseline blocks
            n_random_tokens: Number of random tokens for BigBird
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            max_steps: Maximum number of training steps
            tokenizer_name: Hugging Face model name for tokenizer

        Note:
            For testing or custom use cases where you want to provide a pre-created model,
            use the `set_model()` method after initialization instead of passing it to __init__.
        """
        super().__init__()
        self.save_hyperparameters()

        # Store configuration
        self.num_params = num_params
        self.block = block
        self.window_size = window_size
        self.ssm_state_dim = ssm_state_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.n_global_tokens = n_global_tokens
        self.n_random_tokens = n_random_tokens
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.tokenizer_name = tokenizer_name

        # Initialize model (will be created in setup or set via set_model)
        self.model: LongCtxModel | None = None

    def set_model(self, model: LongCtxModel) -> None:
        """Set a pre-created model (alternative to automatic model creation).

        This is useful for testing or custom use cases where you want to provide
        a pre-configured model instance.

        Args:
            model: Pre-created LongCtxModel instance
        """
        self.model = model

    def setup(self, stage: str) -> None:
        """Set up the model.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        # Only create model if not provided in __init__
        if self.model is None:
            # Set up tokenizer
            tokenizer = setup_tokenizer(self.tokenizer_name)
            vocab_size = len(tokenizer)

            # Set up block-specific arguments
            if self.block == "dpassm":
                block_kwargs = {
                    "window_size": self.window_size,
                    "ssm_state_dim": self.ssm_state_dim,
                }
            elif self.block == "blade":
                block_kwargs = {
                    "chunk_size": self.chunk_size,
                    "state_dim": self.state_dim,
                }
            elif self.block == "baseline_longformer":
                block_kwargs = {
                    "window_size": self.window_size,
                    "n_global_tokens": self.n_global_tokens,
                }
            elif self.block == "baseline_bigbird":
                block_kwargs = {
                    "window_size": self.window_size,
                    "n_random_tokens": self.n_random_tokens,
                    "n_global_tokens": self.n_global_tokens,
                }
            else:  # vanilla
                block_kwargs = {}

            # Create model
            self.model = create_model(
                vocab_size=vocab_size,
                num_params=self.num_params,
                block_type=self.block,
                block_kwargs=block_kwargs,
            )
            logging.info(
                f"Created {self.num_params} model with {self.model.get_num_params():,} parameters"
            )

    def forward(
        self, input_ids: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs

        Returns:
            Logits for next token prediction

        Raises:
            RuntimeError: If model hasn't been set up
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Make sure setup() was called or model was provided in __init__"
            )
        return self.model(input_ids, *args, **kwargs)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (input_ids, labels)
            batch_idx: Batch index

        Returns:
            Training loss
        """
        input_ids, labels = batch

        # Forward pass
        logits = self(input_ids, *args, **kwargs)

        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/perplexity", math.exp(loss.detach()), on_step=True, on_epoch=True
        )

        # Calculate throughput
        tokens_per_step = input_ids.numel()
        self.log("train/tokens_per_step", tokens_per_step, on_step=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (input_ids, labels)
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        input_ids, labels = batch

        # Forward pass
        logits = self(input_ids, *args, **kwargs)

        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/perplexity", math.exp(loss.detach()), on_step=False, on_epoch=True
        )

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Linear warmup scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return self.model.get_model_config() if self.model else {}

    def get_num_params(self) -> int:
        """Get number of parameters."""
        return self.model.get_num_params() if self.model else 0
