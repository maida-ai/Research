#!/usr/bin/env python3
"""Example script showing how to use the Lightning-based trainer.

This script demonstrates how to train a small model using PyTorch Lightning
with the efficient long-context blocks.
"""

import argparse
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from efficient_longctx.models import create_model
from efficient_longctx.training.data import setup_tokenizer
from efficient_longctx.training.train import (
    LongCtxDataModule,
    LongCtxLightningModule,
    create_callbacks,
)


def main() -> None:
    """Example training script."""
    parser = argparse.ArgumentParser(description="Lightning training example")
    parser.add_argument(
        "--out_dir", default="./lightning_output", help="Output directory"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100000, help="Max tokens for quick test"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=1, help="Max epochs")
    parser.add_argument(
        "--num_params",
        default="150m",
        choices=["150m", "250m", "350m"],
        help="Model size",
    )
    parser.add_argument(
        "--block",
        default="vanilla",
        choices=["dpassm", "blade", "vanilla"],
        help="Block type",
    )

    args = parser.parse_args()

    print("🚀 Starting Lightning training example...")

    # Set up tokenizer
    tokenizer = setup_tokenizer()
    vocab_size = len(tokenizer)
    print(f"📝 Tokenizer loaded with vocab size: {vocab_size:,}")

    # Create model
    model = create_model(
        vocab_size=vocab_size,
        num_params=args.num_params,
        block_type=args.block,
        block_kwargs={},
    )
    print(f"🧠 Model created with {model.get_num_params():,} parameters")

    # Create Lightning module
    lightning_module = LongCtxLightningModule(model, learning_rate=1e-3)
    print("⚡ Lightning module created")

    # Create data module
    data_module = LongCtxDataModule(
        dataset_name="openwebtext",
        max_tokens=args.max_tokens,
        seq_len=512,  # Shorter for quick test
        batch_size=args.batch_size,
        val_split=0.1,
    )
    print("📊 Data module created")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create callbacks
    callbacks = create_callbacks(out_dir)
    print(f"🔧 Created {len(callbacks)} callbacks")

    # Create logger
    logger = TensorBoardLogger(save_dir=out_dir, name="example_run")

    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=False,  # Set to True for a single batch test
    )
    print("🏃 Trainer created")

    # Start training
    print("🎯 Starting training...")
    trainer.fit(lightning_module, data_module)

    # Run validation
    print("✅ Running validation...")
    trainer.validate(lightning_module, data_module)

    print("🎉 Training completed!")
    print(f"📁 Results saved to: {out_dir}")
    print(f"📈 TensorBoard logs: {out_dir / 'lightning_logs'}")


if __name__ == "__main__":
    main()
