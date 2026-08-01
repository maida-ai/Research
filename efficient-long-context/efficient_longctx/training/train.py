"""PyTorch Lightning trainer for small models (150-350M params).

Train tiny GPT-style models using DP-ASSM or BLADE blocks with LightningCLI.
This is the main training script that uses LightningCLI for automatic argument
parsing, configuration files, and subcommands.
"""

import logging

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.cli import LightningCLI

from efficient_longctx.models.data import LongCtxDataModule
from efficient_longctx.models.models import LongCtxLightningModule

torch.set_float32_matmul_precision("medium")


class MetricsCallback(Callback):
    """Custom callback for additional metrics logging."""

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LongCtxLightningModule
    ) -> None:
        """Log additional metrics at epoch end."""
        # Log model parameters
        trainer.logger.log_metrics(
            {"model/num_params": pl_module.get_num_params()},
            step=trainer.global_step,
        )


class TrainingCLI(LightningCLI):
    """CLI for long-context model training."""

    def add_arguments_to_parser(self, parser) -> None:
        """Add custom arguments to the parser."""
        # Add custom arguments if needed
        pass

    def before_instantiate_classes(self) -> None:
        """Set up logging and seed before instantiating classes."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Set random seed if not already set
        if hasattr(self.config, "seed_everything"):
            seed_everything(self.config.seed_everything, workers=True)

    def after_instantiate_classes(self) -> None:
        """Add custom callbacks after instantiating classes."""
        # Add custom callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
            MetricsCallback(),
        ]

        # Add callbacks to trainer
        self.trainer.callbacks.extend(callbacks)


def main() -> None:
    """Main CLI entry point."""
    TrainingCLI(
        model_class=LongCtxLightningModule,
        datamodule_class=LongCtxDataModule,
        save_config_callback=None,  # We'll handle config saving manually
        seed_everything_default=42,
        parser_kwargs={"default_env": True},  # Enable environment variables
    )


if __name__ == "__main__":
    main()
