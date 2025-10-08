#!/usr/bin/env python3
"""Example showing how to use the LightningCLI-based trainer.

This demonstrates the benefits of using LightningCLI over manual argument parsing.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and show the output."""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except FileNotFoundError:
        print("Command not found. Make sure you're in the correct directory.")


def main() -> None:
    """Demonstrate LightningCLI usage."""
    print("🚀 LightningCLI Training Examples")
    print("=" * 60)

    # Get the path to the CLI script
    cli_script = (
        Path(__file__).parent.parent / "efficient_longctx" / "training" / "train.py"
    )

    if not cli_script.exists():
        print(f"❌ CLI script not found at {cli_script}")
        print("Make sure you've created the train.py file first.")
        return

    # Example 1: Show help
    run_command(
        [sys.executable, str(cli_script), "--help"],
        "Show CLI help (automatically generated from LightningModule and DataModule)",
    )

    # Example 2: Show model help
    run_command(
        [
            sys.executable,
            str(cli_script),
            "fit",
            "--model.help",
            "LongCtxLightningModule",
        ],
        "Show model-specific arguments (automatically generated)",
    )

    # Example 3: Show data help
    run_command(
        [sys.executable, str(cli_script), "fit", "--data.help", "LongCtxDataModule"],
        "Show data-specific arguments (automatically generated)",
    )

    # Example 4: Show trainer help
    run_command(
        [sys.executable, str(cli_script), "fit", "--trainer.help"],
        "Show trainer-specific arguments (automatically generated)",
    )

    print(f"\n{'=' * 60}")
    print("✅ LightningCLI Benefits Demonstrated:")
    print("=" * 60)
    print("1. 🎯 Automatic argument parsing from LightningModule and DataModule")
    print("2. 📁 Built-in support for configuration files (--config config.yaml)")
    print("3. 🔧 Automatic subcommands (fit, validate, test, predict)")
    print("4. 🌍 Environment variable support (PL_TRAINER__MAX_EPOCHS=10)")
    print("5. 💾 Built-in checkpoint loading support")
    print("6. 📝 Automatic help generation")
    print("7. 🔗 Argument linking between components")
    print("8. 🎨 Much cleaner and more maintainable code")

    print(f"\n{'=' * 60}")
    print("📋 Example Usage Commands:")
    print("=" * 60)
    print("# Basic training")
    print("python efficient_longctx/training/train.py fit \\")
    print("    --model.params=150m \\")
    print("    --model.block=dpassm \\")
    print("    --data.dataset=openwebtext \\")
    print("    --data.max_tokens=100000 \\")
    print("    --trainer.max_epochs=1")
    print()
    print("# Using configuration file")
    print("python efficient_longctx/training/train.py fit --config config.yaml")
    print()
    print("# Validation only")
    print(
        "python efficient_longctx/training/train.py validate --ckpt_path checkpoint.ckpt"
    )
    print()
    print("# Using environment variables")
    print("PL_TRAINER__MAX_EPOCHS=5 PL_MODEL__LEARNING_RATE=1e-3 \\")
    print("python efficient_longctx/training/train.py fit")


if __name__ == "__main__":
    main()
