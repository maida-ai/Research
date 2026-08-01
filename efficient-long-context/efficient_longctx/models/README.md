# Models Directory

This directory contains model definitions and implementations for the efficient long-context research project.

## Structure

- `models.py` - Main model definitions including:
  - `LongCtxModel` - GPT-style model with configurable attention blocks
  - `VanillaAttentionBlock` - Simple causal attention baseline
  - `create_model()` - Factory function for different model sizes
  - `load_model_from_checkpoint()` - Utility for loading saved models

## Model Versions

This directory is designed to track different versions and evolution of models:

- **Current**: `models.py` - Initial implementation with DP-ASSM, BLADE, and vanilla attention blocks
- **Future**: Additional model files can be added as `models_v2.py`, `models_experimental.py`, etc.

## Usage

Models can be imported directly from this directory:

```python
from models.models import LongCtxModel, create_model

# Create a model
model = create_model(1000, "150m", "dpassm", {"window_size": 32, "ssm_state_dim": 64})

# Use for inference
logits = model(input_ids)
```

## Benefits

- **Version Control**: Easy to track model evolution and compare different implementations
- **Independence**: Models are separate from training/inference code
- **Modularity**: Clear separation of concerns
- **Reusability**: Models can be used across different applications and experiments
