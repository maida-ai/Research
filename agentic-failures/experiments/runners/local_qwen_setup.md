# Local Qwen2.5-7B-Instruct Setup — RTX 4090 (24 GB VRAM)

Instructions only; not executed in this build phase.
After setup, record model hash and server version in governance.json.

---

## Hardware target

RTX 4090 (24 GB VRAM). Qwen2.5-7B-Instruct in BF16 requires ~15 GB VRAM.
With max-model-len tuned down, stays comfortably within 24 GB.

---

## Option A — vLLM (recommended for throughput)

### 1. Install vLLM

```bash
pip install vllm==0.6.3   # version-pin; record in governance.json
```

### 2. Download model and record hash

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="models/Qwen2.5-7B-Instruct")
print(path)
EOF

# Record SHA256 of config.json + all .safetensors shards
sha256sum models/Qwen2.5-7B-Instruct/*.safetensors | tee models/Qwen2.5-7B-Instruct/SHA256SUMS
```

### 3. Serve (OpenAI-compatible)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

VRAM notes:
- BF16 7B ~ 14 GB weights. KV cache at max-model-len=8192 adds ~5 GB.
- Total ~ 19 GB. Fits in 24 GB with --gpu-memory-utilization 0.90.
- If OOM, lower --max-model-len to 4096 (~ 17 GB total).
- Do NOT use --max-model-len > 8192 without testing VRAM first.

### 4. Verify server is up

```bash
curl http://localhost:8000/v1/models
```

Expected response: JSON listing `Qwen/Qwen2.5-7B-Instruct`.

### 5. Test with gradient_check harness

```bash
.venv/bin/python -m experiments.runners.gradient_check --live --seed 42
```

---

## Option B — llama.cpp with GGUF Q5_K_M (fallback if vLLM unavailable)

### 1. Build llama.cpp with CUDA

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Record llama.cpp git commit hash in governance.json.

### 2. Download GGUF

```bash
pip install huggingface_hub
huggingface-cli download \
  Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q5_k_m.gguf \
  --local-dir models/
sha256sum models/qwen2.5-7b-instruct-q5_k_m.gguf | tee models/qwen2.5-7b-instruct-q5_k_m.sha256
```

Q5_K_M ~5.5 GB: fits easily within 24 GB VRAM.

### 3. Serve (OpenAI-compatible)

```bash
./build/bin/llama-server \
  --model models/qwen2.5-7b-instruct-q5_k_m.gguf \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --port 8000
```

### 4. Update LocalQwenConfig

```python
from experiments.runners.models import LocalQwenConfig
config = LocalQwenConfig(
    model_path_or_name="qwen2.5-7b-instruct-q5_k_m",
    model_hash="<sha256 from above>",
    endpoint_url="http://localhost:8000/v1",
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
    server_version="llama.cpp git <commit>",
)
```

---

## governance.json — what to fill in after pinning

```json
{
  "reference_agent_model": "Qwen2.5-7B-Instruct",
  "model_hash": "<sha256 of weights>",
  "server": "vllm==0.6.3",
  "server_version": "<vllm version>",
  "endpoint_url": "http://localhost:8000/v1",
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 2048,
  "max_model_len": 8192,
  "scaffold": "experiments/runners/scaffold.py BaselineAgent",
  "terminal_protocol_hash": "<sha256 of TERMINAL_PROTOCOL string>",
  "decoding_params": {"temperature": 0.0, "top_p": 1.0},
  "retry_policy": "none (first-shot success)",
  "tool_environment_version": "experiments/exp1_pipeline_proof/domain.py v1",
  "pinned_date": "YYYY-MM-DD",
  "pinned_by": "gradient_check.py --live",
  "gradient_check_verdict": "PASS",
  "gradient_check_span": 0.0,
  "seeds": [42]
}
```

DO NOT commit governance.json until the gradient check passes with the live model.

---

## Decoding parameters for reproducibility

All reference-agent runs must use:
- temperature = 0.0
- top_p = 1.0
- No sampling: greedy decoding

This ensures full reproducibility of the frozen reference agent.
