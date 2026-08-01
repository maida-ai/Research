import time

import torch

from efficient_longctx.blocks import DPASSMBlock


def _make_models(
    params: dict[str, int], ring_mode: str = "ring_win"
) -> tuple[DPASSMBlock, DPASSMBlock]:
    torch.manual_seed(0)
    model_cat = DPASSMBlock(**params)
    torch.manual_seed(0)
    model_ring = DPASSMBlock(**{**params, "kv_cache_mode": ring_mode})
    # Ensure identical weights
    model_ring.load_state_dict(model_cat.state_dict())
    model_cat.eval()
    model_ring.eval()
    return model_cat, model_ring


def test_streaming_equivalence_ring_vs_cat() -> None:
    params: dict[str, int] = {
        "d_model": 64,
        "n_heads": 4,
        "window_size": 8,
        "ssm_state_dim": 16,
        "dropout": 0.0,
    }
    B, T = 2, 32
    x = torch.randn(B, T, params["d_model"])  # deterministic by default seed

    model_cat, model_ring = _make_models(params)

    with torch.no_grad():
        # Full pass reference (shared)
        y_full, s_full = model_cat(x)

        # Streaming CAT
        s_cat = None
        a_cat = None
        outs_cat = []
        for t in range(T):
            y_t, s_cat, a_cat = model_cat.forward_step(x[:, t : t + 1, :], s_cat, a_cat)
            outs_cat.append(y_t)
        y_cat = torch.cat(outs_cat, dim=1)

        # Streaming RING
        s_ring = None
        a_ring = None
        outs_ring = []
        for t in range(T):
            y_t, s_ring, a_ring = model_ring.forward_step(
                x[:, t : t + 1, :], s_ring, a_ring
            )
            outs_ring.append(y_t)
        y_ring = torch.cat(outs_ring, dim=1)

    # Compare with a modest tolerance (layernorm, dropout off)
    assert torch.allclose(y_full, y_cat, atol=1e-5, rtol=1e-4)
    assert torch.allclose(y_full, y_ring, atol=1e-5, rtol=1e-4)
    assert torch.allclose(s_full, s_cat, atol=1e-5, rtol=1e-4)
    assert torch.allclose(s_full, s_ring, atol=1e-5, rtol=1e-4)


def test_ring_buffer_capacity_stable() -> None:
    params: dict[str, int] = {
        "d_model": 32,
        "n_heads": 4,
        "window_size": 8,
        "ssm_state_dim": 8,
        "dropout": 0.0,
    }
    B, T = 2, 8 * 6  # multiple wraps
    x = torch.randn(B, T, params["d_model"])  # deterministic by default seed

    _, model_ring = _make_models(params)

    with torch.no_grad():
        s = None
        a = None
        for t in range(T):
            _, s, a = model_ring.forward_step(x[:, t : t + 1, :], s, a)

    # attn_state is (K_buf, V_buf, K_win, V_win, idx, filled)
    assert isinstance(a, tuple) and len(a) == 6
    Kb, Vb, Kw, Vw, idx, filled = a
    assert Kb.shape[2] == params["window_size"]
    assert Vb.shape[2] == params["window_size"]
    assert Kw.shape[2] == params["window_size"]
    assert Vw.shape[2] == params["window_size"]
    assert 0 <= idx < params["window_size"]
    assert 1 <= filled <= params["window_size"]


def test_ring2x_equivalence() -> None:
    """Test that ring2x mode produces equivalent results to cat mode."""
    params: dict[str, int] = {
        "d_model": 64,
        "n_heads": 4,
        "window_size": 8,
        "ssm_state_dim": 16,
        "dropout": 0.0,
    }
    B, T = 2, 32
    x = torch.randn(B, T, params["d_model"])

    model_cat, model_ring2x = _make_models(params, "ring2x")

    with torch.no_grad():
        # Full pass reference
        y_full, s_full = model_cat(x)

        # Streaming CAT
        s_cat = None
        a_cat = None
        outs_cat = []
        for t in range(T):
            y_t, s_cat, a_cat = model_cat.forward_step(x[:, t : t + 1, :], s_cat, a_cat)
            outs_cat.append(y_t)
        y_cat = torch.cat(outs_cat, dim=1)

        # Streaming RING2X
        s_ring2x = None
        a_ring2x = None
        outs_ring2x = []
        for t in range(T):
            y_t, s_ring2x, a_ring2x = model_ring2x.forward_step(
                x[:, t : t + 1, :], s_ring2x, a_ring2x
            )
            outs_ring2x.append(y_t)
        y_ring2x = torch.cat(outs_ring2x, dim=1)

    # Compare with modest tolerance
    assert torch.allclose(y_full, y_cat, atol=1e-5, rtol=1e-4)
    assert torch.allclose(y_full, y_ring2x, atol=1e-5, rtol=1e-4)
    assert torch.allclose(s_full, s_cat, atol=1e-5, rtol=1e-4)
    assert torch.allclose(s_full, s_ring2x, atol=1e-5, rtol=1e-4)


def test_ring2x_buffer_structure() -> None:
    """Test that ring2x mode maintains correct buffer structure."""
    params: dict[str, int] = {
        "d_model": 32,
        "n_heads": 4,
        "window_size": 8,
        "ssm_state_dim": 8,
        "dropout": 0.0,
    }
    B, T = 2, 8 * 6  # multiple wraps
    x = torch.randn(B, T, params["d_model"])

    _, model_ring2x = _make_models(params, "ring2x")

    with torch.no_grad():
        s = None
        a = None
        for t in range(T):
            _, s, a = model_ring2x.forward_step(x[:, t : t + 1, :], s, a)

    # attn_state is (K_buf, V_buf, wp, filled)
    assert isinstance(a, tuple) and len(a) == 4
    Kb, Vb, wp, filled = a
    assert Kb.shape[2] == 2 * params["window_size"]  # 2W buffer
    assert Vb.shape[2] == 2 * params["window_size"]  # 2W buffer
    assert 0 <= wp < 2 * params["window_size"]
    assert 1 <= filled <= params["window_size"]


def tiny_tokens_per_sec_benchmark() -> tuple[float, float]:
    params: dict[str, int] = {
        "d_model": 128,
        "n_heads": 8,
        "window_size": 64,
        "ssm_state_dim": 32,
        "dropout": 0.0,
    }
    B, T = 1, 1024
    x = torch.randn(B, T, params["d_model"])  # CPU baseline
    model_cat, model_ring = _make_models(params)

    def bench(model: DPASSMBlock) -> float:
        s = None
        a = None
        t0 = time.time()
        with torch.no_grad():
            for i in range(T):
                _, s, a = model.forward_step(x[:, i : i + 1, :], s, a)
        t1 = time.time()
        return T / (t1 - t0)

    cat_tps = bench(model_cat)
    ring_tps = bench(model_ring)
    return cat_tps, ring_tps


def gpu_benchmark() -> dict[str, float]:
    """GPU benchmark with realistic sizes for streaming inference."""
    if not torch.cuda.is_available():
        return {"cat_tps": 0.0, "ring_tps": 0.0, "device": "cpu"}

    device = torch.device("cuda")
    params: dict[str, int] = {
        "d_model": 256,
        "n_heads": 16,
        "window_size": 128,
        "ssm_state_dim": 64,
        "dropout": 0.0,
    }
    B, T = 4, 2048
    x = torch.randn(B, T, params["d_model"], device=device)

    torch.manual_seed(0)
    model_cat = DPASSMBlock(**params).to(device).eval()
    torch.manual_seed(0)
    model_ring = (
        DPASSMBlock(**{**params, "kv_cache_mode": "ring_win"}).to(device).eval()
    )
    model_ring.load_state_dict(model_cat.state_dict())

    @torch.inference_mode()
    def bench(model: DPASSMBlock, warmup: int = 512, steps: int = 2048) -> float:
        s = a = None
        # Warmup
        for i in range(warmup):
            _, s, a = model.forward_step(x[:, i : i + 1, :], s, a)

        torch.cuda.synchronize()
        t0 = time.time()
        s = a = None
        for i in range(steps):
            _, s, a = model.forward_step(x[:, i : i + 1, :], s, a)
        torch.cuda.synchronize()
        t1 = time.time()
        return steps / (t1 - t0)

    cat_tps = bench(model_cat)
    ring_tps = bench(model_ring)
    return {"cat_tps": cat_tps, "ring_tps": ring_tps, "device": "cuda"}


def gpu_benchmark_ring2x() -> dict[str, float]:
    """GPU benchmark comparing cat vs ring2x modes."""
    if not torch.cuda.is_available():
        return {"cat_tps": 0.0, "ring2x_tps": 0.0, "device": "cpu"}

    device = torch.device("cuda")
    params: dict[str, int] = {
        "d_model": 256,
        "n_heads": 16,
        "window_size": 128,
        "ssm_state_dim": 64,
        "dropout": 0.0,
    }
    B, T = 4, 2048
    x = torch.randn(B, T, params["d_model"], device=device)

    torch.manual_seed(0)
    model_cat = DPASSMBlock(**params).to(device).eval()
    torch.manual_seed(0)
    model_ring2x = (
        DPASSMBlock(**{**params, "kv_cache_mode": "ring2x"}).to(device).eval()
    )
    model_ring2x.load_state_dict(model_cat.state_dict())

    @torch.inference_mode()
    def bench(model: DPASSMBlock, warmup: int = 512, steps: int = 2048) -> float:
        s = a = None
        # Warmup
        for i in range(warmup):
            _, s, a = model.forward_step(x[:, i : i + 1, :], s, a)

        torch.cuda.synchronize()
        t0 = time.time()
        s = a = None
        for i in range(steps):
            _, s, a = model.forward_step(x[:, i : i + 1, :], s, a)
        torch.cuda.synchronize()
        t1 = time.time()
        return steps / (t1 - t0)

    cat_tps = bench(model_cat)
    ring2x_tps = bench(model_ring2x)
    return {"cat_tps": cat_tps, "ring2x_tps": ring2x_tps, "device": "cuda"}
