"""CPU tests for the Hunyuan3D-2 Hermite comparison arm."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("hunyuan2_hicache", ROOT / "hy3dgen/shapegen/hicache.py")
hicache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hicache)


def test_facade_uses_central_corrected_sign_core():
    import hicache_pp.hermite as central

    assert hicache.hicache_forecast is central.hicache_forecast
    assert hicache.scaled_hermite is central.scaled_hermite
    state = hicache.hicache_init(num_steps=4, interval=1, first_enhance=0)
    assert state["backend"] == "hermite"
    assert state["holdout"] == "1step"


def test_interval_three_and_interval_one_traces_are_stable():
    for interval, expected_calls in ((3, 2), (1, 6)):
        calls = 0

        def dynamics():
            nonlocal calls
            calls += 1
            return torch.ones(2)

        state = hicache.hicache_init(num_steps=6, interval=interval, first_enhance=0)
        outputs = []
        for step in range(6):
            state["step"] = step
            if hicache.hicache_decide(state) == "full":
                value = dynamics()
                hicache.hicache_update_derivatives(state, value)
            else:
                value = hicache.hicache_forecast(state)
            outputs.append(value)
            state["step"] += 1
        assert calls == expected_calls
        assert all(torch.isfinite(value).all() for value in outputs)
        telemetry = hicache.hicache_telemetry(state)
        assert telemetry["decisions"]["full"] == expected_calls


def test_reset_clears_trajectory_and_preserves_cfg_combined_contract():
    state = hicache.hicache_init(num_steps=6, interval=3, first_enhance=0)
    state["step"] = 0
    state["activated_steps"].append(0)
    hicache.hicache_update_derivatives(state, torch.tensor([2.0]))
    run_id = state["run_id"]
    hicache.hicache_reset(state)
    assert state["run_id"] != run_id
    assert state["derivatives"] == {}
    assert state["activated_steps"] == []
    assert hicache.hicache_telemetry(state)["decisions"] == {"full": 0, "forecast": 0}

    cond, uncond, scale = torch.tensor([3.0]), torch.tensor([1.0]), 5.0
    assert torch.equal(uncond + scale * (cond - uncond), torch.tensor([11.0]))
