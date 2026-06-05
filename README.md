<div align="center">

# Hunyuan3D-2 mini + HiCache

**Tencent's 0.6B image-to-3D, accelerated by training-free *Hermite-polynomial* velocity forecasting.**

*[Hunyuan3D-2 mini](https://huggingface.co/tencent/Hunyuan3D-2mini) with [HiCache](https://arxiv.org/abs/2508.16984)
wired natively into its flow-matching denoise loop: skip the DiT on most sampling steps and forecast
the cached velocity instead — no retraining, no monkey-patching.*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![image&#8209;to&#8209;3D](https://img.shields.io/badge/image--to--3D-0.6B-2e6db0)
&nbsp;![basis](https://img.shields.io/badge/basis-Hermite%20polynomial-7a5cc6)

</div>

---

## What it is

This is a fork of **Hunyuan3D-2 mini** (Tencent's 0.6B shape generator, `Hunyuan3DDiTFlowMatchingPipeline`)
with **HiCache** built directly into the diffusion sampler. The image-to-3D model is unchanged; the only
addition is a **feature cache on its flow-matching loop**.

A flow-matching sampler integrates `dx/dt = v_θ(x, t)` over `N` steps, calling the expensive DiT at every
step to get the velocity `v`. HiCache computes the DiT only on `1/interval` of the steps and **forecasts**
the velocity on the rest from a small cache of recent anchors — skipping `(interval-1)/interval` of the
forward passes. The forecast is a first-class part of the denoise loop (`__call__` calls the cache helpers
directly — there is **no runtime monkey-patching**).

## Method — Hermite polynomial forecast

At each compute ("full") step HiCache stores the CFG-combined velocity `F_t` and updates backward
finite-difference derivatives `Δ^i F_t`. At a skipped step `k` steps past the last compute step, it forecasts

```
F_hat_{t-k} = F_t + Σ_{i≥1} (Δ^i F_t / i!) · H̃_i(-k)
```

with the **dual-scaled physicist's Hermite** polynomial `H̃_n(x) = σ^n · H_n(σ·x)`, `σ ∈ (0,1)`. The `σ`
contraction keeps the high-order terms bounded, giving a more stable extrapolation than the equivalent
Taylor (monomial) series — TaylorSeer is the special case `H̃_i(-k) = (-k)^i`. With fewer than two anchors
the forecast degenerates to plain reuse of the cached velocity (the correct zero-information forecast).

## Use

Load the mini pipeline and turn the cache on with one call before sampling:

```python
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini', subfolder='hunyuan3d-dit-v2-mini',
)

# HiCache (Hermite): one DiT step then (interval-1) forecasts.
pipe.enable_hicache(
    interval=4,        # N: 1 compute step, then interval-1 forecasts
    max_order=1,       # highest finite-difference / Hermite order
    first_enhance=2,   # always compute the first few steps (warm-up)
    end_enhance=None,  # always compute steps with index >= end_enhance (default: none)
    sigma=0.5,         # Hermite contraction σ ∈ (0,1)
)
# pipe.disable_hicache()  # restore the uncached sampler

mesh = pipe(image=Image.open('assets/demo.png').convert('RGBA'))[0]
mesh.export('demo.glb')
```

`enable_hicache` just stores the config; `Hunyuan3DDiTFlowMatchingPipeline.__call__` reads it and runs the
compute/forecast schedule natively (see [`hy3dgen/shapegen/hicache.py`](hy3dgen/shapegen/hicache.py) and the
denoise loop in [`hy3dgen/shapegen/pipelines.py`](hy3dgen/shapegen/pipelines.py)).

## Results

HiCache is the **polynomial baseline**: lossless at low skip intervals, then degrading as the skip grows,
because the Hermite (polynomial) basis is only a local truncation of the trajectory's true exponential
class and diverges under extrapolation. On Hunyuan3D-2.1 the Toys4K F-score@0.05 falls `0.88 → 0.74 → 0.38`
at interval `3 / 5 / 6`.

For the **exponential** forecaster that stays lossless at larger skips — `0.794` F-score at interval-5,
exactly matching the uncached baseline on the deployed Hunyuan3D-2 mini — see the
[**`hunyuan2-plus-plus`**](../hunyuan2-plus-plus) fork (HiCache++ / DMD), built on the standalone
[**`hicache-plus-plus`**](../hicache-plus-plus) library.

## Attribution

- **Hunyuan3D-2 / Hunyuan3D-2 mini** © Tencent — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)
  (Tencent Hunyuan Non-Commercial License). This fork adds only the HiCache integration; the model,
  weights, and pipeline are unchanged.
- **HiCache** — *Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting*
  ([arXiv:2508.16984](https://arxiv.org/abs/2508.16984)). The Hermite/finite-difference core in
  [`hicache.py`](hy3dgen/shapegen/hicache.py) is a clean reimplementation; only the loop wiring is
  Hunyuan-specific. Built on **TaylorSeer** (monomial feature caching).

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
