<div align="center">

<p align="center"><img src="assets/banner.png" alt="hunyuan2-plus" width="680"></p>

# Hunyuan3D-2 mini + HiCache

<p>
  <a href="https://github.com/Archerkattri/hunyuan2-plus/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Archerkattri/hunyuan2-plus?color=1f6feb"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Archerkattri/hunyuan2-plus?color=0d9488"></a>
</p>


**Tencent's 0.6B image-to-3D, accelerated by training-free *Hermite-polynomial* velocity forecasting.**

*[Hunyuan3D-2 mini](https://huggingface.co/tencent/Hunyuan3D-2mini) with [HiCache](https://arxiv.org/abs/2508.16984)
wired natively into its flow-matching denoise loop: skip the DiT on most sampling steps and forecast
the cached velocity instead — no retraining, no monkey-patching.*

[![base: Hunyuan3D-2 mini](https://img.shields.io/badge/base-Hunyuan3D--2%20mini-d96902)](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
&nbsp;[![arXiv: Hunyuan3D-2.0](https://img.shields.io/badge/arXiv-2501.12202-b5212f?logo=arxiv)](https://arxiv.org/abs/2501.12202)
&nbsp;[![arXiv: HiCache](https://img.shields.io/badge/arXiv-2508.16984-b5212f?logo=arxiv)](https://arxiv.org/abs/2508.16984)
&nbsp;[![license: Tencent Hunyuan 3D 2.0 Community](https://img.shields.io/badge/license-Tencent%20Hunyuan%203D%202.0%20Community-2e6db0)](./LICENSE)
&nbsp;![basis: Hermite polynomial](https://img.shields.io/badge/basis-Hermite%20polynomial-7a5cc6)

</div>

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis — *the same near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `hunyuan2-plus` — **Hunyuan3D-2 mini × HiCache (Hermite)** — published cache on the deployed mini shape model.

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
F_hat_{t+k} = F_t + Σ_{i≥1} (Δ^i F_t / i!) · H̃_i(k)
```

with the **dual-scaled physicist's Hermite** polynomial `H̃_n(x) = σ^n · H_n(σ·x)`, `σ ∈ (0,1)`. The `σ`
contraction keeps the high-order terms bounded, giving a more stable extrapolation than the equivalent
Taylor (monomial) series — TaylorSeer is the special case `H̃_i(k) = k^i`. With fewer than two anchors
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
[**`hunyuan2-plus-plus`**](https://github.com/Archerkattri/hunyuan2-plus-plus) fork (HiCache++ / DMD), built on the standalone
[**`hicache-plus-plus`**](https://github.com/Archerkattri/hicache-plus-plus) library.

### Sign-convention update (2026-06-10)

The vendored Hermite forecast in `hicache.py` evaluated the basis at `x = -k`; the corrected
convention from [hicache-plus-plus 1.2.0](https://github.com/Archerkattri/hicache-plus-plus)
is `x = +k` (the upstream TaylorSeer distance convention; `-k` flips every odd-order term,
extrapolating backwards). This fork now ships the corrected forecast. All previously published
numbers were measured with the as-released code and remain valid as-measured.

Re-validated on the deployed **Hunyuan3D-2 mini** (Toys4K, 10 objects, F-score\@0.05, the
deterministic protocol of the family benchmark; the as-released arm reproduces the published
cells bit-exact):

| config | as-released (`x = -k`) | corrected (`x = +k`) |
|---|---:|---:|
| vanilla (uncached) | 0.794 | 0.794 |
| HiCache i3/o2 | 0.792 | 0.760 |
| HiCache i5/o3 (probe) | 0.738 | **0.784** |
| HiCache i6/o3 (probe) | 0.578 | 0.404 |

Reading the table honestly:

- **At the published interval (i3/o2) the corrected forecast matches the as-released one.**
  The 0.792 vs 0.760 full-set gap is a single Go-ICP alignment timeout on a rotationally
  symmetric bowl (scored unaligned, a known protocol artifact); over the 7 objects aligned
  in both runs the means are 0.922 (as-released) vs 0.921 (corrected), vanilla 0.911.
- **At i5/o3 the corrected sign is clearly better** (0.784 vs 0.738, baseline 0.794), and it
  rescues a catastrophic as-released cell (bottle 0.25 to 0.97). This is the regime where the
  backwards extrapolation of the old sign bites.
- **At i6/o3 both conventions are far below baseline** (0.58 / 0.40 vs 0.79): the polynomial
  skip ceiling dominates regardless of sign. For wide skips use the exponential (DMD) fork,
  which stays at 0.793 at interval-5 with the corrected warm-up fallback (re-validated).

## Attribution

- **Hunyuan3D-2 / Hunyuan3D-2 mini** © Tencent — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)
  (Tencent Hunyuan 3D 2.0 Community License Agreement; note its territorial limits, large-user
  threshold, and no-competing-model-training restrictions). This fork adds only the HiCache
  integration; the model, weights, and pipeline are unchanged.
- **HiCache** — *Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting*
  ([arXiv:2508.16984](https://arxiv.org/abs/2508.16984)). The Hermite/finite-difference core in
  [`hicache.py`](hy3dgen/shapegen/hicache.py) is a clean reimplementation; only the loop wiring is
  Hunyuan-specific. Built on **TaylorSeer** (monomial feature caching).

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.

## Citation

If you use this work, please cite the base model (Tencent Hunyuan3D-2) and the acceleration method (HiCache):

```bibtex
@misc{lai2025flashvdm,
      title={Unleashing Vecset Diffusion Model for Fast Shape Generation}, 
      author={Zeqiang Lai and Yunfei Zhao and Zibo Zhao and Haolin Liu and Fuyun Wang and Huiwen Shi and Xianghui Yang and Qinxiang Lin and Jinwei Huang and Yuhong Liu and Jie Jiang and Chunchao Guo and Xiangyu Yue},
      year={2025},
      eprint={2503.16302},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16302}, 
}
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```bibtex
@misc{hicache2025, title={HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting}, eprint={2508.16984}, archivePrefix={arXiv}, year={2025}}
```

---

## Family

Part of the **HiCache++ acceleration family**.

- **Family hub:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the basis library behind this adapter.
- **Sibling:** [`hunyuan2-plus-plus`](https://github.com/Archerkattri/hunyuan2-plus-plus) — the same base model with the HiCache++ (Dynamic Mode Decomposition / Prony) exponential-forecast variant.
