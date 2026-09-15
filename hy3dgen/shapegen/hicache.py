"""Compatibility facade for the shared scalar Hermite HiCache core.

This repository is the Hermite comparison arm for ``hunyuan2-plus-plus``. The
central package supplies corrected-sign arithmetic, schedule, reset, and
telemetry; the local pipeline keeps the native post-CFG update site and the
same public enable/disable API.
"""

try:
    from hicache_pp.hermite import (
        hicache_decide,
        hicache_forecast,
        hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_derivatives,
        physicists_hermite,
        scaled_hermite,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "hunyuan2-plus requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


__all__ = [
    "hicache_decide", "hicache_forecast", "hicache_init", "hicache_reset",
    "hicache_telemetry", "hicache_update_derivatives", "physicists_hermite",
    "scaled_hermite",
]
