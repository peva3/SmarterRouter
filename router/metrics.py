"""Prometheus metrics for SmarterRouter."""

from typing import cast

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    REGISTRY,
)

# Request metrics
REQUESTS_TOTAL = Counter(
    "smarterrouter_requests_total",
    "Total number of requests",
    ["endpoint", "method"],
)

REQUEST_DURATION = Histogram(
    "smarterrouter_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

# Error metrics
ERRORS_TOTAL = Counter(
    "smarterrouter_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)

# Model selection metrics
MODEL_SELECTIONS_TOTAL = Counter(
    "smarterrouter_model_selections_total",
    "Total number of model selections",
    ["selected_model", "category"],
)

# Cache metrics
CACHE_HITS_TOTAL = Counter(
    "smarterrouter_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # "routing" or "response"
)

CACHE_MISSES_TOTAL = Counter(
    "smarterrouter_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)

# VRAM metrics
VRAM_TOTAL_GB = Gauge(
    "smarterrouter_vram_total_gb",
    "Total GPU VRAM in GB",
)

VRAM_USED_GB = Gauge(
    "smarterrouter_vram_used_gb",
    "Used GPU VRAM in GB",
)

VRAM_UTILIZATION_PCT = Gauge(
    "smarterrouter_vram_utilization_pct",
    "GPU VRAM utilization percentage",
)

# Per-GPU metrics (dynamic, will be registered per GPU index)
def create_gpu_metrics():
    """Create per-GPU metrics with labels."""
    return {
        "total": Gauge("smarterrouter_gpu_total_gb", "Total VRAM per GPU", ["gpu_index", "vendor"]),
        "used": Gauge("smarterrouter_gpu_used_gb", "Used VRAM per GPU", ["gpu_index", "vendor"]),
        "free": Gauge("smarterrouter_gpu_free_gb", "Free VRAM per GPU", ["gpu_index", "vendor"]),
    }

gpu_metrics = create_gpu_metrics()


def generate_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    result = generate_latest(REGISTRY)
    return cast(bytes, result) if result else b""
