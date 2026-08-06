from __future__ import annotations

import gc
import os
from time import perf_counter


def prepare_runtime(allow_tf32: bool, cpu_threads: int) -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cuda_malloc_async")
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = allow_tf32


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def is_cuda_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        signal in message
        for signal in (
            "cuda out of memory",
            "cuda_error_out_of_memory",
            "out of memory",
            "failed to allocate",
        )
    )


def elapsed(started: float) -> float:
    return round(perf_counter() - started, 3)
