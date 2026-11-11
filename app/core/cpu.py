"""CPU detection and progressive SIMD runtime selection.

Detects host CPU features (AVX2, FMA, BMI) to select the optimal
Polars runtime (standard AVX2 for maximum performance vs polars-runtime-compat
for legacy CPUs and virtual machines without AVX2).
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def configure_progressive_runtime() -> str:
    """Inspects CPU capabilities and configures POLARS_FORCE_PKG if not already set.

    Returns:
        The selected runtime ("32", "compat", or user-specified value).
    """
    if "POLARS_FORCE_PKG" in os.environ:
        return os.environ["POLARS_FORCE_PKG"]

    # On Linux, inspect /proc/cpuinfo
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            content = cpuinfo_path.read_text(encoding="utf-8", errors="ignore").lower()
            # x86-64-v3 requires AVX2 and FMA
            has_avx2 = "avx2" in content
            has_fma = "fma" in content

            if has_avx2 and has_fma:
                os.environ["POLARS_FORCE_PKG"] = "32"
                return "32"
            else:
                os.environ["POLARS_FORCE_PKG"] = "compat"
                os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
                return "compat"
        except Exception:
            pass

    return os.environ.get("POLARS_FORCE_PKG", "auto")


# Configured once on module import before any polars usage:
SELECTED_RUNTIME = configure_progressive_runtime()
