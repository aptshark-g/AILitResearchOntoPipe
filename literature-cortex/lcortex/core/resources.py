"""Resource profile — auto-detect VM resources and adjust parallelism.

Reads available memory at startup, classifies into tiers (low/medium/high),
and adjusts concurrency limits.  Supports user override via config or env.

Usage:
    from lcortex.core.resources import ResourceProfile
    rp = ResourceProfile.from_env()
    print(rp.tier)        # "low" | "medium" | "high"
    print(rp.max_arxiv_concurrent)  # 1-4
    print(rp.max_llm_concurrent)    # 1-3
    print(rp.batch_size)            # papers per batch
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

log = logging.getLogger("lcortex.resources")


# ═══════════════════════════════════════════════════════════════════════
# Platform helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_total_memory_mb() -> int:
    """Cross-platform total system memory in MB."""
    try:
        # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    try:
        # macOS
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) // (1024 * 1024)
    except Exception:
        pass
    return 1024  # fallback: assume 1 GB


def _get_available_memory_mb() -> int:
    """Cross-platform available (free + reclaimable) memory in MB."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) >= 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            # MemAvailable exists on Linux 3.14+
            if "MemAvailable" in meminfo:
                return meminfo["MemAvailable"] // 1024
            # Fallback: MemFree + Cached + Buffers
            return (meminfo.get("MemFree", 0) + meminfo.get("Cached", 0) + meminfo.get("Buffers", 0)) // 1024
    except Exception:
        pass
    return _get_total_memory_mb() // 2  # assume 50% available


def _get_cpu_count() -> int:
    """Return logical CPU count."""
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2


# ═══════════════════════════════════════════════════════════════════════
# Resource tier classification
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ResourceProfile:
    """Auto-tuned resource limits based on available system memory.

    Three tiers:

    =======  ===========  ========  =========  ===============
    Tier     Memory       Arxiv     LLM        Batch size
    =======  ===========  ========  =========  ===============
    low      < 512 MB      1 (seq)  1 (seq)    4 papers
    medium   512 MB - 2 GB 2         1          8 papers
    high     > 2 GB        4         2          15 papers
    =======  ===========  ========  =========  ===============
    """

    tier: str = "medium"
    total_memory_mb: int = 1024
    available_memory_mb: int = 512
    cpu_count: int = 2

    # ── Concurrency limits ──
    max_arxiv_concurrent: int = 1       # arXiv search API calls in parallel
    max_openalex_concurrent: int = 1    # OpenAlex API calls in parallel
    max_llm_concurrent: int = 1         # LLM calls in parallel (Phase B2/F/E)

    # ── Pipeline knobs ──
    batch_size: int = 8                 # Papers per batch in B1/B2
    b2_top_n: int = 10                  # Max papers sent to B2 LLM scoring
    max_cd_searches: int = 4            # Max C/D arxiv queries per phase
    stream_mode: bool = True            # Use streaming (JSONL) by default

    # ── Memory guards ──
    gc_after_phase: bool = True         # Explicit gc.collect() after each phase
    gc_after_arxiv: bool = False        # Explicit gc.collect() after each arXiv call

    # ── Metadata ──
    user_overrides: list[str] = field(default_factory=list)
    auto_detected: bool = True

    @classmethod
    def from_env(cls, config: dict | None = None) -> "ResourceProfile":
        """Detect resources + apply user overrides from config / env."""
        total = _get_total_memory_mb()
        avail = _get_available_memory_mb()
        cpus = _get_cpu_count()

        # ── Auto-classify ──
        if avail < 512:
            tier = "low"
        elif avail < 2048:
            tier = "medium"
        else:
            tier = "high"

        rp = cls(
            tier=tier,
            total_memory_mb=total,
            available_memory_mb=avail,
            cpu_count=cpus,
        )
        rp._apply_tier_defaults()
        rp._apply_user_overrides(config or {})
        rp._apply_env_overrides()

        log.info(
            "Resource profile: tier=%s total=%dMB avail=%dMB CPU=%d "
            "arxiv_concurrent=%d llm_concurrent=%d batch=%d",
            rp.tier, total, avail, cpus,
            rp.max_arxiv_concurrent, rp.max_llm_concurrent, rp.batch_size,
        )
        return rp

    def _apply_tier_defaults(self) -> None:
        """Set sensible defaults for each tier."""
        defaults = {
            "low": {
                "max_arxiv_concurrent": 1,
                "max_openalex_concurrent": 1,
                "max_llm_concurrent": 1,
                "batch_size": 4,
                "b2_top_n": 5,
                "max_cd_searches": 2,
                "stream_mode": True,
                "gc_after_phase": True,
                "gc_after_arxiv": True,
            },
            "medium": {
                "max_arxiv_concurrent": 2,
                "max_openalex_concurrent": 1,
                "max_llm_concurrent": 1,
                "batch_size": 8,
                "b2_top_n": 8,
                "max_cd_searches": 4,
                "stream_mode": True,
                "gc_after_phase": True,
                "gc_after_arxiv": False,
            },
            "high": {
                "max_arxiv_concurrent": 4,
                "max_openalex_concurrent": 2,
                "max_llm_concurrent": 2,
                "batch_size": 15,
                "b2_top_n": 15,
                "max_cd_searches": 8,
                "stream_mode": False,
                "gc_after_phase": False,
                "gc_after_arxiv": False,
            },
        }
        d = defaults.get(self.tier, defaults["medium"])
        for k, v in d.items():
            setattr(self, k, v)

    def _apply_user_overrides(self, config: dict) -> None:
        """Apply user overrides from config file or programmatic config."""
        rc = config.get("resources", {})
        if not rc:
            return

        # User can force a tier
        force_tier = rc.get("force_tier", "")
        if force_tier in ("low", "medium", "high"):
            if force_tier != self.tier:
                self.tier = force_tier
                self.auto_detected = False
                self.user_overrides.append(f"force_tier={force_tier}")
                self._apply_tier_defaults()

        # User can override specific limits
        known_keys = [
            "max_arxiv_concurrent", "max_openalex_concurrent",
            "max_llm_concurrent", "batch_size", "b2_top_n",
            "max_cd_searches", "stream_mode", "gc_after_phase",
            "gc_after_arxiv",
        ]
        for key in known_keys:
            if key in rc:
                setattr(self, key, rc[key])
                self.user_overrides.append(f"{key}={rc[key]}")

    def _apply_env_overrides(self) -> None:
        """Apply LCORTEX_RESOURCE_* environment variable overrides."""
        env_map = {
            "LCORTEX_RESOURCE_TIER": "force_tier",
            "LCORTEX_FORCE_TIER": "force_tier",
            "LCORTEX_ARXIV_CONCURRENT": "max_arxiv_concurrent",
            "LCORTEX_LLM_CONCURRENT": "max_llm_concurrent",
            "LCORTEX_BATCH_SIZE": "batch_size",
            "LCORTEX_B2_TOP_N": "b2_top_n",
            "LCORTEX_MAX_CD_SEARCHES": "max_cd_searches",
        }
        for env_var, attr in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    if attr == "force_tier":
                        if val in ("low", "medium", "high"):
                            self.tier = val
                            self.auto_detected = False
                            self._apply_tier_defaults()
                            self.user_overrides.append(f"env:{env_var}={val}")
                    else:
                        typed_val = type(getattr(self, attr))(val) if val.isdigit() else val.lower() == "true" if val.lower() in ("true", "false") else val
                        setattr(self, attr, typed_val)
                        self.user_overrides.append(f"env:{env_var}={val}")
                except (ValueError, TypeError):
                    log.warning("Invalid env override %s=%s", env_var, val)

    @property
    def profile_summary(self) -> str:
        """Human-readable one-liner."""
        flags = []
        if not self.auto_detected:
            flags.append("manual")
        if self.user_overrides:
            flags.append(f"overrides:{len(self.user_overrides)}")
        return (
            f"tier={self.tier} mem={self.available_memory_mb}MB "
            f"arxiv={self.max_arxiv_concurrent} llm={self.max_llm_concurrent} "
            f"batch={self.batch_size}" + (f" [{', '.join(flags)}]" if flags else "")
        )

    @property
    def is_low_memory(self) -> bool:
        return self.tier == "low"

    @property
    def can_parallel_arxiv(self) -> bool:
        return self.max_arxiv_concurrent > 1

    @property
    def can_parallel_llm(self) -> bool:
        return self.max_llm_concurrent > 1


# ═══════════════════════════════════════════════════════════════════════
# Default profile file
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_RESOURCE_CONFIG = """# Literature Cortex — Resource Profile
# Auto-detected at startup from available memory.
# Override specific values or force a tier.

# resources:
#   # ── Force a specific tier (auto-detected otherwise) ──
#   force_tier: medium          # low | medium | high
#
#   # ── Concurrency limits ──
#   max_arxiv_concurrent: 2     # Parallel arXiv API calls (1-4)
#   max_llm_concurrent: 1       # Parallel LLM calls (1-3)
#
#   # ── Pipeline knobs ──
#   batch_size: 8               # Papers per batch
#   b2_top_n: 10                # Max papers sent to B2
#   max_cd_searches: 4          # Max C/D arxiv queries
#   stream_mode: true           # Use JSONL streaming
#
#   # ── Memory guards ──
#   gc_after_phase: true        # Force GC after each phase
#   gc_after_arxiv: false       # Force GC after each arXiv call

# Environment overrides (no config file needed):
#   export LCORTEX_FORCE_TIER=low
#   export LCORTEX_ARXIV_CONCURRENT=1
#   export LCORTEX_BATCH_SIZE=4
"""  # noqa: E501
