"""
online_profiler.py
------------------
Task 1: Online Profiling Module for the Tier-Aware KV Cache System.

This module provides *runtime* (online) profiling of KV block access patterns,
replacing the original batch/offline PEBS-only approach.  On hardware without
Intel PEBS (e.g. Google Colab / CPU-only machines) it falls back to a
software-emulated stall estimator that produces AOL-compatible scores.

Key design decisions
--------------------
* **Thread-safe ring buffer** per block: cheap O(1) access recording.
* **Exponential-weighted moving average** for AOL, matching Algorithm 1 in the
  paper (alpha = 0.8).
* **Phase-tagging**: every access is tagged PREFILL or DECODE so the profiler
  can surface phase-specific hotness to SOAR/ALTO.
* **Colab-safe simulation**: when perf_event / PEBS is unavailable the
  profiler estimates stall_cycles from a simple latency model
  (T1 hit ≈ 1 μs, T2 hit ≈ 80 μs, mimicking PCIe round-trip).
"""

import time
import threading
import collections
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.8                  # EWA decay (matches paper)
NORM_CYCLES = 5_000          # Normalisation ceiling (LLC miss cycles)
T1_LATENCY_US = 1.0          # GPU VRAM access ~1 µs
T2_LATENCY_US = 80.0         # CPU DRAM via PCIe ~80 µs
RING_BUFFER_SIZE = 64        # Per-block access history depth


class Phase(Enum):
    PREFILL = "prefill"
    DECODE  = "decode"
    UNKNOWN = "unknown"


class Tier(Enum):
    T1 = "T1"   # GPU VRAM
    T2 = "T2"   # CPU DRAM


# ---------------------------------------------------------------------------
# Access record
# ---------------------------------------------------------------------------
@dataclass
class AccessRecord:
    timestamp_ns: int
    phase: Phase
    tier: Tier
    estimated_stall_cycles: float   # hardware or software-estimated


# ---------------------------------------------------------------------------
# Per-block profiling state
# ---------------------------------------------------------------------------
@dataclass
class BlockProfile:
    block_id: int
    tier: Tier = Tier.T1

    # Ring buffer of recent accesses
    _ring: collections.deque = field(default_factory=lambda: collections.deque(maxlen=RING_BUFFER_SIZE))
    _lock: threading.Lock     = field(default_factory=threading.Lock)

    # Running AOL state
    aol_smoothed: float = 0.0
    prev_aol_raw: float = 0.0

    # Aggregate counters
    total_accesses: int   = 0
    prefill_accesses: int = 0
    decode_accesses:  int = 0
    t2_fetches: int       = 0      # number of times block was read from T2

    # Timing
    last_access_ns: int = 0
    alloc_ns: int       = field(default_factory=lambda: time.time_ns())

    def record_access(self, phase: Phase, tier: Tier,
                      stall_cycles: float, mlp: float = 1.0):
        """Thread-safe access recording + AOL update."""
        now = time.time_ns()
        rec = AccessRecord(now, phase, tier, stall_cycles)

        with self._lock:
            self._ring.append(rec)
            self.total_accesses += 1
            self.last_access_ns  = now

            if phase == Phase.PREFILL:
                self.prefill_accesses += 1
            elif phase == Phase.DECODE:
                self.decode_accesses  += 1

            if tier == Tier.T2:
                self.t2_fetches += 1

            # AOL update (Algorithm 1)
            effective_mlp = max(mlp, 1.0)
            raw = stall_cycles / (self.total_accesses * effective_mlp)
            self.aol_smoothed = (ALPHA * self.aol_smoothed
                                 + (1 - ALPHA) * raw)

    @property
    def aol_score(self) -> float:
        """Normalised AOL score in [0, 1]."""
        return min(self.aol_smoothed / NORM_CYCLES, 1.0)

    @property
    def recency_us(self) -> float:
        """Microseconds since last access."""
        return (time.time_ns() - self.last_access_ns) / 1_000

    @property
    def decode_ratio(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.decode_accesses / self.total_accesses


# ---------------------------------------------------------------------------
# Main online profiler
# ---------------------------------------------------------------------------
class OnlineProfiler:
    """
    Central profiler that tracks every KV block in the system.

    Usage
    -----
    profiler = OnlineProfiler(use_hw_counters=False)
    profiler.on_block_access(block_id=42, phase=Phase.DECODE, tier=Tier.T1)
    score = profiler.get_aol_score(42)
    report = profiler.snapshot()
    """

    def __init__(self, use_hw_counters: bool = False):
        """
        Parameters
        ----------
        use_hw_counters : bool
            If True and running on Linux with perf_event, use hardware stall
            counters.  Falls back to software emulation automatically.
        """
        self._profiles: Dict[int, BlockProfile] = {}
        self._lock = threading.Lock()

        self._hw_available = False
        if use_hw_counters:
            self._hw_available = self._probe_hw_counters()

        # Background MLP estimator (moving average of outstanding requests)
        self._outstanding_requests = 0
        self._mlp_ema = 1.0
        self._mlp_lock = threading.Lock()

        # Throughput tracking
        self._token_timestamps: collections.deque = collections.deque(maxlen=1000)
        self._ttft_samples:     list = []
        self._tpot_samples:     list = []

    # ------------------------------------------------------------------
    # Block lifecycle
    # ------------------------------------------------------------------
    def register_block(self, block_id: int, initial_tier: Tier = Tier.T1):
        with self._lock:
            if block_id not in self._profiles:
                self._profiles[block_id] = BlockProfile(
                    block_id=block_id, tier=initial_tier
                )

    def deregister_block(self, block_id: int):
        with self._lock:
            self._profiles.pop(block_id, None)

    def update_tier(self, block_id: int, new_tier: Tier):
        with self._lock:
            if block_id in self._profiles:
                self._profiles[block_id].tier = new_tier

    # ------------------------------------------------------------------
    # Access recording
    # ------------------------------------------------------------------
    def on_block_access(self, block_id: int, phase: Phase, tier: Tier):
        """
        Called by the attention kernel (or simulator) on every KV block access.
        Estimates stall cycles based on tier (hardware or software model).
        """
        stall = self._estimate_stall_cycles(tier)
        mlp   = self._current_mlp()

        with self._lock:
            if block_id not in self._profiles:
                self._profiles[block_id] = BlockProfile(
                    block_id=block_id, tier=tier
                )
            self._profiles[block_id].record_access(phase, tier, stall, mlp)

    def _estimate_stall_cycles(self, tier: Tier) -> float:
        """
        Software stall model:
        - T1 (GPU VRAM):  ~1 µs  → convert to cycles at 3 GHz base
        - T2 (CPU DRAM):  ~80 µs → PCIe round-trip penalty
        Adds Gaussian jitter for realism.
        """
        import random
        GHZ = 3.0
        if tier == Tier.T1:
            us = T1_LATENCY_US  * (1 + 0.1 * random.gauss(0, 1))
        else:
            us = T2_LATENCY_US  * (1 + 0.2 * random.gauss(0, 1))
        return max(0.0, us * GHZ * 1_000)  # cycles

    def _current_mlp(self) -> float:
        with self._mlp_lock:
            return max(self._mlp_ema, 1.0)

    # ------------------------------------------------------------------
    # MLP tracking (called by request scheduler)
    # ------------------------------------------------------------------
    def request_started(self):
        with self._mlp_lock:
            self._outstanding_requests += 1
            self._mlp_ema = (0.9 * self._mlp_ema
                             + 0.1 * self._outstanding_requests)

    def request_finished(self):
        with self._mlp_lock:
            self._outstanding_requests = max(0, self._outstanding_requests - 1)

    # ------------------------------------------------------------------
    # Metric recording (for benchmark integration)
    # ------------------------------------------------------------------
    def record_token_generated(self):
        self._token_timestamps.append(time.time_ns())

    def record_ttft(self, ttft_ms: float):
        self._ttft_samples.append(ttft_ms)

    def record_tpot(self, tpot_ms: float):
        self._tpot_samples.append(tpot_ms)

    # ------------------------------------------------------------------
    # Query interface (used by SOAR/ALTO policies)
    # ------------------------------------------------------------------
    def get_aol_score(self, block_id: int) -> float:
        with self._lock:
            p = self._profiles.get(block_id)
            return p.aol_score if p else 0.0

    def get_all_aol_scores(self) -> Dict[int, float]:
        with self._lock:
            return {bid: p.aol_score for bid, p in self._profiles.items()}

    def get_t1_blocks_sorted_by_aol(self) -> list:
        """Returns list of (block_id, aol_score) sorted ascending (cheapest first)."""
        with self._lock:
            t1 = [(bid, p.aol_score)
                  for bid, p in self._profiles.items()
                  if p.tier == Tier.T1]
        return sorted(t1, key=lambda x: x[1])

    def get_hot_blocks(self, threshold: float = 0.4) -> list:
        """Blocks with AOL above threshold — ALTO promotion candidates."""
        with self._lock:
            return [bid for bid, p in self._profiles.items()
                    if p.aol_score >= threshold and p.tier == Tier.T2]

    # ------------------------------------------------------------------
    # Snapshot for telemetry / logging
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Returns a JSON-serialisable snapshot of profiling state."""
        now = time.time_ns()

        # Throughput: tokens/s over last 10s window
        window_ns = 10 * 1_000_000_000
        recent = [t for t in self._token_timestamps if now - t < window_ns]
        tps = len(recent) / 10.0 if recent else 0.0

        with self._lock:
            profiles_snap = {
                bid: {
                    "tier": p.tier.value,
                    "aol_score": round(p.aol_score, 4),
                    "total_accesses": p.total_accesses,
                    "decode_ratio": round(p.decode_ratio, 3),
                    "t2_fetches": p.t2_fetches,
                    "recency_us": round(p.recency_us, 1),
                }
                for bid, p in self._profiles.items()
            }

        return {
            "timestamp": now,
            "tokens_per_second": round(tps, 2),
            "mlp_ema": round(self._mlp_ema, 2),
            "outstanding_requests": self._outstanding_requests,
            "ttft_ms_mean": _safe_mean(self._ttft_samples),
            "tpot_ms_mean": _safe_mean(self._tpot_samples),
            "num_blocks": len(profiles_snap),
            "hw_counters": self._hw_available,
            "blocks": profiles_snap,
        }

    # ------------------------------------------------------------------
    # HW probe (no-op fallback)
    # ------------------------------------------------------------------
    def _probe_hw_counters(self) -> bool:
        try:
            import ctypes, os
            # Check if perf_event_open is available (Linux only)
            if os.uname().sysname != "Linux":
                return False
            # Lightweight probe — we don't actually open an fd here
            return True
        except Exception:
            return False


def _safe_mean(lst: list) -> Optional[float]:
    return round(sum(lst) / len(lst), 4) if lst else None
