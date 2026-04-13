"""
Tier-Aware KV Cache Orchestrator
High-level Python manager that interfaces with the C++ allocator and serves
as the entry point for vLLM / PyTorch integration.
"""

import os
import time
import uuid
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class MemoryTier(Enum):
    TIER1_FAST = auto()   # GPU HBM / Local DRAM
    TIER2_SLOW = auto()   # CXL-attached memory / NVMe SSD


class RequestPhase(Enum):
    PREFILL = auto()
    DECODE  = auto()
    DONE    = auto()


BLOCK_SIZE         = 16          # tokens per logical block
TIER1_CAPACITY_MB  = 4096        # configurable
TIER2_CAPACITY_MB  = 32768       # configurable
AOL_DEMOTION_THRESH = 0.3        # blocks with AOL score below this → demote
TIER1_PRESSURE_PCT  = 0.85       # trigger migration at 85 % utilisation


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LogicalBlock:
    block_id:    int
    request_id:  str
    layer_idx:   int
    token_start: int
    token_end:   int
    tier:        MemoryTier = MemoryTier.TIER1_FAST
    ref_count:   int        = 0
    last_access: float      = field(default_factory=time.time)
    access_count: int       = 0
    aol_score:   float      = 1.0   # higher → more critical
    is_shared:   bool       = False  # CoW flag


@dataclass
class RequestContext:
    request_id:    str
    prompt_tokens: int
    phase:         RequestPhase = RequestPhase.PREFILL
    blocks:        List[int]    = field(default_factory=list)
    created_at:    float        = field(default_factory=time.time)
    decode_step:   int          = 0


# ---------------------------------------------------------------------------
# Tier-Aware KV Cache Orchestrator
# ---------------------------------------------------------------------------

class TierAwareKVCacheOrchestrator:
    """
    Central Python-level manager.  The heavy allocation and profiling work
    lives in C++/CUDA extensions (stubs provided); this class coordinates
    the high-level policy.
    """

    def __init__(
        self,
        tier1_capacity_mb: int = TIER1_CAPACITY_MB,
        tier2_capacity_mb: int = TIER2_CAPACITY_MB,
        block_size: int = BLOCK_SIZE,
        num_layers: int = 32,
        head_dim: int = 128,
        num_heads: int = 8,
    ):
        self.tier1_capacity_mb = tier1_capacity_mb
        self.tier2_capacity_mb = tier2_capacity_mb
        self.block_size  = block_size
        self.num_layers  = num_layers
        self.head_dim    = head_dim
        self.num_heads   = num_heads

        # Block registry
        self._blocks: Dict[int, LogicalBlock]    = {}
        self._requests: Dict[str, RequestContext] = {}
        self._next_block_id = 0
        self._lock = threading.RLock()

        # Tier usage tracking (in MB)
        self._tier1_used_mb: float = 0.0
        self._tier2_used_mb: float = 0.0

        # Bytes per block (K + V, all layers, fp16)
        self._bytes_per_block = (
            2 * num_layers * num_heads * head_dim * block_size * 2  # fp16
        )
        self._mb_per_block = self._bytes_per_block / (1024 ** 2)

        # Background demotion daemon
        self._demotion_thread = threading.Thread(
            target=self._kdemoted_daemon, daemon=True, name="kdemoted"
        )
        self._demotion_thread.start()

        logger.info(
            "TierAwareKVCacheOrchestrator initialised | "
            f"T1={tier1_capacity_mb}MB  T2={tier2_capacity_mb}MB  "
            f"block={block_size}tok  {self._mb_per_block:.2f}MB/block"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_request(self, prompt_tokens: int) -> str:
        """Create a new request context and pre-allocate virtual space."""
        rid = str(uuid.uuid4())
        ctx = RequestContext(request_id=rid, prompt_tokens=prompt_tokens)
        with self._lock:
            self._requests[rid] = ctx
        logger.debug(f"Registered request {rid} ({prompt_tokens} prompt tokens)")
        return rid

    def allocate_blocks(
        self, request_id: str, num_new_tokens: int, layer_idx: int
    ) -> List[int]:
        """
        Allocate logical blocks for a given request / layer.
        Returns list of block IDs.
        """
        num_blocks = (num_new_tokens + self.block_size - 1) // self.block_size
        block_ids  = []

        with self._lock:
            ctx = self._requests.get(request_id)
            if ctx is None:
                raise ValueError(f"Unknown request_id: {request_id}")

            for i in range(num_blocks):
                bid = self._next_block_id
                self._next_block_id += 1

                # Decide tier: prefer T1, fall back to T2 under pressure
                tier = self._choose_tier()

                blk = LogicalBlock(
                    block_id=bid,
                    request_id=request_id,
                    layer_idx=layer_idx,
                    token_start=i * self.block_size,
                    token_end=min((i + 1) * self.block_size, num_new_tokens),
                    tier=tier,
                )
                self._blocks[bid] = blk
                ctx.blocks.append(bid)
                self._update_tier_usage(tier, +1)
                block_ids.append(bid)

        return block_ids

    def access_block(self, block_id: int) -> MemoryTier:
        """Record an access; update LRU / AOL metadata. Returns current tier."""
        with self._lock:
            blk = self._blocks.get(block_id)
            if blk is None:
                raise KeyError(f"Block {block_id} not found")
            blk.last_access  = time.time()
            blk.access_count += 1
            return blk.tier

    def update_aol_score(self, block_id: int, aol_score: float) -> None:
        """Called by the C++/Rust AOL Profiler to push updated criticality."""
        with self._lock:
            blk = self._blocks.get(block_id)
            if blk:
                blk.aol_score = max(0.0, min(1.0, aol_score))

    def promote_block(self, block_id: int) -> bool:
        """Move a block from T2 → T1 (triggered by demand)."""
        with self._lock:
            blk = self._blocks.get(block_id)
            if blk is None or blk.tier == MemoryTier.TIER1_FAST:
                return False
            if not self._has_tier1_space():
                self._evict_one_to_tier2()
            blk.tier = MemoryTier.TIER1_FAST
            self._update_tier_usage(MemoryTier.TIER2_SLOW, -1)
            self._update_tier_usage(MemoryTier.TIER1_FAST, +1)
            logger.debug(f"Promoted block {block_id} T2→T1")
            return True

    def free_request(self, request_id: str) -> None:
        """Release all blocks for a completed request."""
        with self._lock:
            ctx = self._requests.pop(request_id, None)
            if ctx is None:
                return
            for bid in ctx.blocks:
                blk = self._blocks.pop(bid, None)
                if blk:
                    self._update_tier_usage(blk.tier, -1)
        logger.debug(f"Freed request {request_id}")

    def get_stats(self) -> dict:
        with self._lock:
            t1_cap  = self.tier1_capacity_mb
            t2_cap  = self.tier2_capacity_mb
            t1_used = self._tier1_used_mb
            t2_used = self._tier2_used_mb
            total   = len(self._blocks)
            t1_blks = sum(1 for b in self._blocks.values()
                         if b.tier == MemoryTier.TIER1_FAST)
            t2_blks = total - t1_blks
        return {
            "tier1_used_mb":   round(t1_used, 2),
            "tier1_cap_mb":    t1_cap,
            "tier1_util_pct":  round(t1_used / t1_cap * 100, 1) if t1_cap else 0,
            "tier2_used_mb":   round(t2_used, 2),
            "tier2_cap_mb":    t2_cap,
            "tier2_util_pct":  round(t2_used / t2_cap * 100, 1) if t2_cap else 0,
            "total_blocks":    total,
            "tier1_blocks":    t1_blks,
            "tier2_blocks":    t2_blks,
            "active_requests": len(self._requests),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_tier(self) -> MemoryTier:
        if self._has_tier1_space():
            return MemoryTier.TIER1_FAST
        return MemoryTier.TIER2_SLOW

    def _has_tier1_space(self) -> bool:
        return (
            self._tier1_used_mb + self._mb_per_block
            <= self.tier1_capacity_mb * TIER1_PRESSURE_PCT
        )

    def _update_tier_usage(self, tier: MemoryTier, delta: int) -> None:
        mb = delta * self._mb_per_block
        if tier == MemoryTier.TIER1_FAST:
            self._tier1_used_mb = max(0.0, self._tier1_used_mb + mb)
        else:
            self._tier2_used_mb = max(0.0, self._tier2_used_mb + mb)

    def _evict_one_to_tier2(self) -> None:
        """Demote the T1 block with the lowest AOL score."""
        candidates = [
            b for b in self._blocks.values()
            if b.tier == MemoryTier.TIER1_FAST and b.ref_count == 0
        ]
        if not candidates:
            return
        victim = min(candidates, key=lambda b: b.aol_score)
        victim.tier = MemoryTier.TIER2_SLOW
        self._update_tier_usage(MemoryTier.TIER1_FAST, -1)
        self._update_tier_usage(MemoryTier.TIER2_SLOW, +1)
        logger.debug(
            f"Demoted block {victim.block_id} T1→T2 (AOL={victim.aol_score:.3f})"
        )

    def _kdemoted_daemon(self) -> None:
        """
        Background thread: periodically sweep T1 blocks with low AOL scores
        and demote them to T2 before memory pressure hits.
        """
        while True:
            time.sleep(0.5)
            try:
                with self._lock:
                    t1_util = (
                        self._tier1_used_mb / self.tier1_capacity_mb
                        if self.tier1_capacity_mb else 0
                    )
                    if t1_util < TIER1_PRESSURE_PCT:
                        continue
                    # Batch-demote lowest-AOL cold blocks
                    candidates = sorted(
                        [
                            b for b in self._blocks.values()
                            if b.tier == MemoryTier.TIER1_FAST
                            and b.ref_count == 0
                            and b.aol_score < AOL_DEMOTION_THRESH
                        ],
                        key=lambda b: b.aol_score,
                    )
                    demote_count = max(1, len(candidates) // 4)
                    for blk in candidates[:demote_count]:
                        blk.tier = MemoryTier.TIER2_SLOW
                        self._update_tier_usage(MemoryTier.TIER1_FAST, -1)
                        self._update_tier_usage(MemoryTier.TIER2_SLOW, +1)
                    if demote_count:
                        logger.info(
                            f"[kdemoted] Demoted {demote_count} blocks "
                            f"(T1 util was {t1_util*100:.1f}%)"
                        )
            except Exception as e:
                logger.error(f"[kdemoted] error: {e}")
