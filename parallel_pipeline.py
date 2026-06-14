"""
parallel_pipeline.py
--------------------
Task 2: Parallel Prefill–Decode Execution with KV Cache Overlap.

Does our system benefit from parallelising prefill and decode?
YES — and here is why:

  The original system serialises prefill → phase-transition migration → decode.
  The phase-aware migration (Algorithm 4) already hides *some* PCIe transfer
  latency by running DMA transfers during prefill compute.  But two additional
  opportunities remain:

  1. **Multi-request interleaving**: while Request A is in the *decode* phase
     (GPU-bound, reading T1 blocks), Request B can be in *prefill* (writing new
     KV blocks).  Because prefill writes and decode reads touch *different*
     physical addresses, they are non-conflicting and can overlap with CUDA
     streams.

  2. **Background T2→T1 DMA during decode compute**: the ALTO promotion of
     Request B's blocks can be pipelined against Request A's decode kernel on
     Stream 0, hiding the ~80 µs PCIe latency entirely.

  This file implements a Python-level simulation of that pipeline so you can
  validate the logic without CUDA hardware.  The same structure maps 1-to-1
  onto real CUDA streams in the C++ layer.

Architecture
------------
  PrefillDecodeScheduler
    ├── prefill_queue   (asyncio.Queue)
    ├── decode_queue    (asyncio.Queue)
    ├── migration_queue (asyncio.Queue)  — T2→T1 promotions
    └── 3 workers running concurrently (asyncio tasks)

When running on Colab the scheduler uses asyncio; on real hardware the three
workers map to:
    - CUDA Stream 0 : decode kernel
    - CUDA Stream 1 : prefill kernel
    - CUDA Stream 2 : DMA (cuMemcpyAsync)
with cudaEventRecord/Wait barriers at the phase-transition boundary.
"""

import asyncio
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum

from profiling.online_profiler import OnlineProfiler, Phase, Tier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class RequestPhase(Enum):
    QUEUED   = "queued"
    PREFILL  = "prefill"
    MIGRATING = "migrating"   # T2→T1 DMA (overlapped with next prefill)
    DECODE   = "decode"
    DONE     = "done"


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    block_ids: List[int] = field(default_factory=list)
    phase: RequestPhase  = RequestPhase.QUEUED

    # Timing (ns)
    t_queued:    int = 0
    t_prefill_start: int = 0
    t_prefill_end:   int = 0
    t_decode_start:  int = 0
    t_decode_end:    int = 0

    # Token generation state
    tokens_generated: int = 0

    # Metrics
    ttft_ms:  float = 0.0
    tpot_ms:  float = 0.0
    throughput_tps: float = 0.0


# ---------------------------------------------------------------------------
# Parallel scheduler
# ---------------------------------------------------------------------------
class PrefillDecodeScheduler:
    """
    Simulates parallel prefill + decode + DMA migration on 3 async workers.

    On real hardware replace the asyncio.sleep() calls with actual CUDA
    kernel launches and cuMemcpyAsync calls.
    """

    # Simulated timing constants (ms) — calibrated to RTX A4000 + Mistral-7B
    PREFILL_MS_PER_TOKEN  = 0.05    # ~20k tokens/s prefill throughput
    DECODE_MS_PER_TOKEN   = 1.4     # matches paper TPOT baseline
    MIGRATION_MS_PER_BLOCK = 0.08   # PCIe DMA ~80 µs / block (16 tokens, FP16)
    TOKENS_PER_BLOCK      = 16

    def __init__(self,
                 profiler: OnlineProfiler,
                 t1_capacity_blocks: int = 25,
                 t2_capacity_blocks: int = 512,
                 soar_threshold: float = 0.85,
                 alto_threshold: float = 0.40):

        self.profiler = profiler
        self.t1_cap   = t1_capacity_blocks
        self.t2_cap   = t2_capacity_blocks
        self.P        = soar_threshold
        self.theta    = alto_threshold

        # Block pools (simulated)
        self._t1_used   = 0
        self._t2_used   = 0
        self._block_tier: dict = {}        # block_id → Tier
        self._block_ref:  dict = {}        # block_id → ref_count
        self._next_block_id = 0
        self._pool_lock = asyncio.Lock()

        # Queues
        self.prefill_queue   = asyncio.Queue()
        self.decode_queue    = asyncio.Queue()
        self.migration_queue = asyncio.Queue()

        # Stats
        self.completed: List[InferenceRequest] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit(self, req: InferenceRequest):
        req.t_queued = time.time_ns()
        await self.prefill_queue.put(req)

    async def run(self, max_requests: Optional[int] = None):
        """Start all three workers concurrently."""
        self._running = True
        tasks = [
            asyncio.create_task(self._prefill_worker(max_requests)),
            asyncio.create_task(self._decode_worker()),
            asyncio.create_task(self._migration_worker()),
        ]
        await asyncio.gather(*tasks)

    async def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Worker 1: Prefill  (Stream 1 in CUDA)
    # ------------------------------------------------------------------
    async def _prefill_worker(self, max_requests: Optional[int]):
        processed = 0
        while self._running:
            try:
                req = await asyncio.wait_for(self.prefill_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break

            req.phase = RequestPhase.PREFILL
            req.t_prefill_start = time.time_ns()
            profiler = self.profiler
            profiler.request_started()

            # Allocate blocks
            num_blocks = max(1, req.prompt_tokens // self.TOKENS_PER_BLOCK)
            async with self._pool_lock:
                block_ids = self._allocate_blocks(num_blocks)
            req.block_ids = block_ids

            # Register & record prefill accesses
            for bid in block_ids:
                profiler.register_block(bid, self._block_tier.get(bid, Tier.T1))
                profiler.on_block_access(bid, Phase.PREFILL,
                                          self._block_tier.get(bid, Tier.T1))

            # Simulate prefill compute time
            prefill_ms = req.prompt_tokens * self.PREFILL_MS_PER_TOKEN
            await asyncio.sleep(prefill_ms / 1_000)

            req.t_prefill_end = time.time_ns()
            ttft = (req.t_prefill_end - req.t_prefill_start) / 1e6
            req.ttft_ms = ttft
            profiler.record_ttft(ttft)
            profiler.record_token_generated()

            # Enqueue migration (overlaps with next prefill)
            await self.migration_queue.put(req)
            logger.info(f"[PREFILL] req={req.request_id} tokens={req.prompt_tokens} "
                        f"blocks={len(block_ids)} TTFT={ttft:.1f}ms")

            self.prefill_queue.task_done()
            processed += 1
            if max_requests and processed >= max_requests:
                break

    # ------------------------------------------------------------------
    # Worker 2: Migration  (Stream 2 / DMA in CUDA)
    # ------------------------------------------------------------------
    async def _migration_worker(self):
        """
        Phase-aware promotion (Algorithm 4): promote T2 blocks to T1 while
        the prefill worker is busy with the next request.
        This is the KEY parallelism: DMA on stream 2 overlaps prefill on stream 1.
        """
        while self._running:
            try:
                req = await asyncio.wait_for(self.migration_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break

            req.phase = RequestPhase.MIGRATING
            t2_blocks = [bid for bid in req.block_ids
                         if self._block_tier.get(bid) == Tier.T2]

            promoted = 0
            for bid in sorted(t2_blocks):  # layer-order: ascending block_id ≈ layer idx
                async with self._pool_lock:
                    ok = self._try_promote(bid)
                if ok:
                    # Simulate DMA transfer (overlapped with prefill worker)
                    await asyncio.sleep(self.MIGRATION_MS_PER_BLOCK / 1_000)
                    self.profiler.update_tier(bid, Tier.T1)
                    logger.debug(f"[MIGRATION] promoted block {bid} T2→T1")
                    promoted += 1

            if promoted:
                logger.info(f"[MIGRATION] req={req.request_id} promoted {promoted} blocks T2→T1")

            await self.decode_queue.put(req)
            self.migration_queue.task_done()

    # ------------------------------------------------------------------
    # Worker 3: Decode  (Stream 0 in CUDA)
    # ------------------------------------------------------------------
    async def _decode_worker(self):
        while self._running:
            try:
                req = await asyncio.wait_for(self.decode_queue.get(), timeout=3.0)
            except asyncio.TimeoutError:
                break

            req.phase = RequestPhase.DECODE
            req.t_decode_start = time.time_ns()

            tpot_sum = 0.0
            for step in range(req.max_new_tokens):
                # Record decode accesses — all blocks should be T1 now
                for bid in req.block_ids:
                    tier = self._block_tier.get(bid, Tier.T1)
                    self.profiler.on_block_access(bid, Phase.DECODE, tier)

                step_ms = self.DECODE_MS_PER_TOKEN
                # If any block still in T2, add PCIe penalty
                t2_count = sum(1 for b in req.block_ids
                               if self._block_tier.get(b) == Tier.T2)
                if t2_count:
                    step_ms += t2_count * self.MIGRATION_MS_PER_BLOCK

                await asyncio.sleep(step_ms / 1_000)
                self.profiler.record_token_generated()
                tpot_sum += step_ms

            req.t_decode_end = time.time_ns()
            total_decode_ms = (req.t_decode_end - req.t_decode_start) / 1e6
            req.tpot_ms = tpot_sum / max(req.max_new_tokens, 1)
            req.throughput_tps = req.max_new_tokens / (total_decode_ms / 1_000)

            self.profiler.record_tpot(req.tpot_ms)
            req.phase = RequestPhase.DONE
            self.completed.append(req)
            self.profiler.request_finished()

            # Release blocks
            async with self._pool_lock:
                self._free_blocks(req.block_ids)

            logger.info(f"[DECODE]  req={req.request_id} tokens={req.max_new_tokens} "
                        f"TPOT={req.tpot_ms:.2f}ms TPS={req.throughput_tps:.1f}")
            self.decode_queue.task_done()

    # ------------------------------------------------------------------
    # Block pool helpers (simulated)
    # ------------------------------------------------------------------
    def _allocate_blocks(self, n: int) -> List[int]:
        ids = []
        for _ in range(n):
            bid = self._next_block_id
            self._next_block_id += 1
            if self._t1_used < self._t1_cap:
                tier = Tier.T1
                self._t1_used += 1
            elif self._t2_used < self._t2_cap:
                tier = Tier.T2
                self._t2_used += 1
            else:
                logger.warning("OOM: no free blocks!")
                break
            self._block_tier[bid] = tier
            self._block_ref[bid]  = 0
            ids.append(bid)
        return ids

    def _try_promote(self, bid: int) -> bool:
        """Promote a T2 block to T1 if space allows, or evict coldest T1 block."""
        if self._block_tier.get(bid) != Tier.T2:
            return False
        aol = self.profiler.get_aol_score(bid)
        if aol < self.theta:
            return False
        if self._t1_used < self._t1_cap:
            self._block_tier[bid] = Tier.T1
            self._t1_used += 1
            self._t2_used -= 1
            return True
        # Evict coldest T1 block (ALTO swap)
        scores = self.profiler.get_t1_blocks_sorted_by_aol()
        for victim_id, victim_aol in scores:
            if self._block_ref.get(victim_id, 0) == 0 and victim_aol < aol:
                self._block_tier[victim_id] = Tier.T2
                self._block_tier[bid]       = Tier.T1
                return True
        return False

    def _free_blocks(self, block_ids: List[int]):
        for bid in block_ids:
            tier = self._block_tier.pop(bid, None)
            if tier == Tier.T1:
                self._t1_used = max(0, self._t1_used - 1)
            elif tier == Tier.T2:
                self._t2_used = max(0, self._t2_used - 1)
            self._block_ref.pop(bid, None)
            self.profiler.deregister_block(bid)
