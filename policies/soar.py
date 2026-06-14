"""
policies/soar.py
----------------
SOAR: Sweep Offcore And Reassign — background demotion daemon.
Extended from Algorithm 2 in the paper to integrate with OnlineProfiler.
"""
import asyncio
import logging
from profiling.online_profiler import OnlineProfiler, Tier

logger = logging.getLogger(__name__)

class SOARDaemon:
    """Background daemon that sweeps T1 every sweep_interval_ms milliseconds."""

    def __init__(self, profiler: OnlineProfiler,
                 t1_capacity_blocks: int,
                 pressure_threshold: float = 0.85,
                 sweep_interval_ms: float = 200.0):
        self.profiler   = profiler
        self.t1_cap     = t1_capacity_blocks
        self.P          = pressure_threshold
        self.interval   = sweep_interval_ms / 1_000
        self._t1_used   = 0
        self._running   = False
        self.demotions  = 0

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval)
            await self._sweep()

    async def stop(self):
        self._running = False

    async def _sweep(self):
        util = self._t1_used / max(self.t1_cap, 1)
        if util < self.P:
            return
        target = int((util - self.P) * self.t1_cap)
        sorted_blocks = self.profiler.get_t1_blocks_sorted_by_aol()
        freed = 0
        for bid, aol in sorted_blocks:
            if freed >= target:
                break
            self.profiler.update_tier(bid, Tier.T2)
            freed += 1
            self.demotions += 1
        if freed:
            self._t1_used = max(0, self._t1_used - freed)
            logger.info(f"[SOAR] Demoted {freed} blocks (util was {util:.2f})")
