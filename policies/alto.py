"""
policies/alto.py
----------------
ALTO: AOL-Gated Load-Time Promotion — Algorithm 3 from the paper.
"""
import logging
from profiling.online_profiler import OnlineProfiler, Tier

logger = logging.getLogger(__name__)

class ALTOPromoter:
    def __init__(self, profiler: OnlineProfiler,
                 t1_capacity_blocks: int,
                 theta: float = 0.4):
        self.profiler = profiler
        self.t1_cap   = t1_capacity_blocks
        self.theta    = theta
        self._t1_used = 0
        self.promotions = 0

    def try_promote(self, block_id: int) -> bool:
        aol = self.profiler.get_aol_score(block_id)
        if aol < self.theta:
            return False
        if self._t1_used < self.t1_cap:
            self.profiler.update_tier(block_id, Tier.T1)
            self._t1_used += 1
            self.promotions += 1
            logger.debug(f"[ALTO] Promoted block {block_id} aol={aol:.3f}")
            return True
        # Evict coldest T1 block if candidate is hotter
        t1_sorted = self.profiler.get_t1_blocks_sorted_by_aol()
        if t1_sorted and t1_sorted[0][1] < aol:
            victim_id = t1_sorted[0][0]
            self.profiler.update_tier(victim_id, Tier.T2)
            self.profiler.update_tier(block_id,  Tier.T1)
            self.promotions += 1
            logger.debug(f"[ALTO] Swapped block {block_id} in, {victim_id} out")
            return True
        return False
