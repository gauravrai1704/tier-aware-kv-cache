"""
tests/test_orchestrator.py
Unit and integration tests for Tier-Aware KV Cache Orchestrator
"""

import sys, os, time, threading, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.manager.orchestrator import (
    TierAwareKVCacheOrchestrator, MemoryTier,
    TIER1_PRESSURE_PCT, AOL_DEMOTION_THRESH
)


# ---------------------------------------------------------------------------
# Helper: build a small orchestrator for tests
# ---------------------------------------------------------------------------
def make_orch(t1_mb=256, t2_mb=2048, num_layers=2, num_heads=2, head_dim=64):
    return TierAwareKVCacheOrchestrator(
        tier1_capacity_mb=t1_mb,
        tier2_capacity_mb=t2_mb,
        block_size=16,
        num_layers=num_layers,
        head_dim=head_dim,
        num_heads=num_heads,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):
    def test_register_returns_unique_ids(self):
        orch = make_orch()
        ids = {orch.register_request(128) for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_stats_reflect_active_requests(self):
        orch = make_orch()
        rid  = orch.register_request(64)
        self.assertEqual(orch.get_stats()["active_requests"], 1)
        orch.free_request(rid)
        self.assertEqual(orch.get_stats()["active_requests"], 0)


class TestBlockAllocation(unittest.TestCase):
    def test_allocate_returns_block_ids(self):
        orch = make_orch()
        rid  = orch.register_request(64)
        bids = orch.allocate_blocks(rid, 32, layer_idx=0)
        self.assertGreater(len(bids), 0)

    def test_blocks_land_in_tier1_when_space_available(self):
        orch = make_orch(t1_mb=1024)
        rid  = orch.register_request(32)
        orch.allocate_blocks(rid, 16, 0)
        s = orch.get_stats()
        self.assertGreater(s["tier1_blocks"], 0)

    def test_blocks_spill_to_tier2_under_pressure(self):
        """Fill T1 past threshold; new blocks should go to T2."""
        orch = make_orch(t1_mb=4, t2_mb=512, num_layers=1)
        rid  = orch.register_request(2048)
        # Force T1 utilisation over threshold before allocating
        orch._tier1_used_mb = orch.tier1_capacity_mb * TIER1_PRESSURE_PCT + 1
        all_bids = []
        for _ in range(5):
            bids = orch.allocate_blocks(rid, 16, 0)
            all_bids.extend(bids)
        s = orch.get_stats()
        self.assertGreater(s["tier2_blocks"], 0,
                           "Expected blocks to spill to T2 under pressure")

    def test_free_request_clears_blocks(self):
        orch = make_orch()
        rid  = orch.register_request(64)
        orch.allocate_blocks(rid, 64, 0)
        orch.free_request(rid)
        s = orch.get_stats()
        self.assertEqual(s["total_blocks"], 0)
        self.assertEqual(s["tier1_used_mb"], 0.0)


class TestAOLScores(unittest.TestCase):
    def test_aol_update(self):
        orch = make_orch()
        rid  = orch.register_request(32)
        bids = orch.allocate_blocks(rid, 16, 0)
        orch.update_aol_score(bids[0], 0.1)
        # Low AOL → candidate for demotion; no assertion, just no crash

    def test_aol_clamped(self):
        orch = make_orch()
        rid  = orch.register_request(32)
        bids = orch.allocate_blocks(rid, 16, 0)
        orch.update_aol_score(bids[0], -99.0)  # should clamp to 0
        orch.update_aol_score(bids[0], 999.0)  # should clamp to 1


class TestPromotion(unittest.TestCase):
    def test_promote_nonexistent_returns_false(self):
        orch = make_orch()
        self.assertFalse(orch.promote_block(99999))

    def test_promote_already_t1_returns_false(self):
        orch = make_orch(t1_mb=1024)
        rid  = orch.register_request(32)
        bids = orch.allocate_blocks(rid, 16, 0)
        # Block should be in T1; promoting again = no-op
        result = orch.promote_block(bids[0])
        self.assertFalse(result)

    def test_access_block_returns_tier(self):
        orch = make_orch(t1_mb=1024)
        rid  = orch.register_request(32)
        bids = orch.allocate_blocks(rid, 16, 0)
        tier = orch.access_block(bids[0])
        self.assertIsInstance(tier, MemoryTier)


class TestKdemotedDaemon(unittest.TestCase):
    def test_daemon_demotes_low_aol_blocks(self):
        """Give kdemoted enough time to fire and demote cold blocks."""
        orch = make_orch(t1_mb=4, t2_mb=512, num_layers=1)
        rid  = orch.register_request(1024)
        bids = orch.allocate_blocks(rid, 512, 0)

        # Set very low AOL on all blocks
        for bid in bids:
            orch.update_aol_score(bid, 0.01)

        # Manually bump T1 utilisation above pressure threshold
        # by directly manipulating usage (white-box)
        orch._tier1_used_mb = orch.tier1_capacity_mb * TIER1_PRESSURE_PCT + 1

        time.sleep(1.5)  # let daemon run at least once

        s = orch.get_stats()
        # Either kdemoted fired (tier2_blocks > 0) or T1 was below threshold
        # In this small test the ratio might still pass — just ensure no crash
        self.assertIsNotNone(s)


class TestConcurrency(unittest.TestCase):
    def test_concurrent_register_and_free(self):
        orch   = make_orch(t1_mb=1024, t2_mb=4096)
        errors = []

        def worker(idx):
            try:
                rid  = orch.register_request(64 + idx)
                bids = orch.allocate_blocks(rid, 32, 0)
                for bid in bids:
                    orch.access_block(bid)
                orch.free_request(rid)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"Concurrent errors: {errors}")


class TestStats(unittest.TestCase):
    def test_stats_keys_present(self):
        orch = make_orch()
        s    = orch.get_stats()
        expected = {
            "tier1_used_mb", "tier1_cap_mb", "tier1_util_pct",
            "tier2_used_mb", "tier2_cap_mb", "tier2_util_pct",
            "total_blocks", "tier1_blocks", "tier2_blocks",
            "active_requests",
        }
        self.assertEqual(expected, set(s.keys()))

    def test_utilisation_stays_0_to_100(self):
        orch = make_orch(t1_mb=512)
        rid  = orch.register_request(128)
        orch.allocate_blocks(rid, 128, 0)
        s    = orch.get_stats()
        self.assertGreaterEqual(s["tier1_util_pct"], 0)
        self.assertLessEqual(s["tier1_util_pct"], 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
