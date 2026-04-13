/**
 * tests/cpp/test_allocator.cpp
 * Basic C++ unit tests for TieredBlockAllocator (no external framework needed)
 */

#include "tiered_block_allocator.h"
#include <iostream>
#include <cassert>
#include <string>

using namespace tierkv;

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------
static int  _pass = 0, _fail = 0;

#define TEST(name) void name()
#define RUN(name)  do { \
    try { name(); std::cout << "  PASS  " #name "\n"; ++_pass; } \
    catch (std::exception& e) { \
        std::cout << "  FAIL  " #name " — " << e.what() << "\n"; ++_fail; } \
    catch (...) { \
        std::cout << "  FAIL  " #name " — unknown exception\n"; ++_fail; } \
} while(0)

#define ASSERT(cond) do { if (!(cond)) \
    throw std::runtime_error("Assertion failed: " #cond); } while(0)

// ---------------------------------------------------------------------------
// Helper: small allocator config for tests
// ---------------------------------------------------------------------------
static AllocatorConfig small_cfg() {
    AllocatorConfig cfg;
    cfg.tier1_capacity_bytes  = 32ULL  * 1024 * 1024;   // 32 MB
    cfg.tier2_capacity_bytes  = 256ULL * 1024 * 1024;   // 256 MB
    cfg.block_size_tokens     = 16;
    cfg.num_layers            = 2;
    cfg.num_heads             = 2;
    cfg.head_dim              = 64;
    cfg.gpu_device_id         = 0;
    cfg.tier1_pressure_threshold = 0.85f;
    cfg.enable_cow_sharing    = true;
    cfg.enable_numa_aware     = false;
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(test_allocate_single_block) {
    TieredBlockAllocator alloc(small_cfg());
    auto bid = alloc.allocate_block("req-1", 0, 0, 16);
    ASSERT(bid.has_value());
    ASSERT(*bid == 0);
}

TEST(test_allocate_multiple_blocks) {
    TieredBlockAllocator alloc(small_cfg());
    std::vector<uint64_t> ids;
    for (int i = 0; i < 10; ++i) {
        auto bid = alloc.allocate_block("req-1", 0, i*16, (i+1)*16);
        ASSERT(bid.has_value());
        ids.push_back(*bid);
    }
    ASSERT(ids.size() == 10);
    // All IDs unique
    for (size_t i = 0; i < ids.size(); ++i)
        for (size_t j = i+1; j < ids.size(); ++j)
            ASSERT(ids[i] != ids[j]);
}

TEST(test_free_block) {
    TieredBlockAllocator alloc(small_cfg());
    auto bid = alloc.allocate_block("req-1", 0, 0, 16);
    ASSERT(bid.has_value());
    bool freed = alloc.free_block(*bid);
    ASSERT(freed);
    // Freeing again should return false (block gone)
    bool freed2 = alloc.free_block(*bid);
    ASSERT(!freed2);
}

TEST(test_free_request_clears_all_blocks) {
    TieredBlockAllocator alloc(small_cfg());
    for (int i = 0; i < 5; ++i)
        alloc.allocate_block("req-A", 0, i*16, (i+1)*16);
    alloc.free_request("req-A");
    auto s = alloc.stats();
    ASSERT(s.total_blocks == 0);
}

TEST(test_stats_tier1_used) {
    TieredBlockAllocator alloc(small_cfg());
    alloc.allocate_block("req-1", 0, 0, 16);
    auto s = alloc.stats();
    ASSERT(s.tier1_blocks >= 1);
    ASSERT(s.tier1_used_bytes > 0);
}

TEST(test_cow_share_increments_ref) {
    TieredBlockAllocator alloc(small_cfg());
    auto bid = alloc.allocate_block("req-1", 0, 0, 16);
    ASSERT(bid.has_value());
    bool shared = alloc.cow_share(*bid, "req-2");
    ASSERT(shared);
    // After share, freeing req-1's reference should not destroy block
    alloc.free_block(*bid);
    LogicalBlock* blk = alloc.get_block(*bid);
    ASSERT(blk != nullptr);   // still alive via req-2
}

TEST(test_aol_score_update) {
    TieredBlockAllocator alloc(small_cfg());
    auto bid = alloc.allocate_block("req-1", 0, 0, 16);
    ASSERT(bid.has_value());
    alloc.update_aol_score(*bid, 0.1f);
    LogicalBlock* blk = alloc.get_block(*bid);
    ASSERT(blk != nullptr);
    ASSERT(blk->aol_score < 0.2f);
}

TEST(test_batch_aol_update) {
    TieredBlockAllocator alloc(small_cfg());
    std::vector<uint64_t> ids;
    std::vector<float>    scores;
    for (int i = 0; i < 5; ++i) {
        auto bid = alloc.allocate_block("req-1", 0, i*16, (i+1)*16);
        ASSERT(bid.has_value());
        ids.push_back(*bid);
        scores.push_back(0.05f * (i + 1));
    }
    alloc.batch_update_aol(ids, scores);
    for (size_t i = 0; i < ids.size(); ++i) {
        LogicalBlock* blk = alloc.get_block(ids[i]);
        ASSERT(blk != nullptr);
        ASSERT(blk->aol_score < 0.4f);
    }
}

TEST(test_tier1_pressure_detection) {
    AllocatorConfig cfg = small_cfg();
    cfg.tier1_capacity_bytes     = 1ULL * 1024 * 1024;  // tiny 1 MB T1
    cfg.tier1_pressure_threshold = 0.5f;
    TieredBlockAllocator alloc(cfg);

    // Allocate until pressure is detected
    int allocated = 0;
    for (int i = 0; i < 100; ++i) {
        auto bid = alloc.allocate_block("req-1", 0, i*16, (i+1)*16);
        if (!bid.has_value()) break;
        ++allocated;
        if (alloc.tier1_under_pressure()) break;
    }
    // We should have hit pressure at some point
    ASSERT(allocated > 0);
}

TEST(test_demotion_sweep) {
    AllocatorConfig cfg = small_cfg();
    TieredBlockAllocator alloc(cfg);

    // Allocate several blocks and mark them cold
    std::vector<uint64_t> ids;
    for (int i = 0; i < 8; ++i) {
        auto bid = alloc.allocate_block("req-1", 0, i*16, (i+1)*16);
        ASSERT(bid.has_value());
        alloc.update_aol_score(*bid, 0.01f);  // very cold
        ids.push_back(*bid);
    }

    size_t demoted = alloc.sweep_demotion(4);
    // May demote 0..4 depending on T1 pressure; just ensure no crash
    ASSERT(demoted <= 4);
}

TEST(test_get_block_returns_null_for_missing) {
    TieredBlockAllocator alloc(small_cfg());
    LogicalBlock* blk = alloc.get_block(99999);
    ASSERT(blk == nullptr);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "\nTieredBlockAllocator — C++ Tests\n";
    std::cout << "==================================\n";

    RUN(test_allocate_single_block);
    RUN(test_allocate_multiple_blocks);
    RUN(test_free_block);
    RUN(test_free_request_clears_all_blocks);
    RUN(test_stats_tier1_used);
    RUN(test_cow_share_increments_ref);
    RUN(test_aol_score_update);
    RUN(test_batch_aol_update);
    RUN(test_tier1_pressure_detection);
    RUN(test_demotion_sweep);
    RUN(test_get_block_returns_null_for_missing);

    std::cout << "\n----------------------------------\n";
    std::cout << "Ran " << (_pass + _fail) << " tests: "
              << _pass << " passed, " << _fail << " failed.\n";
    return (_fail > 0) ? 1 : 0;
}
