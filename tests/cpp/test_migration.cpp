/**
 * tests/cpp/test_migration.cpp
 * C++ unit tests for MigrationOrchestrator
 */

#include "migration_orchestrator.h"
#include "tiered_block_allocator.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <stdexcept>

using namespace tierkv;

// ---------------------------------------------------------------------------
// Minimal test harness (same as test_allocator.cpp)
// ---------------------------------------------------------------------------
static int _pass = 0, _fail = 0;

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
// Null AOL profiler bridge (returns score=1.0 for all blocks)
// ---------------------------------------------------------------------------
struct NullProfiler : AOLProfilerBridge {
    float get_score(uint64_t) override { return 1.0f; }
    std::vector<std::pair<uint64_t,double>> get_all_scores() override { return {}; }
};

// ---------------------------------------------------------------------------
// Low-score profiler (marks every block as cold → easy to demote)
// ---------------------------------------------------------------------------
struct ColdProfiler : AOLProfilerBridge {
    float score;
    explicit ColdProfiler(float s = 0.01f) : score(s) {}
    float get_score(uint64_t) override { return score; }
    std::vector<std::pair<uint64_t,double>> get_all_scores() override { return {}; }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static AllocatorConfig test_cfg() {
    AllocatorConfig cfg;
    cfg.tier1_capacity_bytes      = 32ULL  * 1024 * 1024;
    cfg.tier2_capacity_bytes      = 256ULL * 1024 * 1024;
    cfg.block_size_tokens         = 16;
    cfg.num_layers                = 2;
    cfg.num_heads                 = 2;
    cfg.head_dim                  = 64;
    cfg.tier1_pressure_threshold  = 0.85f;
    cfg.enable_cow_sharing        = true;
    cfg.enable_numa_aware         = false;
    return cfg;
}

static OrchestratorConfig fast_cfg() {
    OrchestratorConfig cfg;
    cfg.sweep_interval_ms      = 50;    // fast sweep for tests
    cfg.demotion_batch_size    = 8;
    cfg.alto_promote_threshold = 0.4f;
    cfg.llm_aware_policy       = true;
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(test_orchestrator_starts_and_stops) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    orch.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    orch.stop();
    // No crash = pass
}

TEST(test_stats_initially_zero) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    auto s = orch.stats();
    ASSERT(s.total_demotions  == 0);
    ASSERT(s.total_promotions == 0);
    ASSERT(s.remote_accesses  == 0);
}

TEST(test_on_demand_promote_missing_block) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    // Block 99999 doesn't exist; promote should return false gracefully
    bool ok = orch.on_demand_promote(99999);
    ASSERT(!ok);
}

TEST(test_on_demand_promote_cold_block_skipped) {
    // If AOL score < alto_promote_threshold, promotion is skipped (remote access)
    TieredBlockAllocator alloc(test_cfg());
    ColdProfiler         profiler(0.01f);  // score way below 0.4 threshold
    OrchestratorConfig cfg = fast_cfg();
    cfg.alto_promote_threshold = 0.4f;
    MigrationOrchestrator orch(&alloc, &profiler, cfg);

    // Allocate a block
    auto bid = alloc.allocate_block("req-1", 0, 0, 16);
    ASSERT(bid.has_value());

    // Manually demote it to T2 so promote has something to do
    alloc.demote(*bid);

    // With cold profiler, on_demand_promote should skip (remote access path)
    bool promoted = orch.on_demand_promote(*bid);
    ASSERT(!promoted);  // cold score → remote access, no promotion

    auto s = orch.stats();
    ASSERT(s.remote_accesses >= 1);
}

TEST(test_notify_phase_change_no_crash) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    orch.start();
    orch.notify_phase_change("req-X", RequestPhase::DECODE);
    orch.notify_phase_change("req-X", RequestPhase::DONE);
    orch.notify_request_done("req-X");
    orch.stop();
}

TEST(test_notify_prefill_phase_is_noop) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    // PREFILL phase change should not trigger promotions
    orch.notify_phase_change("req-1", RequestPhase::PREFILL);
    auto s = orch.stats();
    ASSERT(s.total_promotions == 0);
}

TEST(test_daemon_runs_without_error) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
    orch.start();

    // Allocate some blocks while daemon is running
    for (int i = 0; i < 10; ++i)
        alloc.allocate_block("req-1", 0, i*16, (i+1)*16);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    orch.stop();
    // No crash or deadlock = pass
}

TEST(test_multiple_start_stop_cycles) {
    TieredBlockAllocator alloc(test_cfg());
    NullProfiler         profiler;
    // Create and destroy multiple orchestrators (checks no resource leak)
    for (int i = 0; i < 3; ++i) {
        MigrationOrchestrator orch(&alloc, &profiler, fast_cfg());
        orch.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        orch.stop();
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "\nMigrationOrchestrator — C++ Tests\n";
    std::cout << "===================================\n";

    RUN(test_orchestrator_starts_and_stops);
    RUN(test_stats_initially_zero);
    RUN(test_on_demand_promote_missing_block);
    RUN(test_on_demand_promote_cold_block_skipped);
    RUN(test_notify_phase_change_no_crash);
    RUN(test_notify_prefill_phase_is_noop);
    RUN(test_daemon_runs_without_error);
    RUN(test_multiple_start_stop_cycles);

    std::cout << "\n-----------------------------------\n";
    std::cout << "Ran " << (_pass + _fail) << " tests: "
              << _pass << " passed, " << _fail << " failed.\n";
    return (_fail > 0) ? 1 : 0;
}
