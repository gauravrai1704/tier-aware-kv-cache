/**
 * migration_orchestrator.h
 */
#pragma once

#include "tiered_block_allocator.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>
#include <string>

namespace tierkv {

enum class RequestPhase { PREFILL, DECODE, DONE };

// Stub bridge to Rust AOL profiler FFI
struct AOLProfilerBridge {
    virtual float              get_score(uint64_t block_id) = 0;
    virtual std::vector<std::pair<uint64_t,double>> get_all_scores() = 0;
    virtual ~AOLProfilerBridge() = default;
};

struct OrchestratorConfig {
    uint32_t sweep_interval_ms      = 200;    // kdemoted sweep period
    size_t   demotion_batch_size    = 32;     // SOAR batch
    float    alto_promote_threshold = 0.4f;   // min AOL to bother promoting
    bool     llm_aware_policy       = true;   // decode-phase preemption
};

struct OrchestratorStats {
    uint64_t total_demotions;
    uint64_t total_promotions;
    uint64_t remote_accesses;  // T2 accesses skipped promotion
};

class MigrationOrchestrator {
public:
    MigrationOrchestrator(
        TieredBlockAllocator*     allocator,
        AOLProfilerBridge*        profiler,
        const OrchestratorConfig& cfg = OrchestratorConfig{});
    ~MigrationOrchestrator();

    void start();
    void stop();

    void notify_phase_change(const std::string& request_id, RequestPhase phase);
    void notify_request_done(const std::string& request_id);
    bool on_demand_promote(uint64_t block_id);

    OrchestratorStats stats() const;

private:
    void   _daemon_loop();
    size_t _run_soar_sweep(size_t max_demote);
    void   _promote_for_request(const std::string& request_id);

    TieredBlockAllocator*  _allocator;
    AOLProfilerBridge*     _profiler;
    OrchestratorConfig     _cfg;
    std::atomic<bool>      _running;
    std::thread            _daemon;

    std::mutex                     _phase_mutex;
    std::unordered_set<std::string> _decode_requests;

    std::atomic<uint64_t> _stat_demotions{0};
    std::atomic<uint64_t> _stat_promotions{0};
    std::atomic<uint64_t> _stat_remote_accesses{0};
};

} // namespace tierkv
