/**
 * tiered_block_allocator.h
 * Tier-Aware KV Cache — Block Allocator
 *
 * Manages the mapping between virtual KV cache addresses and physical
 * frames spread across Tier-1 (DRAM/HBM) and Tier-2 (CXL/NVMe).
 * Uses CUDA Virtual Memory Management (VMM) APIs so that attention
 * kernels always see a contiguous virtual address space.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <optional>
#include <atomic>
#include <string>
#include <functional>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#else
// Stub CUDA types so the header compiles without the CUDA toolkit
using CUdeviceptr = uint64_t;
#endif

namespace tierkv {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
struct PhysicalFrame;
struct LogicalBlock;
class  AOLProfiler;   // defined in aol_profiler.h

// ---------------------------------------------------------------------------
// Tier identifiers
// ---------------------------------------------------------------------------
enum class MemoryTier : uint8_t {
    TIER1_FAST = 0,   ///< GPU HBM or local DRAM
    TIER2_SLOW = 1,   ///< CXL-attached memory or NVMe (via DAX)
    UNALLOCATED = 2,
};

// ---------------------------------------------------------------------------
// Physical frame descriptor
// ---------------------------------------------------------------------------
struct PhysicalFrame {
    uint64_t    frame_id;
    MemoryTier  tier;
    void*       host_ptr;   ///< host-side mapped address (CXL/NVMe)
    CUdeviceptr device_ptr; ///< GPU virtual address (TIER1 only)
    size_t      size_bytes;
    bool        in_use;

    PhysicalFrame() = default;
    PhysicalFrame(uint64_t id, MemoryTier t, size_t sz)
        : frame_id(id), tier(t), host_ptr(nullptr),
          device_ptr(0), size_bytes(sz), in_use(false) {}
};

// ---------------------------------------------------------------------------
// Logical block: the unit of KV cache management
// ---------------------------------------------------------------------------
struct LogicalBlock {
    uint64_t    block_id;
    std::string request_id;
    int         layer_idx;
    uint32_t    token_start;
    uint32_t    token_end;
    int         ref_count;
    float       aol_score;      ///< updated by AOL Profiler
    MemoryTier  current_tier;
    uint64_t    physical_frame_id;
    bool        is_shared;      ///< CoW flag

    LogicalBlock() = default;
};

// ---------------------------------------------------------------------------
// Allocation configuration
// ---------------------------------------------------------------------------
struct AllocatorConfig {
    size_t tier1_capacity_bytes;  ///< fast-tier pool size
    size_t tier2_capacity_bytes;  ///< slow-tier pool size
    size_t block_size_tokens;     ///< tokens per logical block
    int    num_layers;
    int    num_heads;
    int    head_dim;
    int    gpu_device_id;
    float  tier1_pressure_threshold; ///< 0..1, trigger migration above this
    bool   enable_cow_sharing;
    bool   enable_numa_aware;
    std::string tier2_path;       ///< mount path for NVMe DAX FS (optional)

    AllocatorConfig()
        : tier1_capacity_bytes(4ULL << 30),
          tier2_capacity_bytes(32ULL << 30),
          block_size_tokens(16),
          num_layers(32), num_heads(8), head_dim(128),
          gpu_device_id(0),
          tier1_pressure_threshold(0.85f),
          enable_cow_sharing(true),
          enable_numa_aware(true),
          tier2_path("/dev/dax0.0") {}
};

// ---------------------------------------------------------------------------
// AllocatorStats — returned for telemetry
// ---------------------------------------------------------------------------
struct AllocatorStats {
    size_t   tier1_used_bytes;
    size_t   tier1_cap_bytes;
    size_t   tier2_used_bytes;
    size_t   tier2_cap_bytes;
    uint64_t total_blocks;
    uint64_t tier1_blocks;
    uint64_t tier2_blocks;
    uint64_t promotions;
    uint64_t demotions;
    uint64_t cow_shares;
};

// ---------------------------------------------------------------------------
// TieredBlockAllocator
// ---------------------------------------------------------------------------
class TieredBlockAllocator {
public:
    explicit TieredBlockAllocator(const AllocatorConfig& cfg);
    ~TieredBlockAllocator();

    // Disable copy
    TieredBlockAllocator(const TieredBlockAllocator&)            = delete;
    TieredBlockAllocator& operator=(const TieredBlockAllocator&) = delete;

    // -----------------------------------------------------------------------
    // Core allocation API
    // -----------------------------------------------------------------------

    /**
     * Reserve a contiguous virtual address range for the entire KV cache.
     * Physical backing is NOT committed yet (lazy allocation / vAttention style).
     *
     * @param total_tokens  Maximum sequence length expected.
     * @return              GPU virtual base address of reserved range.
     */
    CUdeviceptr reserve_virtual_range(size_t total_tokens);

    /**
     * Allocate one logical block and map it to a physical frame.
     * Chooses T1 or T2 based on current pressure.
     */
    std::optional<uint64_t> allocate_block(
        const std::string& request_id,
        int                layer_idx,
        uint32_t           token_start,
        uint32_t           token_end
    );

    /**
     * Free a single logical block; decrements ref_count,
     * physically unmaps if ref_count reaches 0.
     */
    bool free_block(uint64_t block_id);

    /**
     * Free all blocks belonging to request_id.
     */
    void free_request(const std::string& request_id);

    // -----------------------------------------------------------------------
    // Copy-on-Write sharing
    // -----------------------------------------------------------------------

    /**
     * Share a block between two requests (beam search / parallel sampling).
     * Increments ref_count; actual copy deferred until write.
     */
    bool cow_share(uint64_t block_id, const std::string& new_owner_id);

    /**
     * Materialise a private copy for new_owner_id if block is shared.
     */
    uint64_t cow_copy_on_write(uint64_t block_id,
                                const std::string& new_owner_id);

    // -----------------------------------------------------------------------
    // Tier migration
    // -----------------------------------------------------------------------

    /** Move block from T2 → T1 (demand-based promotion). */
    bool promote(uint64_t block_id);

    /** Move block from T1 → T2 (background demotion). */
    bool demote(uint64_t block_id);

    /**
     * Run one sweep of the demotion policy.
     * Typically called by the Migration Orchestrator daemon.
     * @param n_to_demote  Max number of blocks to demote in one call.
     * @return             Number of blocks actually demoted.
     */
    size_t sweep_demotion(size_t n_to_demote = 32);

    // -----------------------------------------------------------------------
    // AOL integration
    // -----------------------------------------------------------------------

    /** Called by AOLProfiler to push an updated criticality score. */
    void update_aol_score(uint64_t block_id, float score);

    /** Batch AOL update — more efficient for the profiler. */
    void batch_update_aol(
        const std::vector<uint64_t>& block_ids,
        const std::vector<float>&    scores
    );

    // -----------------------------------------------------------------------
    // Accessors / telemetry
    // -----------------------------------------------------------------------

    LogicalBlock*  get_block(uint64_t block_id);
    AllocatorStats stats() const;
    bool           tier1_under_pressure() const;

    /**
     * Register a callback invoked each time a demotion occurs.
     * Useful for telemetry dashboards.
     */
    void on_demotion(std::function<void(uint64_t, MemoryTier)> cb) {
        _demotion_callback = std::move(cb);
    }

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    bool   _init_cuda_vmm();
    bool   _init_tier1_pool();
    bool   _init_tier2_pool();
    void*  _mmap_cxl_or_nvme(size_t bytes);
    bool   _map_physical_to_virtual(uint64_t block_id);
    bool   _unmap_physical(uint64_t block_id);
    bool   _copy_tier1_to_tier2(uint64_t block_id);
    bool   _copy_tier2_to_tier1(uint64_t block_id);
    size_t _bytes_per_block() const;
    PhysicalFrame* _alloc_frame_tier1();
    PhysicalFrame* _alloc_frame_tier2();
    void   _free_frame(PhysicalFrame* frame);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    AllocatorConfig _cfg;
    mutable std::mutex _mutex;

    // Virtual address range (CUDA VMM)
    CUdeviceptr _va_base{0};
    size_t      _va_size{0};

    // Physical frame pools
    std::vector<std::unique_ptr<PhysicalFrame>> _tier1_frames;
    std::vector<std::unique_ptr<PhysicalFrame>> _tier2_frames;

    // Free-frame stacks (indices into pools above)
    std::vector<PhysicalFrame*> _tier1_free;
    std::vector<PhysicalFrame*> _tier2_free;

    // Logical block registry
    std::unordered_map<uint64_t, LogicalBlock>              _blocks;
    std::unordered_map<std::string, std::vector<uint64_t>>  _request_blocks;

    // Counters
    std::atomic<uint64_t> _next_block_id{0};
    std::atomic<uint64_t> _stat_promotions{0};
    std::atomic<uint64_t> _stat_demotions{0};
    std::atomic<uint64_t> _stat_cow_shares{0};

    // Optional demotion callback
    std::function<void(uint64_t, MemoryTier)> _demotion_callback;
};

} // namespace tierkv
