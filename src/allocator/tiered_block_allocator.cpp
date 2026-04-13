/**
 * tiered_block_allocator.cpp
 * Tier-Aware KV Cache — Block Allocator Implementation
 *
 * Implements CUDA VMM-based virtual contiguity + multi-tier physical backing.
 */

#include "tiered_block_allocator.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    do { \
        CUresult _r = (call); \
        if (_r != CUDA_SUCCESS) { \
            const char* msg; cuGetErrorString(_r, &msg); \
            throw std::runtime_error(std::string("CUDA: ") + msg); \
        } \
    } while (0)
#else
// Stubs for non-CUDA build
using CUdeviceptr = uint64_t;
#define CUDA_CHECK(call) (void)(call)
#endif

namespace tierkv {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

TieredBlockAllocator::TieredBlockAllocator(const AllocatorConfig& cfg)
    : _cfg(cfg)
{
    if (!_init_tier1_pool()) {
        throw std::runtime_error("Failed to initialise Tier-1 pool");
    }
    if (!_init_tier2_pool()) {
        throw std::runtime_error("Failed to initialise Tier-2 pool");
    }
#ifdef USE_CUDA
    if (!_init_cuda_vmm()) {
        throw std::runtime_error("Failed to initialise CUDA VMM");
    }
    // Auto-reserve VA range large enough for all T1+T2 blocks
    {
        size_t total_bytes = cfg.tier1_capacity_bytes + cfg.tier2_capacity_bytes;
        size_t total_tokens = (total_bytes / 2) / cfg.block_size_tokens;
        reserve_virtual_range(total_tokens);
    }
#endif
    std::cout << "[TieredBlockAllocator] Initialised | "
              << "T1=" << (cfg.tier1_capacity_bytes >> 20) << "MB "
              << "T2=" << (cfg.tier2_capacity_bytes >> 20) << "MB\n";
}

TieredBlockAllocator::~TieredBlockAllocator() {
#ifdef USE_CUDA
    if (_va_base) {
        cuMemUnmap(_va_base, _va_size);
        cuMemAddressFree(_va_base, _va_size);
    }
#endif
}

// ---------------------------------------------------------------------------
// Virtual address reservation (vAttention-style)
// ---------------------------------------------------------------------------

CUdeviceptr TieredBlockAllocator::reserve_virtual_range(size_t total_tokens) {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lk(_mutex);
    size_t sz = _bytes_per_block() *
                ((total_tokens + _cfg.block_size_tokens - 1) / _cfg.block_size_tokens);
    // Align to 2 MB
    sz = (sz + (2 << 20) - 1) & ~((2 << 20) - 1);

    CUmemAllocationProp prop{};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type    = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id      = _cfg.gpu_device_id;

    size_t granularity = 0;
    CUDA_CHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    sz = (sz + granularity - 1) & ~(granularity - 1);

    CUDA_CHECK(cuMemAddressReserve(&_va_base, sz, 0, 0, 0));
    _va_size = sz;
    return _va_base;
#else
    // CPU-only stub
    (void)total_tokens;
    // VA reservation must use granularity-aligned block size
    {
        CUmemAllocationProp _prop{};
        _prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        _prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        _prop.location.id = _cfg.gpu_device_id;
        size_t _gran = 0;
        cuMemGetAllocationGranularity(&_gran, &_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        size_t _aligned = (_bytes_per_block() + _gran - 1) & ~(_gran - 1);
        _va_size = _aligned * 4096;
    }
    _va_base = reinterpret_cast<CUdeviceptr>(malloc(_va_size));
    return _va_base;
#endif
}

// ---------------------------------------------------------------------------
// Block allocation
// ---------------------------------------------------------------------------

std::optional<uint64_t> TieredBlockAllocator::allocate_block(
    const std::string& request_id,
    int    layer_idx,
    uint32_t token_start,
    uint32_t token_end)
{
    std::lock_guard<std::mutex> lk(_mutex);

    // Choose physical frame based on tier pressure
    PhysicalFrame* frame = nullptr;
    MemoryTier     chosen_tier;

    bool t1_pressure = (_tier1_frames.size() - _tier1_free.size()) >=
                       static_cast<size_t>(_cfg.tier1_pressure_threshold *
                                           _tier1_frames.size());

    if (!t1_pressure && !_tier1_free.empty()) {
        frame = _alloc_frame_tier1();
        chosen_tier = MemoryTier::TIER1_FAST;
    } else if (!_tier2_free.empty()) {
        frame = _alloc_frame_tier2();
        chosen_tier = MemoryTier::TIER2_SLOW;
    } else {
        // OOM — attempt emergency demotion (unlocked call would deadlock; inline)
        return std::nullopt;
    }

    uint64_t bid = _next_block_id.fetch_add(1, std::memory_order_relaxed);
    LogicalBlock blk{};
    blk.block_id         = bid;
    blk.request_id       = request_id;
    blk.layer_idx        = layer_idx;
    blk.token_start      = token_start;
    blk.token_end        = token_end;
    blk.ref_count        = 1;
    blk.aol_score        = 1.0f;
    blk.current_tier     = chosen_tier;
    blk.physical_frame_id = frame->frame_id;
    blk.is_shared        = false;

    _blocks[bid] = blk;
    _request_blocks[request_id].push_back(bid);
    frame->in_use = true;

#ifdef USE_CUDA
    _map_physical_to_virtual(bid);
#endif

    return bid;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

bool TieredBlockAllocator::free_block(uint64_t block_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) return false;

    LogicalBlock& blk = it->second;
    if (--blk.ref_count > 0) return true;  // still shared

#ifdef USE_CUDA
    _unmap_physical(block_id);
#endif

    // Return frame to free pool
    auto& pool = (blk.current_tier == MemoryTier::TIER1_FAST)
                 ? _tier1_frames : _tier2_frames;
    auto& free_list = (blk.current_tier == MemoryTier::TIER1_FAST)
                      ? _tier1_free : _tier2_free;

    for (auto& fp : pool) {
        if (fp->frame_id == blk.physical_frame_id) {
            fp->in_use = false;
            free_list.push_back(fp.get());
            break;
        }
    }
    _blocks.erase(it);
    return true;
}

void TieredBlockAllocator::free_request(const std::string& request_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _request_blocks.find(request_id);
    if (it == _request_blocks.end()) return;
    for (uint64_t bid : it->second) {
        auto bit = _blocks.find(bid);
        if (bit == _blocks.end()) continue;
        LogicalBlock& blk = bit->second;
        if (--blk.ref_count <= 0) {
            _blocks.erase(bit);
        }
    }
    _request_blocks.erase(it);
}

// ---------------------------------------------------------------------------
// Copy-on-Write
// ---------------------------------------------------------------------------

bool TieredBlockAllocator::cow_share(uint64_t block_id,
                                      const std::string& new_owner_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) return false;
    it->second.ref_count++;
    it->second.is_shared = true;
    _request_blocks[new_owner_id].push_back(block_id);
    _stat_cow_shares.fetch_add(1, std::memory_order_relaxed);
    return true;
}

uint64_t TieredBlockAllocator::cow_copy_on_write(uint64_t block_id,
                                                   const std::string& new_owner_id) {
    // Allocate a new block and memcpy the contents
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) return UINT64_MAX;
    LogicalBlock& src = it->second;

    // Unlock before recursive allocate to avoid deadlock — release, reallocate
    // For simplicity here we duplicate inline:
    PhysicalFrame* frame = (src.current_tier == MemoryTier::TIER1_FAST)
                           ? _alloc_frame_tier1() : _alloc_frame_tier2();
    if (!frame) return UINT64_MAX;

    uint64_t new_bid = _next_block_id.fetch_add(1, std::memory_order_relaxed);
    LogicalBlock new_blk = src;
    new_blk.block_id         = new_bid;
    new_blk.request_id       = new_owner_id;
    new_blk.ref_count        = 1;
    new_blk.is_shared        = false;
    new_blk.physical_frame_id = frame->frame_id;

    if (frame->host_ptr && src.current_tier == MemoryTier::TIER2_SLOW) {
        // Find src host_ptr
        for (auto& fp : _tier2_frames) {
            if (fp->frame_id == src.physical_frame_id && fp->host_ptr) {
                memcpy(frame->host_ptr, fp->host_ptr, _bytes_per_block());
                break;
            }
        }
    }
    // Decrement src ref
    if (--src.ref_count <= 0) src.is_shared = false;

    _blocks[new_bid] = new_blk;
    _request_blocks[new_owner_id].push_back(new_bid);
    frame->in_use = true;
    return new_bid;
}

// ---------------------------------------------------------------------------
// Promotion / Demotion
// ---------------------------------------------------------------------------

bool TieredBlockAllocator::promote(uint64_t block_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end() || it->second.current_tier == MemoryTier::TIER1_FAST)
        return false;
    if (_tier1_free.empty()) return false;  // no space; caller should evict first

    if (!_copy_tier2_to_tier1(block_id)) return false;

    PhysicalFrame* new_frame = _alloc_frame_tier1();
    // Update block tier metadata
    it->second.current_tier = MemoryTier::TIER1_FAST;
    it->second.physical_frame_id = new_frame->frame_id;
    new_frame->in_use = true;

    _stat_promotions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool TieredBlockAllocator::demote(uint64_t block_id) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end() || it->second.current_tier == MemoryTier::TIER2_SLOW)
        return false;
    if (_tier2_free.empty()) return false;

    if (!_copy_tier1_to_tier2(block_id)) return false;

    PhysicalFrame* new_frame = _alloc_frame_tier2();
    it->second.current_tier = MemoryTier::TIER2_SLOW;
    it->second.physical_frame_id = new_frame->frame_id;
    new_frame->in_use = true;

    if (_demotion_callback)
        _demotion_callback(block_id, MemoryTier::TIER2_SLOW);

    _stat_demotions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

size_t TieredBlockAllocator::sweep_demotion(size_t n_to_demote) {
    std::vector<uint64_t> candidates;
    {
        std::lock_guard<std::mutex> lk(_mutex);
        for (auto& [bid, blk] : _blocks) {
            if (blk.current_tier == MemoryTier::TIER1_FAST &&
                blk.ref_count == 0 &&
                blk.aol_score < 0.3f)
            {
                candidates.push_back(bid);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [this](uint64_t a, uint64_t b) {
        return _blocks[a].aol_score < _blocks[b].aol_score;
    });
    size_t count = 0;
    for (size_t i = 0; i < std::min(n_to_demote, candidates.size()); ++i) {
        if (demote(candidates[i])) ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// AOL integration
// ---------------------------------------------------------------------------

void TieredBlockAllocator::update_aol_score(uint64_t block_id, float score) {
    std::lock_guard<std::mutex> lk(_mutex);
    auto it = _blocks.find(block_id);
    if (it != _blocks.end())
        it->second.aol_score = std::clamp(score, 0.0f, 1.0f);
}

void TieredBlockAllocator::batch_update_aol(
    const std::vector<uint64_t>& block_ids,
    const std::vector<float>&    scores)
{
    assert(block_ids.size() == scores.size());
    std::lock_guard<std::mutex> lk(_mutex);
    for (size_t i = 0; i < block_ids.size(); ++i) {
        auto it = _blocks.find(block_ids[i]);
        if (it != _blocks.end())
            it->second.aol_score = std::clamp(scores[i], 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

LogicalBlock* TieredBlockAllocator::get_block(uint64_t block_id) {
    auto it = _blocks.find(block_id);
    return (it != _blocks.end()) ? &it->second : nullptr;
}

AllocatorStats TieredBlockAllocator::stats() const {
    std::lock_guard<std::mutex> lk(_mutex);
    size_t bsz = _bytes_per_block();
    size_t t1_used = (_tier1_frames.size() - _tier1_free.size()) * bsz;
    size_t t2_used = (_tier2_frames.size() - _tier2_free.size()) * bsz;
    uint64_t t1_blks = 0, t2_blks = 0;
    for (auto& [_, blk] : _blocks) {
        if (blk.current_tier == MemoryTier::TIER1_FAST) ++t1_blks;
        else ++t2_blks;
    }
    return AllocatorStats{
        t1_used, _cfg.tier1_capacity_bytes,
        t2_used, _cfg.tier2_capacity_bytes,
        _blocks.size(), t1_blks, t2_blks,
        _stat_promotions.load(), _stat_demotions.load(),
        _stat_cow_shares.load()
    };
}

bool TieredBlockAllocator::tier1_under_pressure() const {
    size_t total = _tier1_frames.size();
    size_t free  = _tier1_free.size();
    return total > 0 &&
           (float)(total - free) / total >= _cfg.tier1_pressure_threshold;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool TieredBlockAllocator::_init_tier1_pool() {
    size_t bsz   = _bytes_per_block();
    size_t nframes = _cfg.tier1_capacity_bytes / bsz;
    _tier1_frames.reserve(nframes);
    for (size_t i = 0; i < nframes; ++i) {
        auto fp = std::make_unique<PhysicalFrame>(i, MemoryTier::TIER1_FAST, bsz);
#ifndef USE_CUDA
        fp->host_ptr = malloc(bsz);
#endif
        _tier1_free.push_back(fp.get());
        _tier1_frames.push_back(std::move(fp));
    }
    return true;
}

bool TieredBlockAllocator::_init_tier2_pool() {
    size_t bsz    = _bytes_per_block();
    size_t nframes = _cfg.tier2_capacity_bytes / bsz;
    _tier2_frames.reserve(nframes);
    for (size_t i = 0; i < nframes; ++i) {
        auto fp = std::make_unique<PhysicalFrame>(
            _tier1_frames.size() + i, MemoryTier::TIER2_SLOW, bsz);
        fp->host_ptr = malloc(bsz);  // In production: DAX mmap
        _tier2_free.push_back(fp.get());
        _tier2_frames.push_back(std::move(fp));
    }
    return true;
}

bool TieredBlockAllocator::_init_cuda_vmm() {
#ifdef USE_CUDA
    CUDA_CHECK(cuInit(0));
#endif
    return true;
}

bool TieredBlockAllocator::_map_physical_to_virtual(uint64_t block_id) {
#ifdef USE_CUDA
    auto it = _blocks.find(block_id);

    CUmemAllocationProp prop{};
    prop.type           = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type  = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id    = _cfg.gpu_device_id;

    size_t _gran = 0;
    cuMemGetAllocationGranularity(&_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t alloc_sz = (_bytes_per_block() + _gran - 1) & ~(_gran - 1);
    CUmemGenericAllocationHandle handle;
    CUDA_CHECK(cuMemCreate(&handle, alloc_sz, &prop, 0));

    // Compute offset in VA range based on block_id
    size_t offset = block_id * alloc_sz;
    CUdeviceptr va = _va_base + offset;
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CHECK(cuMemMap(va, alloc_sz, 0, handle, 0));
    CUDA_CHECK(cuMemSetAccess(va, alloc_sz, &access, 1));
#else
    (void)block_id;
#endif
    return true;
}

bool TieredBlockAllocator::_unmap_physical(uint64_t block_id) {
#ifdef USE_CUDA
    CUmemAllocationProp _p{}; _p.type=CU_MEM_ALLOCATION_TYPE_PINNED; _p.location.type=CU_MEM_LOCATION_TYPE_DEVICE; _p.location.id=_cfg.gpu_device_id; size_t _g=0; cuMemGetAllocationGranularity(&_g,&_p,CU_MEM_ALLOC_GRANULARITY_MINIMUM); size_t alloc_sz=(_bytes_per_block()+_g-1)&~(_g-1); size_t offset=block_id*alloc_sz;
    CUdeviceptr va = _va_base + offset;
    cuMemUnmap(va, alloc_sz);
#else
    (void)block_id;
#endif
    return true;
}

bool TieredBlockAllocator::_copy_tier1_to_tier2(uint64_t block_id) {
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) return false;
    LogicalBlock& blk = it->second;

    PhysicalFrame* dst = nullptr;
    for (auto& fp : _tier2_frames) {
        if (!fp->in_use) { dst = fp.get(); break; }
    }
    if (!dst || !dst->host_ptr) return false;

    // src: pinned GPU memory or host ptr
    for (auto& fp : _tier1_frames) {
        if (fp->frame_id == blk.physical_frame_id) {
#ifdef USE_CUDA
            size_t offset = block_id * _bytes_per_block();
            cudaMemcpy(dst->host_ptr,
                       reinterpret_cast<void*>(_va_base + offset),
                       _bytes_per_block(), cudaMemcpyDeviceToHost);
#else
            if (fp->host_ptr)
                memcpy(dst->host_ptr, fp->host_ptr, _bytes_per_block());
#endif
            return true;
        }
    }
    return false;
}

bool TieredBlockAllocator::_copy_tier2_to_tier1(uint64_t block_id) {
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) return false;
    LogicalBlock& blk = it->second;

    for (auto& fp : _tier2_frames) {
        if (fp->frame_id == blk.physical_frame_id && fp->host_ptr) {
#ifdef USE_CUDA
            size_t offset = block_id * _bytes_per_block();
            cudaMemcpy(reinterpret_cast<void*>(_va_base + offset),
                       fp->host_ptr, _bytes_per_block(), cudaMemcpyHostToDevice);
#endif
            return true;
        }
    }
    return false;
}

PhysicalFrame* TieredBlockAllocator::_alloc_frame_tier1() {
    if (_tier1_free.empty()) return nullptr;
    PhysicalFrame* f = _tier1_free.back();
    _tier1_free.pop_back();
    return f;
}

PhysicalFrame* TieredBlockAllocator::_alloc_frame_tier2() {
    if (_tier2_free.empty()) return nullptr;
    PhysicalFrame* f = _tier2_free.back();
    _tier2_free.pop_back();
    return f;
}

size_t TieredBlockAllocator::_bytes_per_block() const {
    // K + V tensors, all layers, fp16
    return 2ULL * _cfg.num_layers * _cfg.num_heads *
           _cfg.head_dim * _cfg.block_size_tokens * sizeof(uint16_t);
}

} // namespace tierkv
