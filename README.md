# Tier-Aware KV Cache Management for LLM Inference

An OS-inspired tiered memory manager for LLM KV cache, built as a course project.

## Key Results (LLaMA-7B & 13B, simulation)
| Workload | Throughput Δ | TTFT Δ |
|----------|-------------|--------|
| 7B Short | +12.5% | -32.2% |
| 7B Long  | +3.3%  | -18.7% |
| 13B Beam | +31.1% | -35.2% |

## Architecture
- **AOL Metric**: Amortized Offcore Latency = stall_cycles / (access_count × MLP)
- **VMM**: CUDA cuMemMap for virtual contiguity across T1 (GPU) and T2 (CPU)
- **SOAR**: Background demotion daemon (kdemoted), sweeps every 200ms
- **ALTO**: AOL-gated promotion threshold (θ=0.4)
- **Phase-aware**: Proactive T2→T1 promotion at PREFILL→DECODE transition

## Stack
| Layer | Language |
|-------|----------|
| Policy engine | Python |
| Block allocator | C++ / CUDA VMM |
| AOL profiler | Rust |
| Attention kernel | CUDA |
| REST API | FastAPI |

## Setup (Google Colab)
```bash
unzip tier_aware_kv_cache.zip -d /content/project
cd /content/project/tier_aware_kv_cache
pip install fastapi uvicorn pydantic
python -m unittest discover -s tests -v
```

## Build C++ Library
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j4
ctest --output-on-failure
```

## Tests: 34/34 passing
- 15 Python (orchestrator, AOL, kdemoted)
- 11 C++ (TieredBlockAllocator)
- 8 C++ (MigrationOrchestrator)
