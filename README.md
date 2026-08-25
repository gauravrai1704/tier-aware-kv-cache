# Tier-Aware KV Cache Management for LLM Inference

An OS-inspired tiered memory manager for LLM KV cache, built as a course project.
Evaluated on real hardware (NVIDIA RTX A4000, 16GB) with Mistral-7B via vLLM.

## Key Result

vLLM exhausts its 400MB GPU KV cache budget at **3,072 tokens** (OOM).  
Our tiered system serves up to **8,192+ tokens** — a **2.7× capacity extension**  
with less than 0.2% throughput overhead.

| Context Length | vLLM T1-Only | Tiered System |
|----------------|-------------|---------------|
| 1,024 tokens   | ✅ 5.51s    | ✅ Served     |
| 2,048 tokens   | ✅ 6.14s    | ✅ Served     |
| 2,560 tokens   | ✅ 6.36s    | ✅ Served     |
| 3,072 tokens   | ❌ OOM      | ✅ Served via T2 |
| 4,096 tokens   | ❌ OOM      | ✅ Served via T2 |
| 8,192 tokens   | ❌ OOM      | ✅ Served via T2 |

| Workload     | T1-Only (tok/s) | Tiered (tok/s) | T2 Util |
|--------------|----------------|----------------|---------|
| long-context | 515.8 ± 2.3    | 514.7 ± 1.3    | 941.2%  |
| mixed        | 575.4 ± 2.1    | 576.0 ± 2.2    | 470.8%  |
| beam-search  | 655.5 ± 3.7    | 652.6 ± 5.2    | 381.8%  |

## Architecture

- **AOL Metric**: Amortized Offcore Latency = stall_cycles / (access_count × MLP)
- **VMM**: CUDA cuMemMap for virtual contiguity across T1 (GPU) and T2 (CPU DRAM)
- **SOAR**: Background demotion daemon (kdemoted), sweeps every 200ms at 85% T1 pressure
- **ALTO**: AOL-gated promotion threshold (θ=0.4), phase-aware T2→T1 at PREFILL→DECODE
- **Baseline**: Mistral-7B-v0.1 via vLLM v0.19.0 on RTX A4000

## Stack

| Layer | Language |
|-------|----------|
| Policy engine | Python |
| Block allocator | C++ / CUDA VMM |
| AOL profiler | Rust |
| Attention kernel | CUDA |
| REST API | FastAPI |

## Setup

```bash
git clone https://github.com/gauravrai1704/tier-aware-kv-cache.git
cd tier-aware-kv-cache
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Build C++ Library

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j4
ctest --output-on-failure
```

## Run Benchmarks

```bash
# T1-only vs Tiered comparison (matches vLLM's 400MB KV budget)
python3 benchmarks/benchmark_suite.py \
    --model 7b --workload long --compare --runs 5 \
    --tier1-mb 400 --tier2-mb 8192

# vLLM baseline results are in results/vllm_baseline/
```

## Tests: 34/34 passing

- 15 Python (orchestrator, AOL, kdemoted)
- 11 C++ (TieredBlockAllocator)
- 8 C++ (MigrationOrchestrator)
