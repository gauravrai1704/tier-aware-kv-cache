# PACT-KV: Phase-Aware Cost-Optimal Tiering for LLM KV Cache

An OS-inspired tiered memory manager for the LLM KV cache. KV blocks are placed and migrated between GPU VRAM (**T1**) and CPU DRAM (**T2**) using a hardware-measured cost signal — **Amortized Offcore Latency (AOL)** — instead of recency.

The work is evaluated on two tracks:

| Track | Environment | Purpose |
|-------|-------------|---------|
| **Hardware** | NVIDIA RTX A4000 (16 GB) + Intel i9-12900K, Mistral-7B-v0.1 (FP16) via vLLM v0.19.0 | Feasibility, and the timing constants that calibrate the simulator |
| **Simulated** | Hardware-calibrated discrete-event model (Google Colab) | Six-system comparison, ablations, and VRAM-budget sweeps |

---

## Key Results

- **3.2× servable context** on real hardware (2,560 → 8,192+ tokens) where vLLM T1-only hits OOM.
- **< 0.5% throughput overhead** vs. T1-only (0.44% worst case).
- **100% KV-cache hit rate** on every simulated workload, including RAG traffic where every recency-based baseline drops below 3%.
- **TPOT matches VRAM-only (1.438 ms)** under heavy T2 pressure, where naive LRU offloading degrades to 19.9 ms.

---

## Architecture

| Component | What it does |
|-----------|--------------|
| **AOL profiler** | Per-block cost score: `AOL = stall_cycles / (access_count × MLP)`, EMA-smoothed (α = 0.8), normalized to [0, 1] |
| **PDD** — Pressure-Driven Demotion (`kdemoted`) | Every 200 ms, if T1 > 85% full, demotes the lowest-AOL (cheapest-to-refetch) blocks to T2 |
| **AGP** — AOL-Gated Promotion | Promotes a T2 block only if AOL ≥ θ = 0.4 **and** it exceeds the minimum AOL in T1 |
| **PAM** — Phase-Aware Migration | At the prefill → decode boundary, restores a request's T2 blocks once (layer order), so PCIe cost lands in TTFT, not every decode step |
| **APF** — Attention Pattern Forecasting | Online-trained single-layer LSTM (hidden 32, 8,384 params) predicts attention 3 steps ahead; prefetches T2 blocks with score > 0.6 |
| **VMM** | CUDA `cuMemAddressReserve` / `cuMemMap` — each block keeps a fixed virtual address; migration only remaps the physical frame, so the PagedAttention kernel is unmodified |
| **Async orchestrator** | Prefill, decode, and DMA migration run as concurrent workers |

---

## Track 1 — Hardware Results

**Setup:** RTX A4000 (16 GB), Mistral-7B-v0.1 FP16, vLLM v0.19.0, `gpu-memory-utilization=0.95`, `enforce-eager`. ≈400 MB of KV budget remains after the 13.5 GiB model load.

### Context capacity

| Context Length | KV Needed | vLLM T1-Only | PACT-KV |
|----------------|-----------|--------------|---------|
| 1,024 tokens | ~0.35 GiB | ✅ 5.51 s | ✅ Served |
| 2,048 tokens | ~0.39 GiB | ✅ 6.14 s | ✅ Served |
| 2,560 tokens | ~0.40 GiB | ✅ 6.36 s | ✅ Served |
| 3,072 tokens | 0.41 GiB | ❌ OOM | ✅ Served via T2 |
| 4,096 tokens | 0.53 GiB | ❌ OOM | ✅ Served via T2 |
| 8,192 tokens | 1.03 GiB | ❌ OOM | ✅ Served via T2 |

### Throughput and latency under T2 pressure (T1 = 400 MB, T2 = 8,192 MB, 5 runs)

| Workload | Throughput T1-only (tok/s) | Throughput PACT-KV (tok/s) | TTFT T1-only (ms) | TTFT PACT-KV (ms) | TPOT T1-only (ms) | TPOT PACT-KV (ms) |
|----------|------|------|------|------|-------|-------|
| long-context | 515.8 ± 2.3 | 514.7 ± 1.3 | 7.03 | 8.02 | 1.442 | 1.436 |
| mixed | 575.4 ± 2.1 | 576.0 ± 2.2 | 4.25 | 5.84 | 1.432 | 1.428 |
| beam-search | 655.5 ± 3.7 | 652.6 ± 5.2 | 1.24 | 1.26 | 1.325 | 1.325 |

TPOT is unchanged while TTFT rises by 1.0–1.6 ms — the signature of PAM: tiering cost is paid once at the phase boundary, not on every decode step. Measured per-block access latency on this platform: **≈1 µs (T1)**, **≈80 µs (T2)**.

---

## Track 2 — Simulated Results (Google Colab)

### Overview

Running six systems × five workloads × four VRAM budgets on physical hardware is infeasible in GPU-hours, and several baselines are research prototypes without stable deployments. We therefore built a **discrete-event performance model** whose timing constants (PCIe bandwidth, T1/T2 access latency, prefill/decode cost per token) are **fit directly from the Track 1 measurements**.

Every baseline runs against the **same cost model and constants**; systems differ **only** in their placement / promotion / demotion logic, reproduced from each system's paper. This isolates placement policy from transfer-engineering effort.

- **Systems:** vLLM T1-Only, vLLM 2-Tier, LRU, AttentionStore, LMCache, Strata, PACT-KV
- **Workloads:** ShareGPT (N=150), LongBench (N=75), Alpaca (N=150), 2WikiMultihopQA (N=100), RULER 4K–32K (N=75)
- **VRAM budgets:** 100 / 200 / 300 / 400 MB
- **Total:** 7 systems × 5 workloads × 4 budgets = **140 configurations**
- **Metrics:** TTFT, TPOT, throughput (TPS), KV-hit rate, T2 utilization. PAM restores are charged to TTFT and any decode-time T2 fetch to TPOT; a hit counts only if the transfer completed before the read.

### Six-system comparison (highest-pressure workloads)

| Dataset | System | TTFT (ms) ↓ | TPOT (ms) ↓ | TPS ↑ | KV-Hit ↑ | T2-Util % |
|---------|--------|-------------|-------------|-------|----------|-----------|
| **LongBench** | LRU | 137.94 | 5.300 | 130.5 | 0.012 | 65.5 |
| | AttentionStore | 135.02 | 1.792 | 242.0 | 0.719 | 65.5 |
| | LMCache | 138.25 | 3.031 | 181.7 | 0.012 | 65.5 |
| | vLLM 2-Tier | 143.89 | 1.438 | 247.2 | 0.012 | 66.8 |
| | Strata | 135.62 | 1.510 | 251.6 | 0.012 | 66.8 |
| | **PACT-KV** | **131.50** | **1.438** | **261.3** | **1.000** | 65.5 |
| **2WikiMultihopQA** | LRU | 54.02 | 2.918 | 193.7 | 0.028 | 35.8 |
| | AttentionStore | 52.40 | 1.568 | 258.2 | 0.738 | 35.8 |
| | LMCache | 53.84 | 2.049 | 228.3 | 0.028 | 35.8 |
| | vLLM 2-Tier | 56.00 | 1.438 | 255.6 | 0.028 | 36.5 |
| | Strata | 52.85 | 1.510 | 258.9 | 0.028 | 36.5 |
| | **PACT-KV** | **51.55** | **1.438** | **267.9** | **1.000** | 35.8 |
| **RULER** | LRU | 651.36 | 19.909 | 21.5 | 0.014 | 76.4 |
| | AttentionStore | 637.70 | 3.197 | 26.8 | 0.720 | 76.4 |
| | LMCache | 654.73 | 9.057 | 24.4 | 0.014 | 76.4 |
| | vLLM 2-Tier | 681.76 | 1.438 | 26.0 | 0.014 | 77.8 |
| | Strata | 641.96 | 1.510 | 27.3 | 0.014 | 77.8 |
| | **PACT-KV** | **620.47** | **1.438** | **28.3** | **1.000** | 76.4 |

**Takeaways**
- On 2WikiMultihopQA's disjoint, read-once RAG access, LRU, LMCache and Strata collapse to a 0.028 hit rate; AttentionStore recovers to 0.738 via session clustering; only PACT-KV reaches 1.000.
- Under RULER's 76.4% T2 pressure, LRU's TPOT blows out to 19.9 ms and LMCache's to 9.1 ms; PACT-KV has the best TTFT, TPOT and throughput simultaneously.
- **vs. Strata:** PACT-KV's TTFT is 2–3% lower, TPOT ~5% lower, throughput 3–4% higher. Strata lowers the cost of each transfer; PACT-KV reduces how often expensive transfers happen — the two are complementary. (Strata's reported 5× TTFT gain depends on queueing/delay-hit regimes not modeled here.)

### Ablation — contribution of APF (ShareGPT, n=150)

| System | TTFT | TPOT | TPS | OOM % | KV-Hit | T2-Util % |
|--------|------|------|-----|-------|--------|-----------|
| vLLM T1-Only | 4.78 | 1.436 | 662.4 | 0.0 | 1.000 | 0.0 |
| LRU | 5.37 | 1.548 | 637.4 | 0.0 | 0.204 | 3.9 |
| PACT-KV w/o APF | 5.18 | 1.436 | 660.3 | 0.0 | 1.000 | 3.9 |
| **PACT-KV** | **4.82** | **1.435** | **662.8** | **0.0** | **1.000** | 45.0 |

- **APF prefetch accuracy: 98.0%** vs. 77.3% for a StreamingLLM-style recency/attention-sink heuristic (+20.7 pts; block counts 8–128, ten seeds).
- APF brings TTFT to within 0.04 ms of the T1-only floor. Higher T2 utilization alongside lower latency means APF moves the *right* data earlier, not less data.
- Without APF, end-to-end latency diverges from T1-only past the OOM wall; with APF it tracks T1-only to the longest context tested (single-request load).

### Sensitivity to VRAM budget (RULER, 400 → 100 MB)

Sweeping simulated T1 capacity down to 100 MB (~640 tokens of headroom), every reactive baseline's hit rate starts near its floor and degrades further, while **PACT-KV holds a 1.000 hit rate at every budget** — AOL-gated eviction always sheds the cheapest-to-refetch blocks first.

> Simulator notebook and raw outputs will be added to this branch.

---

## Stack

| Layer | Language |
|-------|----------|
| Policy engine / orchestrator | Python |
| Block allocator | C++ / CUDA VMM |
| AOL profiler | Rust |
| Attention kernel | CUDA (PagedAttention, unmodified) |
| REST API | FastAPI |
| Simulator (Track 2) | Python (Google Colab) |

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

## Run Hardware Benchmarks

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

### Code Availability

> **Note:** The Track 2 code — the discrete-event simulator, the baseline policy implementations, and the APF LSTM training and inference pipeline (originally developed in Google Colab) — is **not yet publicly released**, as this work is ongoing. The results above come from that codebase, and it will be added to this repository once the work is finalized. This repository currently contains the full **Track 1 (hardware)** implementation: the policy engine, the C++/CUDA block allocator, the migration orchestrator, the benchmarks, and the tests.
