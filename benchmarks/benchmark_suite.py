"""
benchmarks/benchmark_suite.py
Tier-Aware KV Cache — Benchmark Suite

Measures token generation throughput and memory efficiency across:
  - Model sizes (7B / 13B / 70B parameter configs)
  - Workload patterns (short / long / mixed context)
  - Memory configurations (T1-only vs T1+T2 tiered)

Run:
    python benchmark_suite.py --model 7b --context long --runs 5
"""

import argparse
import time
import math
import random
import statistics
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.manager.orchestrator import TierAwareKVCacheOrchestrator, MemoryTier


# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name:      str
    num_layers: int
    num_heads:  int
    head_dim:   int
    vocab_size: int
    hidden_dim: int

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "7b":  ModelConfig("LLaMA-7B",  32, 8, 128, 32000, 4096),
    "13b": ModelConfig("LLaMA-13B", 40, 8, 128, 32000, 5120),
    "70b": ModelConfig("LLaMA-70B", 80, 8, 128, 32000, 8192),
}


# ---------------------------------------------------------------------------
# Workload Patterns
# ---------------------------------------------------------------------------

@dataclass
class WorkloadPattern:
    name:             str
    prompt_len_range: tuple   # (min, max) tokens
    gen_len_range:    tuple
    batch_size:       int
    concurrency:      int

WORKLOAD_PATTERNS: Dict[str, WorkloadPattern] = {
    "short": WorkloadPattern(
        "short-context", (128, 512), (64, 256), batch_size=32, concurrency=16),
    "long": WorkloadPattern(
        "long-context", (2048, 6144), (512, 1024), batch_size=4, concurrency=4),
    "mixed": WorkloadPattern(
        "mixed", (256, 4096), (128, 512), batch_size=8, concurrency=8),
    "beam":  WorkloadPattern(
        "beam-search", (512, 1024), (256, 512), batch_size=4, concurrency=4),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config_name:    str
    workload_name:  str
    tier_mode:      str   # "t1_only" or "tiered"
    total_tokens:   int
    total_time_s:   float
    throughput_tps: float  # tokens / second
    mean_ttft_ms:   float  # time-to-first-token
    mean_tpot_ms:   float  # time-per-output-token
    t1_peak_util:   float  # peak Tier-1 utilisation 0..1
    t2_peak_util:   float
    demotion_count: int
    promotion_count: int
    oom_events:     int


@dataclass
class BenchmarkReport:
    timestamp:   str
    runs:        List[RunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulator (stands in for real LLM inference on GPU)
# ---------------------------------------------------------------------------

class InferenceSimulator:
    """
    Simulates LLM inference token generation without an actual model.
    Models:
      - Prefill latency ∝ prompt_len
      - Decode latency ∝ 1 + kv_cache_pressure (T2 adds latency penalty)
    """

    T2_LATENCY_FACTOR = 1.8    # T2 access ≈ 1.8× slower than T1

    def __init__(
        self,
        model:     ModelConfig,
        workload:  WorkloadPattern,
        orch:      TierAwareKVCacheOrchestrator,
    ):
        self.model    = model
        self.workload = workload
        self.orch     = orch
        self._rng     = random.Random(42)

    def run(self, num_requests: int) -> RunResult:
        total_tokens   = 0
        ttfts: List[float] = []
        tpots: List[float] = []
        t1_peak = 0.0
        t2_peak = 0.0
        demotions  = 0
        promotions = 0
        ooms       = 0
        total_start = time.perf_counter()

        for _ in range(num_requests):
            prompt_len = self._rng.randint(*self.workload.prompt_len_range)
            gen_len    = self._rng.randint(*self.workload.gen_len_range)

            rid = self.orch.register_request(prompt_len)
            try:
                # ---- Prefill phase ----
                t0 = time.perf_counter()
                block_ids = []
                for layer in range(self.model.num_layers):
                    bids = self.orch.allocate_blocks(rid, prompt_len, layer)
                    block_ids.extend(bids)
                ttft = (time.perf_counter() - t0) * 1000
                ttfts.append(ttft)

                # ---- Decode phase ----
                for step in range(gen_len):
                    t1 = time.perf_counter()
                    # Simulate access pattern — some blocks may be in T2
                    for bid in self._rng.sample(block_ids, min(4, len(block_ids))):
                        tier = self.orch.access_block(bid)
                        if tier == MemoryTier.TIER2_SLOW:
                            # Simulate PCIe T2 latency: 2.5MB block @ ~16GB/s = ~156us
                            time.sleep(0.000156 * self.T2_LATENCY_FACTOR)
                    # Append new token to cache
                    for layer in range(self.model.num_layers):
                        self.orch.allocate_blocks(rid, 1, layer)
                    tpot = (time.perf_counter() - t1) * 1000
                    tpots.append(tpot)
                    total_tokens += 1

                    # Track peak utilisation
                    s = self.orch.get_stats()
                    t1_peak = max(t1_peak, s["tier1_util_pct"] / 100.0)
                    t2_peak = max(t2_peak, s["tier2_util_pct"] / 100.0)

            except Exception:
                ooms += 1
            finally:
                self.orch.free_request(rid)

        elapsed = time.perf_counter() - total_start
        throughput = total_tokens / elapsed if elapsed > 0 else 0.0

        return RunResult(
            config_name    = self.model.name,
            workload_name  = self.workload.name,
            tier_mode      = "tiered",
            total_tokens   = total_tokens,
            total_time_s   = elapsed,
            throughput_tps = throughput,
            mean_ttft_ms   = statistics.mean(ttfts) if ttfts else 0,
            mean_tpot_ms   = statistics.mean(tpots) if tpots else 0,
            t1_peak_util   = t1_peak,
            t2_peak_util   = t2_peak,
            demotion_count = demotions,
            promotion_count= promotions,
            oom_events     = ooms,
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model_key:    str,
    workload_key: str,
    num_runs:     int = 3,
    tier1_mb:     int = 4096,
    tier2_mb:     int = 32768,
) -> BenchmarkReport:

    model    = MODEL_CONFIGS[model_key]
    workload = WORKLOAD_PATTERNS[workload_key]
    report   = BenchmarkReport(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

    print(f"\n{'='*60}")
    print(f"Benchmark: {model.name} | {workload.name} | runs={num_runs}")
    print(f"  T1={tier1_mb}MB  T2={tier2_mb}MB")
    print(f"{'='*60}")

    for run_idx in range(num_runs):
        print(f"\n  Run {run_idx+1}/{num_runs} ...", end=" ", flush=True)

        orch = TierAwareKVCacheOrchestrator(
            tier1_capacity_mb = tier1_mb,
            tier2_capacity_mb = tier2_mb,
            block_size        = 16,
            num_layers        = model.num_layers,
            head_dim          = model.head_dim,
            num_heads         = model.num_heads,
        )

        sim    = InferenceSimulator(model, workload, orch)
        result = sim.run(num_requests=workload.batch_size)

        report.runs.append(result)
        print(
            f"throughput={result.throughput_tps:.1f} tok/s  "
            f"ttft={result.mean_ttft_ms:.2f}ms  "
            f"tpot={result.mean_tpot_ms:.3f}ms  "
            f"T1={result.t1_peak_util*100:.1f}%  "
            f"T2={result.t2_peak_util*100:.1f}%"
        )

    # Aggregate stats
    throughputs = [r.throughput_tps for r in report.runs]
    ttfts       = [r.mean_ttft_ms   for r in report.runs]
    tpots       = [r.mean_tpot_ms   for r in report.runs]

    print(f"\n  Summary:")
    print(f"    Throughput : {statistics.mean(throughputs):.1f} ± "
          f"{statistics.stdev(throughputs) if len(throughputs)>1 else 0:.1f} tok/s")
    print(f"    TTFT       : {statistics.mean(ttfts):.2f} ms")
    print(f"    TPOT       : {statistics.mean(tpots):.3f} ms")

    return report


# ---------------------------------------------------------------------------
# Comparison: T1-only vs Tiered
# ---------------------------------------------------------------------------

def compare_configurations(model_key: str, workload_key: str, num_runs: int = 2,
                           tier1_mb: int = 4096, tier2_mb: int = 32768):
    """Directly compare baseline (T1-only) vs Tiered policy."""
    model    = MODEL_CONFIGS[model_key]
    workload = WORKLOAD_PATTERNS[workload_key]

    print(f"\n{'#'*60}")
    print(f"  Comparison: T1-Only vs Tiered")
    print(f"  Model: {model.name}  |  Workload: {workload.name}")
    print(f"  T1 cap: {tier1_mb}MB  T2 cap: {tier2_mb}MB")
    print(f"{'#'*60}")

    results = {}
    for label, t1, t2 in [
        (f"T1-Only ({tier1_mb}MB)",  tier1_mb,  0),
        (f"Tiered  ({tier1_mb}+{tier2_mb})", tier1_mb, tier2_mb),
    ]:
        print(f"\n[{label}]")
        r = run_benchmark(model_key, workload_key, num_runs=num_runs,
                          tier1_mb=t1, tier2_mb=t2)
        results[label] = r

    # Print comparison table
    print(f"\n{'─'*60}")
    print(f"{'Configuration':<22} {'Throughput (tok/s)':>20} {'TTFT (ms)':>12} {'TPOT (ms)':>10}")
    print(f"{'─'*60}")
    for label, report in results.items():
        thr  = statistics.mean([r.throughput_tps for r in report.runs])
        ttft = statistics.mean([r.mean_ttft_ms   for r in report.runs])
        tpot = statistics.mean([r.mean_tpot_ms   for r in report.runs])
        print(f"  {label:<20} {thr:>20.1f} {ttft:>12.2f} {tpot:>10.3f}")
    print(f"{'─'*60}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tier-Aware KV Cache Benchmark")
    p.add_argument("--model",    choices=list(MODEL_CONFIGS),    default="7b")
    p.add_argument("--workload", choices=list(WORKLOAD_PATTERNS), default="long")
    p.add_argument("--runs",     type=int, default=3)
    p.add_argument("--compare",  action="store_true",
                   help="Run T1-only vs Tiered comparison")
    p.add_argument("--tier1-mb", type=int, default=4096)
    p.add_argument("--tier2-mb", type=int, default=32768)
    p.add_argument("--output",   type=str, default=None,
                   help="Save JSON report to file")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        results = compare_configurations(args.model, args.workload, args.runs,
                                         tier1_mb=args.tier1_mb, tier2_mb=args.tier2_mb)
        if args.output:
            report_data = {k: [asdict(r) for r in v.runs]
                           for k, v in results.items()}
            with open(args.output, "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\nReport saved to {args.output}")
    else:
        report = run_benchmark(
            args.model, args.workload, args.runs,
            tier1_mb=args.tier1_mb, tier2_mb=args.tier2_mb,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in report.runs], f, indent=2)
            print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
