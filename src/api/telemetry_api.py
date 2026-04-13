"""
api/telemetry_api.py
Tier-Aware KV Cache — Telemetry & Management REST API

Run with:
    uvicorn telemetry_api:app --host 0.0.0.0 --port 8080
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Import the orchestrator (adjust path as needed)
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from manager.orchestrator import TierAwareKVCacheOrchestrator, MemoryTier

# ---------------------------------------------------------------------------
# Global orchestrator instance (shared with inference engine)
# ---------------------------------------------------------------------------
_orchestrator: Optional[TierAwareKVCacheOrchestrator] = None
_history: List[dict] = []          # rolling stats history
_history_lock = threading.Lock()
MAX_HISTORY = 300                   # 5 min at 1-s intervals


def get_orchestrator() -> TierAwareKVCacheOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TierAwareKVCacheOrchestrator(
            tier1_capacity_mb=4096,
            tier2_capacity_mb=32768,
            block_size=16,
            num_layers=32,
            head_dim=128,
            num_heads=8,
        )
    return _orchestrator


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tier-Aware KV Cache API",
    version="1.0.0",
    description="Management & telemetry for the multi-tiered KV cache manager",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background stats collector
# ---------------------------------------------------------------------------
def _stats_collector():
    while True:
        time.sleep(1.0)
        orch = get_orchestrator()
        snap = orch.get_stats()
        snap["timestamp"] = time.time()
        with _history_lock:
            _history.append(snap)
            if len(_history) > MAX_HISTORY:
                _history.pop(0)

_collector_thread = threading.Thread(
    target=_stats_collector, daemon=True, name="stats-collector"
)
_collector_thread.start()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequestBody(BaseModel):
    prompt_tokens: int

class AllocateBlocksBody(BaseModel):
    request_id: str
    num_tokens: int
    layer_idx: int = 0

class AOLUpdateBody(BaseModel):
    block_id: int
    aol_score: float  # 0.0 .. 1.0

class FreeRequestBody(BaseModel):
    request_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# ---- Stats & telemetry ----

@app.get("/stats")
def current_stats():
    """Return current memory utilisation snapshot."""
    return get_orchestrator().get_stats()


@app.get("/stats/history")
def stats_history(last_n: int = 60):
    """Return last N seconds of stats snapshots."""
    with _history_lock:
        return _history[-last_n:]


# ---- Request lifecycle ----

@app.post("/requests/register")
def register_request(body: RegisterRequestBody):
    rid = get_orchestrator().register_request(body.prompt_tokens)
    return {"request_id": rid}


@app.post("/requests/free")
def free_request(body: FreeRequestBody):
    get_orchestrator().free_request(body.request_id)
    return {"freed": body.request_id}


# ---- Block operations ----

@app.post("/blocks/allocate")
def allocate_blocks(body: AllocateBlocksBody):
    orch = get_orchestrator()
    block_ids = orch.allocate_blocks(
        body.request_id, body.num_tokens, body.layer_idx
    )
    return {"block_ids": block_ids}


@app.post("/blocks/aol_update")
def update_aol(body: AOLUpdateBody):
    get_orchestrator().update_aol_score(body.block_id, body.aol_score)
    return {"updated": body.block_id, "score": body.aol_score}


@app.post("/blocks/{block_id}/promote")
def promote_block(block_id: int):
    ok = get_orchestrator().promote_block(block_id)
    if not ok:
        raise HTTPException(status_code=409,
                            detail="Block already in T1 or no T1 space")
    return {"promoted": block_id}


# ---- Tier summary ----

@app.get("/tiers")
def tier_summary():
    orch = get_orchestrator()
    stats = orch.get_stats()
    return {
        "tier1": {
            "used_mb":   stats["tier1_used_mb"],
            "cap_mb":    stats["tier1_cap_mb"],
            "util_pct":  stats["tier1_util_pct"],
            "blocks":    stats["tier1_blocks"],
        },
        "tier2": {
            "used_mb":   stats["tier2_used_mb"],
            "cap_mb":    stats["tier2_cap_mb"],
            "util_pct":  stats["tier2_util_pct"],
            "blocks":    stats["tier2_blocks"],
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
