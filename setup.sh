#!/usr/bin/env bash
# =============================================================================
# Tier-Aware KV Cache — Setup Script
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

info "=================================================="
info " Tier-Aware KV Cache — Setup"
info "=================================================="

# ---------------------------------------------------------------------------
# 1. Python virtual environment
# ---------------------------------------------------------------------------
info "Creating Python virtual environment..."
python3 -m venv .venv || error "python3 -m venv failed. Install python3-venv."
source .venv/bin/activate

info "Upgrading pip..."
pip install --upgrade pip -q

info "Installing Python dependencies..."
pip install -r requirements.txt -q
info "Python dependencies installed."

# ---------------------------------------------------------------------------
# 2. C++ / CMake build
# ---------------------------------------------------------------------------
if command -v cmake &>/dev/null && command -v make &>/dev/null; then
    info "Building C++ library (CPU-only)..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_RUST=OFF \
          -DBUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          2>&1 | tail -5
    make -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)" \
         2>&1 | tail -10
    cd ..
    info "C++ build complete → build/libtierkv_core.a"
else
    warn "cmake or make not found — skipping C++ build."
    warn "Install cmake + build-essential and re-run to build the C++ allocator."
fi

# ---------------------------------------------------------------------------
# 3. Rust AOL profiler (optional)
# ---------------------------------------------------------------------------
if command -v cargo &>/dev/null; then
    info "Building Rust AOL profiler..."
    cd src/profiler
    cargo build --release 2>&1 | tail -5
    cd "$PROJECT_DIR"
    info "Rust build complete → src/profiler/target/release/libaol_profiler.*"
else
    warn "cargo not found — skipping Rust AOL profiler build."
    warn "Install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# ---------------------------------------------------------------------------
# 4. CUDA check
# ---------------------------------------------------------------------------
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',')
    info "CUDA detected: $CUDA_VER"
    info "To build with CUDA support, run:"
    info "  mkdir -p build && cd build && cmake .. -DUSE_CUDA=ON && make -j\$(nproc)"
else
    warn "nvcc not found — CUDA attention kernels will not be compiled."
    warn "Install CUDA Toolkit 11.8+ for GPU support."
fi

# ---------------------------------------------------------------------------
# 5. Python tests
# ---------------------------------------------------------------------------
info "Running Python unit tests..."
source .venv/bin/activate
python -m unittest discover -s tests -v 2>&1 | tail -5
info "Tests complete."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
info "=================================================="
info " Setup complete!"
info "=================================================="
echo ""
echo "  Activate venv:     source .venv/bin/activate"
echo ""
echo "  Run benchmarks:    python benchmarks/benchmark_suite.py --model 7b --workload long --compare"
echo ""
echo "  Start API server:  uvicorn src.api.telemetry_api:app --host 0.0.0.0 --port 8080"
echo ""
echo "  Run tests:         python -m unittest discover -s tests -v"
echo ""
