// aol_profiler/src/lib.rs
//! Amortized Offcore Latency (AOL) Profiler
//!
//! Interfaces with Linux perf/PEBS hardware performance counters to compute
//! the true performance impact of each KV-cache logical block.
//!
//! AOL = (total_stall_cycles) / (access_count * MLP)
//!
//! A high AOL → block access is causing real stalls → keep in Tier-1.
//! A low  AOL → latency is hidden by MLP         → safe to demote to Tier-2.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-block statistics maintained by the profiler.
#[derive(Debug, Clone, Default)]
pub struct BlockStats {
    pub block_id:      u64,
    pub access_count:  u64,
    pub stall_cycles:  u64,   /// sum of off-core stall cycles attributed to this block
    pub mlp:           f64,   /// average memory-level parallelism during accesses
    pub aol_score:     f64,   /// normalised 0..1 (lower = safer to demote)
    pub last_updated:  Option<Instant>,
}

impl BlockStats {
    pub fn compute_aol(&self) -> f64 {
        if self.access_count == 0 || self.mlp == 0.0 {
            return 1.0; // unknown → assume critical
        }
        let raw = self.stall_cycles as f64 / (self.access_count as f64 * self.mlp);
        // Normalise: typical raw values 0..5000 cycles
        (raw / 5000.0).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Perf event handle (Linux perf_event_open wrapper)
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod perf {
    use std::os::unix::io::RawFd;

    pub const PERF_TYPE_RAW:          u32 = 4;
    // Intel Skylake/Cascade Lake: MEM_LOAD_L3_MISS_RETIRED.REMOTE_DRAM
    pub const OFFCORE_RESPONSE_EVENT: u64 = 0x01B7;  // configurable

    /// Open a hardware perf counter for the calling thread.
    pub fn open_event(event_config: u64) -> Option<RawFd> {
        use std::mem;
        #[repr(C)]
        struct PerfEventAttr {
            type_:       u32,
            size:        u32,
            config:      u64,
            sample_freq: u64,
            sample_type: u64,
            read_format: u64,
            flags:       u64,
            wakeup:      u32,
            bp_type:     u32,
            bp_addr:     u64,
            bp_len:      u64,
            branch_sample_type: u64,
            sample_regs_user:   u64,
            sample_stack_user:  u32,
            clockid:     i32,
            sample_regs_intr:   u64,
            aux_watermark: u32,
            sample_max_stack: u16,
            _reserved2:  u16,
        }
        let mut attr: PerfEventAttr = unsafe { mem::zeroed() };
        attr.type_  = PERF_TYPE_RAW;
        attr.size   = mem::size_of::<PerfEventAttr>() as u32;
        attr.config = event_config;
        attr.flags  = 1 << 0;  // disabled = false after enable

        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const _,
                0i32,   // pid = current
                -1i32,  // cpu = any
                -1i32,  // group_fd
                0u64,   // flags
            )
        };
        if fd < 0 { None } else { Some(fd as RawFd) }
    }

    /// Read counter value from a perf fd.
    pub fn read_counter(fd: RawFd) -> u64 {
        let mut val: u64 = 0;
        unsafe {
            libc::read(fd, &mut val as *mut u64 as *mut libc::c_void, 8);
        }
        val
    }
}

// ---------------------------------------------------------------------------
// AOL Profiler
// ---------------------------------------------------------------------------

pub struct AOLProfiler {
    stats:            Arc<Mutex<HashMap<u64, BlockStats>>>,
    sampling_interval: Duration,
    /// Callback called after each sweep with updated (block_id, aol_score) pairs.
    on_update: Option<Arc<dyn Fn(Vec<(u64, f64)>) + Send + Sync>>,
}

impl AOLProfiler {
    pub fn new(sampling_interval_ms: u64) -> Self {
        AOLProfiler {
            stats: Arc::new(Mutex::new(HashMap::new())),
            sampling_interval: Duration::from_millis(sampling_interval_ms),
            on_update: None,
        }
    }

    /// Register a callback for pushing AOL scores back to the C++ allocator.
    pub fn set_update_callback<F>(&mut self, cb: F)
    where F: Fn(Vec<(u64, f64)>) + Send + Sync + 'static
    {
        self.on_update = Some(Arc::new(cb));
    }

    // ------------------------------------------------------------------
    // Block registration / access recording
    // ------------------------------------------------------------------

    pub fn register_block(&self, block_id: u64) {
        let mut stats = self.stats.lock().unwrap();
        stats.entry(block_id).or_insert_with(|| BlockStats {
            block_id,
            ..Default::default()
        });
    }

    pub fn unregister_block(&self, block_id: u64) {
        self.stats.lock().unwrap().remove(&block_id);
    }

    /// Record an access event with measured stall cycles and MLP.
    pub fn record_access(
        &self,
        block_id:     u64,
        stall_cycles: u64,
        mlp:          f64,
    ) {
        let mut stats = self.stats.lock().unwrap();
        let entry = stats.entry(block_id).or_insert_with(|| BlockStats {
            block_id,
            ..Default::default()
        });
        entry.access_count += 1;
        entry.stall_cycles += stall_cycles;
        // Running average for MLP
        if entry.access_count == 1 {
            entry.mlp = mlp;
        } else {
            let n = entry.access_count as f64;
            entry.mlp = entry.mlp * (n - 1.0) / n + mlp / n;
        }
        entry.last_updated = Some(Instant::now());
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    pub fn get_aol_score(&self, block_id: u64) -> f64 {
        let stats = self.stats.lock().unwrap();
        stats.get(&block_id).map(|s| s.compute_aol()).unwrap_or(1.0)
    }

    pub fn get_all_scores(&self) -> Vec<(u64, f64)> {
        let stats = self.stats.lock().unwrap();
        stats.values().map(|s| (s.block_id, s.compute_aol())).collect()
    }

    pub fn get_block_stats(&self, block_id: u64) -> Option<BlockStats> {
        self.stats.lock().unwrap().get(&block_id).cloned()
    }

    // ------------------------------------------------------------------
    // Background sampling thread
    // ------------------------------------------------------------------

    /// Spawn the background sweep thread.  Returns a handle.
    pub fn spawn_sampler(self: Arc<Self>) -> thread::JoinHandle<()> {
        let profiler = Arc::clone(&self);
        thread::Builder::new()
            .name("aol-sampler".into())
            .spawn(move || {
                profiler.run_sampler();
            })
            .expect("Failed to spawn aol-sampler thread")
    }

    fn run_sampler(&self) {
        #[cfg(target_os = "linux")]
        let offcore_fd = perf::open_event(perf::OFFCORE_RESPONSE_EVENT);
        #[cfg(not(target_os = "linux"))]
        let offcore_fd: Option<i32> = None;
        let _ = offcore_fd; // suppress warning on non-Linux

        loop {
            thread::sleep(self.sampling_interval);
            self.sweep_and_update();
        }
    }

    fn sweep_and_update(&self) {
        let scores: Vec<(u64, f64)> = {
            let stats = self.stats.lock().unwrap();
            stats.values().map(|s| (s.block_id, s.compute_aol())).collect()
        };

        // Decay stale accesses (exponential decay over time)
        {
            let mut stats = self.stats.lock().unwrap();
            let now = Instant::now();
            for s in stats.values_mut() {
                if let Some(last) = s.last_updated {
                    let age_secs = now.duration_since(last).as_secs_f64();
                    // Half-life ≈ 2 s
                    let decay = (-age_secs * std::f64::consts::LN_2 / 2.0).exp();
                    s.stall_cycles = (s.stall_cycles as f64 * decay) as u64;
                    s.access_count = (s.access_count as f64 * decay) as u64;
                }
            }
        }

        if let Some(cb) = &self.on_update {
            cb(scores);
        }
    }
}

// ---------------------------------------------------------------------------
// Simulated PEBS sampler (when real HW counters are unavailable)
// ---------------------------------------------------------------------------

pub struct SimulatedPEBSSampler {
    rng_seed: u64,
}

impl SimulatedPEBSSampler {
    pub fn new(seed: u64) -> Self { SimulatedPEBSSampler { rng_seed: seed } }

    /// Simulate a stall-cycle sample for a block access.
    /// In production this is replaced by actual PEBS interrupt data.
    pub fn sample(&mut self, block_id: u64, tier: u8) -> (u64, f64) {
        // LCG RNG
        self.rng_seed = self.rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (self.rng_seed >> 33) as f64 / u32::MAX as f64;

        // T1 accesses: low stalls (~50–200 cycles), high MLP (4–8)
        // T2 accesses: high stalls (~500–2000 cycles), low MLP (1–2)
        let (stall_cycles, mlp) = if tier == 0 {
            (50 + (r * 150.0) as u64, 4.0 + r * 4.0)
        } else {
            (500 + (block_id % 10) * 150 + (r * 1000.0) as u64, 1.0 + r)
        };
        (stall_cycles, mlp)
    }
}

// ---------------------------------------------------------------------------
// C FFI bridge (called from C++ allocator)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn aol_profiler_create(sampling_interval_ms: u64) -> *mut AOLProfiler {
    let p = Box::new(AOLProfiler::new(sampling_interval_ms));
    Box::into_raw(p)
}

#[no_mangle]
pub extern "C" fn aol_profiler_destroy(ptr: *mut AOLProfiler) {
    if !ptr.is_null() { unsafe { drop(Box::from_raw(ptr)); } }
}

#[no_mangle]
pub extern "C" fn aol_profiler_register_block(ptr: *mut AOLProfiler, block_id: u64) {
    if let Some(p) = unsafe { ptr.as_ref() } { p.register_block(block_id); }
}

#[no_mangle]
pub extern "C" fn aol_profiler_record_access(
    ptr:          *mut AOLProfiler,
    block_id:     u64,
    stall_cycles: u64,
    mlp:          f64,
) {
    if let Some(p) = unsafe { ptr.as_ref() } {
        p.record_access(block_id, stall_cycles, mlp);
    }
}

#[no_mangle]
pub extern "C" fn aol_profiler_get_score(ptr: *mut AOLProfiler, block_id: u64) -> f64 {
    unsafe { ptr.as_ref() }.map_or(1.0, |p| p.get_aol_score(block_id))
}
