//! TEMPORARY hot-path instrumentation, behind the `hotprof` feature.
//!
//! Off by default and zero-cost when off: every macro below expands to just the
//! wrapped expression unless `hotprof` is enabled. Added 2026-10-06 to split
//! `next_batch` into descent / encode / other. Delete, or keep disabled.

/// A sentinel so a measurement can prove it is running the instrumented binary
/// rather than a stale one.
pub const SENTINEL: &str = "hotprof-2026-10-06-a";

#[cfg(feature = "hotprof")]
mod imp {
    use std::sync::atomic::{AtomicU64, Ordering};

    macro_rules! counters {
        ($($name:ident),* $(,)?) => {
            #[allow(non_upper_case_globals)]
            mod slots { use super::AtomicU64; $(pub static $name: AtomicU64 = AtomicU64::new(0);)* }
            pub const NAMES: &[&str] = &[$(stringify!($name)),*];
            pub fn get(name: &str) -> u64 {
                match name { $(stringify!($name) => slots::$name.load(Ordering::Relaxed),)* _ => 0 }
            }
            pub fn add(name: &str, v: u64) {
                match name { $(stringify!($name) => { slots::$name.fetch_add(v, Ordering::Relaxed); })* _ => {} }
            }
            pub fn reset() { $(slots::$name.store(0, Ordering::Relaxed);)* }
        };
    }

    // Nanosecond accumulators (ns_*) and plain event counts (n_*).
    counters!(
        ns_next_batch,
        ns_prepare,
        ns_encode,
        ns_resize,
        ns_commit,
        ns_submit,
        ns_apply,
        ns_expand_node,
        ns_backup,
        ns_seed_gumbel,
        ns_search_new,
        ns_create_child,
        ns_select_interior,
        ns_improved_policy,
        ns_sigma_completed,
        ns_result,
        ns_root_value,
        n_next_batch,
        n_evals,
        n_prepare,
        n_commit,
        n_search_new,
        n_create_child,
        n_select_interior,
        n_improved_policy,
        n_gumbel_draws,
        n_alloc_calls,
        n_alloc_bytes,
        n_alloc_calls_descent,
        n_alloc_bytes_descent,
    );

    #[inline]
    pub fn bump(name: &'static str, v: u64) {
        add(name, v);
    }

    #[inline]
    pub fn read(name: &str) -> u64 {
        get(name)
    }

    pub fn snapshot() -> Vec<(&'static str, u64)> {
        NAMES.iter().map(|&n| (n, get(n))).collect()
    }

    pub fn clear() {
        reset()
    }
}

#[cfg(feature = "hotprof")]
pub use imp::{bump, clear, read, snapshot};

#[cfg(not(feature = "hotprof"))]
pub fn snapshot() -> Vec<(&'static str, u64)> {
    Vec::new()
}
#[cfg(not(feature = "hotprof"))]
pub fn clear() {}
#[cfg(not(feature = "hotprof"))]
#[inline]
pub fn bump(_name: &'static str, _v: u64) {}
#[cfg(not(feature = "hotprof"))]
#[inline]
pub fn read(_name: &str) -> u64 {
    0
}

/// A global allocator that counts into `n_alloc_calls` / `n_alloc_bytes`.
/// Only useful with `hotprof`; harmless otherwise.
pub struct CountingAlloc;

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        bump("n_alloc_calls", 1);
        bump("n_alloc_bytes", l.size() as u64);
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        bump("n_alloc_calls", 1);
        bump("n_alloc_bytes", n as u64);
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}

/// Time an expression into the named ns accumulator.
#[cfg(feature = "hotprof")]
#[macro_export]
macro_rules! prof_time {
    ($name:literal, $e:expr) => {{
        let __t = std::time::Instant::now();
        let __r = $e;
        $crate::prof::bump($name, __t.elapsed().as_nanos() as u64);
        __r
    }};
}

#[cfg(not(feature = "hotprof"))]
#[macro_export]
macro_rules! prof_time {
    ($name:literal, $e:expr) => {
        $e
    };
}

/// Increment a named event counter.
#[cfg(feature = "hotprof")]
#[macro_export]
macro_rules! prof_count {
    ($name:literal, $v:expr) => {
        $crate::prof::bump($name, $v as u64)
    };
}

#[cfg(not(feature = "hotprof"))]
#[macro_export]
macro_rules! prof_count {
    ($name:literal, $v:expr) => {{
        let _ = $v;
    }};
}
