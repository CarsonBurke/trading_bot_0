// For burn Wgpu
#![recursion_limit = "256"]
#![feature(f16)]
#![feature(stdarch_x86_avx512_bf16)]

pub mod agent;
pub mod charts;
pub mod constants;
pub mod data;
pub mod genetic;
pub mod history;
pub mod neural_net;
pub mod strategies;
pub mod torch;
pub mod types;
pub mod utils;

/// Pin libtorch's thread pools before the test harness runs anything.
///
/// libtorch sizes its INTRA-OP pool at one thread per PHYSICAL core - measured 12 on this
/// 24-thread box, not 24 - so ONE single-threaded test doing ONE forward pass through the
/// 10-layer / 512-dim bar trunk reaches 12 cores by itself, before the interop pool and rayon
/// are counted at all. `cargo test -- --test-threads=N` bounds the harness, not those pools,
/// so it cannot cap the load: a lib test binary was observed at 2086% CPU with a load average
/// of 159, which is 12 intra-op plus interop plus rayon plus concurrent harness threads. That
/// decomposition is why this calls [`torch::train::pretrain::configure_threads`], which pins
/// interop as well, rather than `tch::set_num_threads` alone.
///
/// `TORCH_NUM_THREADS` is this repo's own convention, NOT a libtorch variable — nothing in
/// tch or libtorch reads it. It binds only where repo code explicitly calls
/// `tch::set_num_threads`, which before this constructor happened in production entry points
/// and nowhere else, so the variable measured as a no-op in every test binary. This is the
/// call that makes it bind.
///
/// An ELF constructor rather than a `Once` in a test helper, because a helper can be
/// forgotten: `.init_array` entries run at load time, before `main` and therefore before the
/// harness has enumerated a single test, so a test added later is capped whether or not its
/// author knows this exists. It is also the only moment at which the cap is guaranteed to
/// take — `tch::set_num_interop_threads` is a one-shot compare-exchange inside libtorch that
/// raises once anything has read or set it, which is why
/// [`torch::train::pretrain::configure_threads`] is documented as entry-point-only work. Here
/// nothing has run yet: libtorch's own initializers belong to a shared-library dependency and
/// have already completed, and no Rust code has touched a tensor.
///
/// Consequences worth knowing:
/// * Measured by linking libtorch directly and reading `at::get_num_threads`: an unpinned
///   binary reports 12 even with `TORCH_NUM_THREADS=1` set, because that variable is this
///   repo's own convention and nothing reads it without this call; it reports 1 under
///   `OMP_NUM_THREADS=1`; and with this constructor it reports 1 under no environment at all.
/// * An explicit `set_num_threads` overrides `OMP_NUM_THREADS` in BOTH directions, so
///   `OMP_NUM_THREADS=1 TORCH_NUM_THREADS=4` yields 4 here, where it yields 1 without this.
///   Do not give the two variables conflicting values for this binary.
/// * That upward override is deliberately NOT clamped to the OMP-derived default. This
///   constructor sets POLICY - the default is one thread and an explicit request is honoured -
///   whereas the in-fixture caps in `pretrain` and `pretrain_reports` impose a CEILING on top
///   of whatever policy applies, which is why those take `.min(tch::get_num_threads())` and
///   can only ever lower. Clamping here would instead make `TORCH_NUM_THREADS` a no-op again
///   for anyone who exports `OMP_NUM_THREADS=1` globally, which is the exact confusion this
///   constructor exists to end.
/// * A test must not call a production entry point (`pretrain`, `pretrain_candles`,
///   `pretrain_trade`) in this process: that is a second `configure_threads`, and the interop
///   setter raises on the second call. Two entry points in one process already aborted for
///   that reason before this existed; no current test calls one.
/// * Test-only. Production keeps its own `configure_threads` calls and its behaviour is
///   unchanged.
/// * `.init_array` is ELF, and this crate builds only on Linux. Deliberately not `cfg`-gated
///   to one target: a cap that silently degrades to a no-op on a platform someone ports to is
///   worse than a missing section that fails loudly.
#[cfg(test)]
#[used]
#[link_section = ".init_array"]
static PIN_TORCH_THREADS: extern "C" fn() = pin_torch_threads;

/// What the constructor actually pinned the intra-op pool to, or `0` if it never ran.
///
/// Captured HERE, pre-main, rather than read back inside the test. The in-fixture caps in
/// `pretrain` and `pretrain_reports` also call `set_num_threads`, so a value read at test time
/// would depend on which fixtures happened to have run first under `--test-threads=N` and the
/// assertion below would be order-dependent. A snapshot taken before `main` cannot be.
#[cfg(test)]
static PINNED_THREADS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(test)]
extern "C" fn pin_torch_threads() {
    torch::train::pretrain::configure_threads();
    PINNED_THREADS.store(tch::get_num_threads(), std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod thread_pin_tests {
    /// The cap is a property of the BINARY, not of any test that remembers to ask for it.
    ///
    /// Fails if the constructor stops running - `#[used]` dropped, the section renamed, the
    /// static garbage-collected by a future linker flag - and fails if libtorch ever stops
    /// honouring a pre-main `set_num_threads`. Either way the whole suite would silently go
    /// back to one thread per core, which is what got a test binary killed at 2086% CPU.
    #[test]
    fn the_constructor_pinned_the_pool_before_any_test_ran() {
        let requested = std::env::var("TORCH_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(1);
        let observed = super::PINNED_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        assert_ne!(
            observed, 0,
            "the .init_array constructor never ran, so this test binary is NOT capped and \
             every tch test in it will fan out to one thread per core"
        );
        assert_eq!(
            observed, requested,
            "the constructor ran but the intra-op pool did not take the pin: asked for \
             {requested} thread(s), libtorch reported {observed}"
        );
    }
}
