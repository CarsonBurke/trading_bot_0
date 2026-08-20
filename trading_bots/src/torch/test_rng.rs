//! Serialization of libtorch's process-global RNG across concurrent tests.
//!
//! # The invariant
//!
//! `tch::manual_seed` does not create a generator. It mutates the *one*
//! generator libtorch keeps per process, and every thread in that process draws
//! from it: `Tensor::randn`, `Tensor::rand`, `Tensor::randint`, `normal_`,
//! `multinomial`, and every `nn::VarStore`-backed module constructor, whose
//! default parameter init is a draw. Rust's test harness runs `#[test]`
//! functions on several threads of one process, so by default every torch test
//! in this crate shares one mutable generator with every other torch test.
//!
//! Two consequences, both measured against the libtorch this crate links
//! (torch 2.13 from `.venv-fa4`):
//!
//! * a concurrent `manual_seed` on another test thread *rewinds* the stream a
//!   seeded test is reading; and
//! * a concurrent **draw** on another test thread — with no seeding anywhere in
//!   sight, e.g. a plain `nn::VarStore::new` plus a module constructor —
//!   *advances* that same stream and perturbs it just as effectively.
//!
//! The second point is why this module guards more than seeding. A lock taken
//! only by the tests that call `manual_seed` is a lock with a hole in it the
//! size of every model-building test in the crate: a mutex cannot protect a
//! global from code that does not take it. That was the original defect.
//! `TORCH_RNG_TEST_LOCK` used to be a private `static` inside
//! `torch::train::trainer`'s test module, where exactly two tests could reach
//! it, while twenty-odd tests in other modules reseeded the same global and
//! hundreds more drew from it — so the lock excluded only the two tests that
//! were already well behaved, and
//! `production_ppo_frontier_is_exact_across_interruption` failed in the full
//! suite while passing in isolation.
//!
//! # The protocol
//!
//! Every test that touches the torch RNG takes one of two guards:
//!
//! * [`exclusive`] — the test seeds the RNG, or depends on the exact values a
//!   seeded stream produces. That includes tests which never write
//!   `manual_seed` themselves but call production code that does (for example
//!   `Trainer::collect_rollout` and `pretrain::build_trainer` both reseed).
//!   Excludes every other participant, readers and writers alike.
//! * [`shared`] — the test only *consumes* the RNG and does not care what it
//!   draws: it builds a `VarStore`, or makes a throwaway `Tensor::randn` whose
//!   value is irrelevant to the assertion. Shared guards do not exclude one
//!   another, so this costs essentially no parallelism; it exists only to keep
//!   such a test out of an [`exclusive`] section.
//!
//! Two rules that are easy to get wrong:
//!
//! 1. **Bind the guard.** `let _guard = test_rng::exclusive();` holds it to the
//!    end of the scope. `let _ = test_rng::exclusive();` drops it *immediately*
//!    and protects nothing at all, while looking like it does.
//! 2. **Take it in the `#[test]` function, not in a shared helper.** Helpers
//!    such as `production_test_trainer` or muon's `train_adamw` are called
//!    several times per test; locking per call would release the generator
//!    between two runs the test then compares bit-for-bit, which is precisely
//!    the failure being fixed. The guard must live until the test's last
//!    RNG-consuming operation.
//!
//! The guards are not reentrant. Never take a second guard while holding one,
//! and never call a helper that takes one from inside a guarded test.
//!
//! # Panicking tests
//!
//! A test that fails while holding a guard must report as *one* failing test,
//! not as a cascade of confusing secondary failures in every other RNG test.
//! `parking_lot`'s locks have no poison state at all: unwinding drops the guard
//! and the next waiter simply acquires it. That is why there is no
//! `unwrap_or_else(PoisonError::into_inner)` recovery dance here — with
//! `std::sync` it would be mandatory, and forgetting it is exactly how one real
//! assertion failure becomes twenty. `parking_lot` is also the crate's
//! established lock (see `torch::env::earnings`, `macro_ind`, `momentum`).
//! The test `a_panic_while_holding_the_guard_does_not_wedge_the_lock` below
//! pins the behaviour down so a future switch back to `std::sync` cannot
//! regress it silently.

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// The one lock arbitrating libtorch's process-global generator in tests.
///
/// See the module documentation for the invariant. Prefer [`exclusive`] and
/// [`shared`] over locking this directly; they name the two halves of the
/// protocol, and the choice between them is the whole point.
pub(crate) static TORCH_RNG_TEST_LOCK: RwLock<()> = RwLock::new(());

/// Take exclusive ownership of the process-global torch RNG.
///
/// For tests that seed it, or that depend on the exact values a seeded stream
/// produces — including via production code that seeds internally.
pub(crate) fn exclusive() -> RwLockWriteGuard<'static, ()> {
    TORCH_RNG_TEST_LOCK.write()
}

/// Take shared access to the process-global torch RNG.
///
/// For tests that draw from it but do not care what they get. Shared holders
/// run concurrently with each other and only exclude [`exclusive`] holders.
pub(crate) fn shared() -> RwLockReadGuard<'static, ()> {
    TORCH_RNG_TEST_LOCK.read()
}

#[cfg(test)]
mod tests {
    use super::{exclusive, shared};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tch::{Device, Kind, Tensor};

    /// A short, cheap, bit-comparable read of the global stream.
    fn draw() -> Vec<u32> {
        Vec::<f32>::try_from(Tensor::randn([16], (Kind::Float, Device::Cpu)))
            .expect("draw from the global generator")
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn draws(chunks: usize) -> Vec<Vec<u32>> {
        (0..chunks).map(|_| draw()).collect()
    }

    /// The hazard this module exists for, demonstrated without a race.
    ///
    /// A draw performed on a *different thread*, joined before we continue,
    /// still shifts the stream this thread is reading. Deterministic: the join
    /// orders the two, so there is nothing to interleave and nothing to flake.
    ///
    /// If this test ever fails, libtorch has stopped sharing one generator
    /// across threads and `TORCH_RNG_TEST_LOCK` can be deleted along with every
    /// guard that takes it. Until then, the lock is load-bearing.
    #[test]
    fn another_threads_draw_disturbs_this_threads_seeded_stream() {
        let _torch_rng_guard = exclusive();

        tch::manual_seed(0x0BAD_5EED);
        let reference = draws(4);

        tch::manual_seed(0x0BAD_5EED);
        std::thread::spawn(|| {
            let _ = draw();
        })
        .join()
        .expect("foreign drawing thread");
        let observed = draws(4);

        assert_ne!(
            reference, observed,
            "a draw on another thread left this thread's seeded stream untouched: \
             libtorch's generator is no longer process-global, so TORCH_RNG_TEST_LOCK \
             and every guard that takes it are now dead weight and should be removed"
        );
    }

    /// The same hazard from an extra draw on *this* thread, which is why the
    /// guard must span the whole test rather than just the `manual_seed` call:
    /// one stray draw between two seeded segments is enough to break them apart.
    #[test]
    fn one_extra_draw_between_seeded_segments_changes_the_stream() {
        let _torch_rng_guard = exclusive();

        tch::manual_seed(0x5EED_0001);
        let reference = draws(4);

        tch::manual_seed(0x5EED_0001);
        let _intruder = draw();
        let observed = draws(4);

        assert_ne!(reference, observed);
    }

    /// The protocol actually excludes: a `shared` holder cannot enter while an
    /// `exclusive` holder is inside.
    ///
    /// The "did not acquire" half cannot false-fail — `RwLock` makes that
    /// acquisition impossible, not merely unlikely. It waits for the worker to
    /// announce it is at the lock before asserting, so it observes exclusion
    /// rather than a thread the scheduler simply had not started yet.
    #[test]
    fn a_shared_holder_waits_for_the_exclusive_section_to_end() {
        let approaching = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(AtomicBool::new(false));
        let worker = {
            let at_the_lock = Arc::clone(&approaching);
            let acquired = Arc::clone(&entered);
            let guard = exclusive();

            let handle = std::thread::spawn(move || {
                at_the_lock.store(true, Ordering::SeqCst);
                let _shared = shared();
                acquired.store(true, Ordering::SeqCst);
            });

            while !approaching.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            // The worker is now one statement away from `shared()`. Give it
            // every chance to get in; it must not.
            for _ in 0..1_000 {
                std::thread::yield_now();
            }
            assert!(
                !entered.load(Ordering::SeqCst),
                "a shared guard was granted while an exclusive guard was held"
            );

            drop(guard);
            handle
        };

        worker.join().expect("shared-guard thread");
        assert!(
            entered.load(Ordering::SeqCst),
            "the shared guard was never granted after the exclusive section ended"
        );
    }

    /// One failing test must stay one failing test.
    ///
    /// A panic while the guard is held unwinds through the guard's `Drop`, and
    /// the lock is immediately usable again with no poison state to unwrap.
    /// Were this lock ever moved back to `std::sync`, every subsequent
    /// `.lock().unwrap()` would panic instead, turning a single real assertion
    /// failure into a suite-wide cascade — this test is what would catch that.
    #[test]
    fn a_panic_while_holding_the_guard_does_not_wedge_the_lock() {
        std::thread::spawn(|| {
            let _held = exclusive();
            panic!(
                "deliberate panic from a_panic_while_holding_the_guard_does_not_wedge_the_lock; \
                 this message in the test log is expected"
            );
        })
        .join()
        .expect_err("the panicking thread must actually panic");

        // Reacquisition must not panic. It may well *block* — sibling tests
        // legitimately hold shared guards while this one runs, which is why
        // `try_write` would be the wrong probe here: it reports contention, not
        // damage. Under `std::sync` this would panic on the poison instead.
        let reacquired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(exclusive());
            drop(shared());
        }));
        assert!(
            reacquired.is_ok(),
            "the lock could not be taken after a holder panicked: one real failure \
             will now cascade into every other RNG test"
        );
    }

    // ---------------------------------------------------------------------
    // Tripwire for new, unguarded seeding.
    //
    // This one reads the crate's source, which is normally the wrong tool. It
    // is here because the behavioural alternative cannot be made to work: a
    // test can only *observe* a foreign unguarded `manual_seed` if that foreign
    // test happens to run while this one is holding the lock, and libtest
    // offers no control over which tests overlap. Detection would be a
    // coin-flip on scheduling, and a probabilistic guard against a bug whose
    // whole signature is "passes in isolation, fails in the suite" is worse
    // than no guard, because it teaches readers to re-run until green. The
    // tests above cover everything that *can* be established behaviourally:
    // that the generator is shared across threads, that a single stray draw
    // shifts it, that the lock excludes, and that a panicking holder does not
    // wedge it. What is left — "did someone add a seeding site and forget the
    // guard" — is a property of the source, so it is checked against the
    // source.
    // ---------------------------------------------------------------------

    /// Production functions that seed the global torch RNG on purpose.
    ///
    /// These run inside the real training binary, where there is one thread of
    /// control and reseeding at a causal boundary is the intended behaviour. A
    /// test that reaches any of them is seeding whether it says so or not, and
    /// must take an [`exclusive`] guard — which [`SEEDING_CALLEES`] enforces
    /// from the caller's side.
    ///
    /// Keyed by `(file suffix, function name)` rather than a bare name: `new`,
    /// `update` and `planner` are far too common to exempt crate-wide, and this
    /// crate has 76 other `fn new`s plus a production `HlGaussBins::planner`.
    const PRODUCTION_SEEDING_FNS: &[(&str, &str)] = &[
        // `Trainer::new`. Also matches the `MiniPpo::new` test helper in the
        // same file, which seeds too and whose callers hold the guard.
        ("torch/train/trainer.rs", "new"),
        ("torch/train/rollout.rs", "collect_rollout"), // per-update reseed
        ("torch/train/pretrain.rs", "build_trainer"),  // incl. `manual_seed_all`
        ("torch/planner/runner.rs", "train_planner"),  // per-update reseed
        ("torch/train/horizon.rs", "run_horizon_sweep"), // per-replicate reseed
    ];

    /// Test helpers that seed.
    ///
    /// Each is called several times within a single test, so the guard belongs
    /// to the calling `#[test]` and cannot live in the helper — module docs,
    /// rule 2.
    const TEST_HELPER_SEEDING_FNS: &[(&str, &str)] = &[
        ("torch/train/trainer.rs", "production_test_trainer"),
        ("torch/train/trainer.rs", "update"), // `MiniPpo::update`
        ("torch/optim/muon.rs", "train_adamw"),
        ("torch/optim/muon.rs", "train_muon"),
        ("torch/optim/muon.rs", "train_muon_bf16"),
        ("torch/planner/mod.rs", "planner"),
        ("torch/world_model.rs", "wake_projections"),
        ("torch/train/pretrain.rs", "drifting_beliefs"),
        // A parameterized body shared by two `#[test]` wrappers, which is why
        // it is not itself a test.
        (
            "torch/model/stream/live.rs",
            "non_uniform_stream_matches_shifted_full_history",
        ),
        // `Fixture::new`, the horizon test fixture: two per differential test.
        ("torch/train/horizon.rs", "new"),
        ("torch/train/growth.rs", "seeded_perturbed_head"), // growth probe head
        ("torch/train/growth.rs", "probe_beliefs"),         // growth probe input
        ("torch/train/skill.rs", "seeded_perturbed_head"),  // skill probe head
    ];

    /// Calls that reach a seeding function.
    ///
    /// Allowlisting a seeding function must not make its *callers* unchecked:
    /// "the well-behaved code is exempt, so nobody else has to care" is the
    /// exact shape of the bug this module fixes. Any `#[test]` naming one of
    /// these is seeding, and needs [`exclusive`].
    const SEEDING_CALLEES: &[&str] = &[
        "Trainer::new(",
        "collect_rollout(",
        "build_trainer(",
        "train_planner(",
        "production_test_trainer(",
        "MiniPpo::new(",
        "train_adamw(",
        "train_muon(",
        "train_muon_bf16(",
        "run_horizon_sweep(",
        // Also matches `RolloutFixture::new` in `planner/runner.rs`, which
        // draws through a model constructor and whose callers hold the guard.
        "Fixture::new(",
        // Named apart from `trade_bench.rs`'s unseeded `perturbed_head`, which
        // only draws and whose callers hold `shared` on purpose.
        "seeded_perturbed_head(",
        "probe_beliefs(",
        "planner();",
        "wake_projections(",
        "write_fixture(",
        "drifting_beliefs(",
        "non_uniform_stream_matches_shifted_full_history(",
    ];

    /// Calls that draw from the global generator without seeding it.
    ///
    /// Seeding is the loud half of the hazard; this is the quiet half. A test
    /// that merely draws still advances the one shared stream and perturbs
    /// whichever seeded test is running beside it, so it needs a guard just the
    /// same. The constructors and `Init::` variants are here because parameter
    /// initialization is a draw — building a model is drawing, even though
    /// nothing in the call says so.
    const DRAW_APIS: &[&str] = &[
        "Tensor::randn(",
        "Tensor::rand(",
        "Tensor::randint(",
        ".randn(",
        ".randn_standard(",
        "randn_like(",
        "rand_like(",
        ".normal_(",
        ".uniform_(",
        ".bernoulli(",
        ".multinomial(",
        "randperm(",
        "Init::Randn",
        "Init::Uniform",
        "Init::Kaiming",
        "nn::linear(",
        "BarModules::new(",
        "BarEmissionHead::new(",
        "BarDynamics::new(",
        "BarTrunk::new(",
        "TradingModel::new(",
        "TradingModel::new_with_config(",
        "WorldModelPlanner::new(",
    ];

    /// Everything left of a line comment.
    ///
    /// Both scans run on this rather than the raw line, so that neither a
    /// commented-out seed nor prose mentioning `test_rng::` can move the
    /// verdict.
    fn code_of(line: &str) -> &str {
        line.split("//").next().unwrap_or(line)
    }

    /// `Some("exclusive" | "shared")` when the line takes a guard and *binds*
    /// it.
    ///
    /// The binding is part of the pattern on purpose. `let _ = exclusive();`
    /// drops the guard immediately and protects nothing, which the module docs
    /// call the easiest mistake to make here; a check that accepted it would
    /// wave through the very thing it exists to catch.
    fn guard_kind(line: &str) -> Option<&'static str> {
        let rest = code_of(line)
            .split_once("let _torch_rng_guard = test_rng::")?
            .1;
        if rest.starts_with("exclusive()") {
            Some("exclusive")
        } else if rest.starts_with("shared()") {
            Some("shared")
        } else {
            None
        }
    }

    fn holds(body: &[&str], kind: &str) -> bool {
        body.iter()
            .any(|line| guard_kind(line).is_some_and(|found| found == kind))
    }

    fn holds_any_guard(body: &[&str]) -> bool {
        body.iter().any(|line| guard_kind(line).is_some())
    }

    /// The name in a `fn` header, whatever qualifiers precede it.
    ///
    /// Missing a header shape is not a harmless gap: the seed scan walks
    /// backwards to the nearest header it recognizes, so an unrecognized one
    /// silently attributes a seed to the *previous* function and lets it
    /// inherit that function's guard.
    fn fn_name(line: &str) -> Option<&str> {
        let mut rest = code_of(line).trim_start();
        if let Some(after) = rest.strip_prefix("pub") {
            rest = match after.strip_prefix('(') {
                Some(scoped) => scoped.split_once(')').map_or(after, |(_, tail)| tail),
                None if after.starts_with(char::is_whitespace) => after,
                None => rest,
            }
            .trim_start();
        }
        loop {
            let stripped = ["default ", "const ", "async ", "unsafe "]
                .iter()
                .find_map(|keyword| rest.strip_prefix(keyword));
            match stripped {
                Some(tail) => rest = tail.trim_start(),
                None => break,
            }
        }
        if let Some(after) = rest.strip_prefix("extern ") {
            let after = after.trim_start();
            rest = match after.strip_prefix('"') {
                Some(abi) => abi.split_once('"').map_or(after, |(_, tail)| tail),
                None => after,
            }
            .trim_start();
        }
        let name = rest.strip_prefix("fn ")?.trim_start();
        Some(
            name.split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default(),
        )
    }

    fn rust_sources(root: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).expect("read crate source directory") {
            let path = entry.expect("source dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    /// Crate sources, minus this file.
    ///
    /// This module names the seeding APIs in prose, in its inventories and in
    /// its own guarded tests; scanning it would only ever find itself. Its four
    /// seeding lines both sit inside tests that hold `exclusive`.
    fn scannable_sources() -> Vec<PathBuf> {
        let mut sources = Vec::new();
        rust_sources(
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
            &mut sources,
        );
        sources.retain(|path| !path.ends_with("torch/test_rng.rs"));
        sources.sort();
        assert!(
            sources.len() > 20,
            "found only {} source files; the walk is broken, not the crate",
            sources.len()
        );
        sources
    }

    fn declared(inventory: &[(&str, &str)], path: &Path, function: &str) -> bool {
        let path = path.to_string_lossy().replace('\\', "/");
        inventory
            .iter()
            .any(|(file, name)| *name == function && path.ends_with(file))
    }

    /// The body of each `#[test]` in `lines`, as `(name, header, end)` line
    /// indices spanning the `fn` header to its closing brace.
    ///
    /// The closer is found by indentation rather than by counting braces,
    /// because brace counting trips over every `{}` in a format string — of
    /// which this crate has plenty. The source is rustfmt-formatted, so a
    /// function's closer is the first line that is exactly its own indentation
    /// followed by `}`.
    fn test_bodies(lines: &[&str]) -> Vec<(String, usize, usize)> {
        let mut bodies = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            // `#[test]`, `#[test] // slow`, `#[tokio::test]`.
            let attribute = code_of(line).trim();
            if !attribute.starts_with("#[") || !attribute.ends_with("test]") {
                continue;
            }
            // Skip any further attributes (`#[should_panic]`, `#[ignore]`).
            let Some(header) = (index + 1..lines.len()).find(|&i| {
                let trimmed = lines[i].trim_start();
                !trimmed.starts_with('#') && !trimmed.is_empty()
            }) else {
                continue;
            };
            let Some(name) = fn_name(lines[header]) else {
                continue;
            };
            let indent = &lines[header][..lines[header].len() - lines[header].trim_start().len()];
            let closer = format!("{indent}}}");
            let end = (header + 1..lines.len())
                .find(|&i| lines[i] == closer)
                .unwrap_or(lines.len() - 1);
            bodies.push((name.to_owned(), header, end));
        }
        bodies
    }

    #[test]
    fn every_torch_rng_seeding_site_is_guarded_or_declared() {
        let mut unguarded = Vec::new();
        for path in &scannable_sources() {
            let text = std::fs::read_to_string(path).expect("read source file");
            if !text.contains("manual_seed") {
                continue;
            }
            let lines: Vec<&str> = text.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                // Matched unqualified so that `tch::manual_seed`,
                // `tch::Cuda::manual_seed`, `manual_seed_all` and a bare
                // `manual_seed` imported via `use` are all caught.
                if !code_of(line).contains("manual_seed") {
                    continue;
                }
                let Some((header, function)) = lines[..=index]
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(at, text)| fn_name(text).map(|name| (at, name)))
                else {
                    unguarded.push(format!("{}:{} (outside any fn)", path.display(), index + 1));
                    continue;
                };
                // The guard has to be held *before* the seed lands, so it must
                // appear between the enclosing function's header and this line.
                // This is the check that matters: it verifies a bound guard is
                // really there rather than trusting a name.
                if holds(&lines[header..index], "exclusive")
                    || declared(PRODUCTION_SEEDING_FNS, path, function)
                    || declared(TEST_HELPER_SEEDING_FNS, path, function)
                {
                    continue;
                }
                unguarded.push(format!(
                    "{}:{} in fn `{function}`",
                    path.display(),
                    index + 1
                ));
            }
        }

        assert!(
            unguarded.is_empty(),
            "these call sites seed libtorch's process-global RNG without holding \
             `test_rng::exclusive()`, and are not declared in PRODUCTION_SEEDING_FNS or \
             TEST_HELPER_SEEDING_FNS:\n  {}\n\n\
             The generator is shared by every test thread in the process, so an unguarded \
             seed rewinds the stream of whatever seeded test happens to be running beside \
             it. If this is test code, take `let _torch_rng_guard = \
             crate::torch::test_rng::exclusive();` as the first statement of the `#[test]` \
             and hold it to the end. If the seed lives in a helper that a test calls several \
             times, guard each calling `#[test]` instead and add the helper to \
             TEST_HELPER_SEEDING_FNS. See torch/test_rng.rs.",
            unguarded.join("\n  ")
        );
    }

    /// The quiet half of the invariant, checked the same way and for the same
    /// reason as the seeding tripwire above.
    ///
    /// This is deliberately a *tripwire*, not a proof. It sees draws and
    /// seeding calls written directly in a `#[test]` body; a draw reached only
    /// through a test helper this file does not know about still slips past.
    /// Closing that completely would mean an allowlist of every drawing helper
    /// in the crate, which rots faster than it protects. The module docs carry
    /// the full invariant; this catches the ways it actually gets broken.
    #[test]
    fn every_test_that_touches_the_torch_rng_holds_a_guard() {
        let mut findings = Vec::new();
        let mut torch_bodies = 0usize;
        let mut guarded_drawers = 0usize;

        for path in &scannable_sources() {
            let text = std::fs::read_to_string(path).expect("read source file");
            if !text.contains("#[test]") && !text.contains("::test]") {
                continue;
            }
            let is_torch = path.to_string_lossy().replace('\\', "/").contains("/torch/");
            let lines: Vec<&str> = text.lines().collect();
            for (name, header, end) in test_bodies(&lines) {
                let body = &lines[header..=end];
                torch_bodies += usize::from(is_torch);

                // A test that reaches a known seeding function needs the write
                // guard, not merely some guard.
                let seeds = body.iter().enumerate().find_map(|(offset, line)| {
                    let code = code_of(line);
                    SEEDING_CALLEES
                        .iter()
                        .find(|callee| code.contains(*callee))
                        .map(|callee| (offset, *callee))
                });
                if let Some((offset, callee)) = seeds {
                    if !holds(body, "exclusive") {
                        findings.push(format!(
                            "{}:{} in `{name}` reaches `{callee}`, which seeds, but holds no \
                             exclusive guard",
                            path.display(),
                            header + offset + 1
                        ));
                        continue;
                    }
                }

                let draw = body.iter().enumerate().find_map(|(offset, line)| {
                    let code = code_of(line);
                    DRAW_APIS
                        .iter()
                        .find(|api| code.contains(*api))
                        .map(|api| (offset, *api))
                });
                let Some((offset, api)) = draw else {
                    continue;
                };
                if holds_any_guard(body) {
                    guarded_drawers += 1;
                    continue;
                }
                findings.push(format!(
                    "{}:{} in `{name}` draws via `{api}` but holds no guard",
                    path.display(),
                    header + offset + 1
                ));
            }
        }

        // A scan that parses nothing also reports nothing. These floors make a
        // vacuous pass impossible: if `test_bodies` stops finding bodies, or
        // `DRAW_APIS` drifts from how this crate draws, the scan fails loudly
        // instead of quietly approving everything. They are scoped to `torch/`
        // so that a parse break confined to these modules cannot hide behind
        // the crate's many non-torch tests.
        assert!(
            torch_bodies > 150,
            "only {torch_bodies} torch test bodies parsed; the scan is broken, not the crate"
        );
        assert!(
            guarded_drawers > 40,
            "only {guarded_drawers} guarded drawing tests recognized; DRAW_APIS has \
             drifted from how this crate draws, so the scan is no longer checking anything"
        );

        assert!(
            findings.is_empty(),
            "these tests touch libtorch's process-global RNG without the right \
             `test_rng` guard:\n  {}\n\n\
             Drawing is not harmless: it advances the one generator every test thread \
             shares, which is enough to break a seeded test running beside it. Take \
             `let _torch_rng_guard = crate::torch::test_rng::shared();` as the first \
             statement if the drawn values do not matter to the assertions, or \
             `::exclusive()` if the test seeds or needs a reproducible stream. See \
             torch/test_rng.rs.",
            findings.join("\n  ")
        );
    }
}
