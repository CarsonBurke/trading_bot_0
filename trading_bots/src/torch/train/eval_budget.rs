use anyhow::{ensure, Result};

/// Fail closed before corpus/model/GPU initialization when an evaluation exceeds its
/// command-specific minute-scale work budget. Long evaluations require an explicit CLI opt-in.
pub fn enforce(
    command: &str,
    work: &str,
    estimated_units: u64,
    short_limit: u64,
    allow_long_eval: bool,
) -> Result<()> {
    ensure!(
        allow_long_eval || estimated_units <= short_limit,
        "{command} estimates {estimated_units} {work}, above the short-evaluation budget of \
         {short_limit}. Reduce the evaluation grid/windows/context or pass --allow-long-eval \
         to explicitly authorize a long evaluation"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_evaluations_fail_closed_without_explicit_opt_in() {
        let error = enforce("eval", "work units", 101, 100, false)
            .expect_err("oversized implicit evaluation must fail");
        assert!(error.to_string().contains("--allow-long-eval"));
        enforce("eval", "work units", 101, 100, true)
            .expect("explicit long-evaluation opt-in permits the work");
    }
}
