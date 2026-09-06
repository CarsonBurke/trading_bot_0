//! Measures both fused kernels against the composed-ATen forms they replace, at the real
//! training shapes, and writes `timexer_segment_fused_kernels.report.bin`.
//!
//! Its own binary rather than a `benchmark-timexer-segment` flag: the whole point of the
//! kernels living in a crate of their own is that they can be measured and adopted
//! independently, and a probe that needs no model, no corpus and no optimizer should not
//! have to build one. The job is still submitted as a `benchmark-timexer-segment` probe.

use anyhow::{ensure, Result};
use clap::Parser;
use fused_kernels::probe::{activation_saving, measure};
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use std::path::PathBuf;
use tch::{Cuda, Device};

#[derive(Parser)]
struct Args {
    /// Report generation directory.
    #[arg(long)]
    output: PathBuf,
    /// Batch rows. Tokens are `rows * origins`; 256 is the measured training batch.
    #[arg(long, default_value_t = 256)]
    rows: i64,
    #[arg(long, default_value_t = 375)]
    origins: i64,
    #[arg(long, default_value_t = 512)]
    d_model: i64,
    #[arg(long, default_value_t = 8)]
    heads: i64,
    #[arg(long, default_value_t = 2048)]
    ffn: i64,
    #[arg(long, default_value_t = 8)]
    layers: i64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        Cuda::is_available(),
        "the fused kernels are CUDA-only; this probe needs a device"
    );
    let device = Device::Cuda(0);
    let measurement = measure(
        device,
        args.rows,
        args.origins,
        args.d_model,
        args.heads,
        args.ffn,
        args.layers,
    );
    let peak_gbs = measurement.streaming_roof_gbs;
    let rows = &measurement.comparisons;

    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>7} {:>7} {:>11}",
        "kernel",
        "cmp fwd ms",
        "fus fwd ms",
        "cmp bwd ms",
        "fus bwd ms",
        "fus bwd k",
        "fwd GB/s",
        "bwd GB/s",
        "% fwd",
        "% bwd",
        "step ms cut"
    );
    // Only the additive rows: `QK norm + rotary` measures against the whole composed
    // sequence and already contains the rotary row's saving.
    let mut total_saved = 0.0;
    for row in rows {
        if row.additive {
            total_saved += row.step_ms_saved();
        }
        println!(
            "{:<16} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>9.0} {:>9.0} {:>7.1} {:>7.1} {:>11.1}",
            row.name,
            row.composed_forward_ms,
            row.fused_forward_ms,
            row.composed_backward_ms,
            row.fused_backward_ms,
            row.fused_backward_kernel_ms,
            row.fused_forward_gbs(),
            row.fused_backward_gbs(),
            100.0 * row.fused_forward_gbs() / peak_gbs,
            100.0 * row.fused_backward_gbs() / peak_gbs,
            row.step_ms_saved()
        );
    }
    println!(
        "streaming roof {peak_gbs:.0} GB/s (128-bit vectorized copy kernel, interleaved with every measurement above) | the kernels together cut {total_saved:.1} ms from a {} row step",
        args.rows
    );

    // Measured after the timings, behind its own `empty_cache`: allocating a `layers`-deep
    // retained chain would otherwise sit under every millisecond above.
    let memory = activation_saving(
        device,
        args.rows,
        args.origins,
        args.d_model,
        args.heads,
        args.layers,
    );
    println!(
        "QK-norm activation footprint over {} layers: composed peak {:.1} MiB, fused peak {:.1} MiB, saved {:.1} MiB | retained composed {:.1} MiB, fused {:.1} MiB, saved {:.1} MiB",
        memory.layers,
        memory.composed_peak_mib,
        memory.fused_peak_mib,
        memory.peak_saved_mib(),
        memory.composed_live_mib,
        memory.fused_live_mib,
        memory.retained_saved_mib()
    );
    let series: Vec<ReportSeries> = rows
        .iter()
        .flat_map(|row| {
            let name = row.name;
            [
                (
                    format!("{name}: composed forward milliseconds per layer"),
                    row.composed_forward_ms,
                ),
                (
                    format!("{name}: fused forward milliseconds per layer"),
                    row.fused_forward_ms,
                ),
                (
                    format!("{name}: composed backward milliseconds per layer (autograd harness included)"),
                    row.composed_backward_ms,
                ),
                (
                    format!("{name}: fused backward milliseconds per layer (autograd harness included)"),
                    row.fused_backward_ms,
                ),
                (
                    format!("{name}: fused backward kernel milliseconds per layer"),
                    row.fused_backward_kernel_ms,
                ),
                (
                    format!("{name}: fused forward achieved GB/s"),
                    row.fused_forward_gbs(),
                ),
                (
                    format!("{name}: fused backward achieved GB/s"),
                    row.fused_backward_gbs(),
                ),
                (
                    format!("{name}: fused forward percent of the measured streaming roof"),
                    100.0 * row.fused_forward_gbs() / peak_gbs,
                ),
                (
                    format!("{name}: fused backward percent of the measured streaming roof"),
                    100.0 * row.fused_backward_gbs() / peak_gbs,
                ),
                (
                    format!("{name}: step milliseconds saved over {} layers", args.layers),
                    row.step_ms_saved(),
                ),
            ]
        })
        .map(|(label, value)| ReportSeries {
            label,
            values: vec![value as f32],
        })
        .chain([
            ReportSeries {
                label: "measured streaming roof GB/s (128-bit vectorized copy kernel)".to_owned(),
                values: vec![peak_gbs as f32],
            },
            ReportSeries {
                label: "every kernel: step milliseconds saved".to_owned(),
                values: vec![total_saved as f32],
            },
            ReportSeries {
                label: format!(
                    "QK norm: composed allocator peak MiB over {} layers",
                    memory.layers
                ),
                values: vec![memory.composed_peak_mib as f32],
            },
            ReportSeries {
                label: format!(
                    "QK norm: fused allocator peak MiB over {} layers",
                    memory.layers
                ),
                values: vec![memory.fused_peak_mib as f32],
            },
            ReportSeries {
                label: "QK norm: allocator peak MiB saved".to_owned(),
                values: vec![memory.peak_saved_mib() as f32],
            },
            ReportSeries {
                label: format!(
                    "QK norm: composed retained MiB over {} layers",
                    memory.layers
                ),
                values: vec![memory.composed_live_mib as f32],
            },
            ReportSeries {
                label: format!("QK norm: fused retained MiB over {} layers", memory.layers),
                values: vec![memory.fused_live_mib as f32],
            },
            ReportSeries {
                label: "QK norm: retained MiB saved".to_owned(),
                values: vec![memory.retained_saved_mib() as f32],
            },
        ])
        .collect();

    std::fs::create_dir_all(&args.output)?;
    write_report(
        args.output.join("timexer_segment_fused_kernels.report.bin"),
        &Report {
            title: format!(
                "fused ReLU^2, packed rotary and QK-norm+rotary against the composed-ATen forms | {} rows, {} tokens, d_model {}, {} heads, ffn {}, {} layers | {peak_gbs:.0} GB/s measured streaming roof",
                args.rows,
                args.rows * args.origins,
                args.d_model,
                args.heads,
                args.ffn,
                args.layers
            ),
            x_label: Some("batch rows".to_owned()),
            y_label: Some(
                "named units (each label states its own; lower is better in milliseconds and MiB, higher in GB/s and in percent of the streaming roof)"
                    .to_owned(),
            ),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: vec![args.rows as u64],
                series,
            },
        },
    )?;
    println!(
        "fused kernel report: {}",
        args.output
            .join("timexer_segment_fused_kernels.report.bin")
            .display()
    );
    Ok(())
}
