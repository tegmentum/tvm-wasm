//! `tvm-test-harness` — reusable benchmarking primitives.
//!
//! What this gives you:
//!
//! - A [`Workload`] trait you implement for your code under test.
//! - A [`time_workload`] runner that handles warmup, sampling, percentiles.
//! - A [`Sample`] struct serializable to JSON for downstream tooling
//!   (the `bench-framework/plot.py` script consumes the same shape).
//! - A non-parametric [`mann_whitney_u`] for variant comparisons.
//!
//! Use it when you want to measure your own TVM-related workload through
//! the same statistical pipeline that `tvm-bench-runner` uses.

use std::time::{Duration, Instant};

use serde::Serialize;

/// What you implement for each thing you want to measure.
pub trait Workload {
    /// Called once before any timed iteration. Set up state.
    fn setup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    /// One timed iteration. Should be fast (target: tens of µs to ms).
    /// The runner calls this `warmup + samples` times.
    fn iter(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    /// Called once after all iterations. Tear down.
    fn teardown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub warmup_rounds: usize,
    pub samples: usize,
    /// Optional payload size for throughput calculations. If 0, throughput
    /// is reported as 0; otherwise used to compute GiB/s.
    pub payload_bytes: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            warmup_rounds: 5,
            samples: 50,
            payload_bytes: 0,
        }
    }
}

/// One row of measurements. Serialize to JSON for offline analysis.
#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub label: String,
    pub samples: usize,
    pub warmup: usize,
    pub mean_ns: f64,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    pub min_ns: u128,
    pub max_ns: u128,
    pub stddev_ns: f64,
    pub throughput_gib_per_s: f64,
    /// Raw timings, useful for offline statistics. Hide via
    /// `#[serde(skip_serializing_if)]` if writing for end users.
    pub raw_ns: Vec<u128>,
}

/// Run a workload through warmup + sampling. Returns timings; pair with
/// [`summarize`] to get a `Sample`.
pub fn time_workload<W: Workload>(
    workload: &mut W,
    cfg: RunConfig,
) -> Result<Vec<Duration>, Box<dyn std::error::Error>> {
    workload.setup()?;
    for _ in 0..cfg.warmup_rounds {
        workload.iter()?;
    }
    let mut timings = Vec::with_capacity(cfg.samples);
    for _ in 0..cfg.samples {
        let start = Instant::now();
        workload.iter()?;
        timings.push(start.elapsed());
    }
    workload.teardown()?;
    Ok(timings)
}

/// Build a `Sample` from raw timings.
pub fn summarize(label: impl Into<String>, timings: Vec<Duration>, cfg: RunConfig) -> Sample {
    let mut sorted = timings.clone();
    sorted.sort();
    let n = sorted.len();
    let raw: Vec<u128> = timings.iter().map(|d| d.as_nanos()).collect();
    let mean_ns = raw.iter().map(|n| *n as f64).sum::<f64>() / n as f64;
    let variance = raw
        .iter()
        .map(|n| (*n as f64 - mean_ns).powi(2))
        .sum::<f64>()
        / n as f64;
    let stddev_ns = variance.sqrt();
    let throughput_gib_per_s = if mean_ns > 0.0 && cfg.payload_bytes > 0 {
        (cfg.payload_bytes as f64) / (mean_ns / 1e9) / (1u64 << 30) as f64
    } else {
        0.0
    };
    Sample {
        label: label.into(),
        samples: n,
        warmup: cfg.warmup_rounds,
        mean_ns,
        p50_ns: percentile(&sorted, 50.0),
        p95_ns: percentile(&sorted, 95.0),
        p99_ns: percentile(&sorted, 99.0),
        min_ns: sorted.first().map(|d| d.as_nanos()).unwrap_or(0),
        max_ns: sorted.last().map(|d| d.as_nanos()).unwrap_or(0),
        stddev_ns,
        throughput_gib_per_s,
        raw_ns: raw,
    }
}

fn percentile(sorted: &[Duration], p: f64) -> u128 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)].as_nanos()
}

/// Mann-Whitney U statistic, normalized to [0, 1]. 0.5 = no difference;
/// values toward 0/1 indicate one sample stochastically dominates the
/// other. Useful as a non-parametric significance hint when comparing
/// variants.
pub fn mann_whitney_u(a: &[u128], b: &[u128]) -> f64 {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    if n_a == 0.0 || n_b == 0.0 {
        return 0.5;
    }
    let mut combined: Vec<(u128, usize)> = a
        .iter()
        .map(|v| (*v, 0))
        .chain(b.iter().map(|v| (*v, 1)))
        .collect();
    combined.sort_by_key(|p| p.0);
    let mut ranks = vec![0.0f64; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j + 1 < combined.len() && combined[j + 1].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let mut rank_sum_a = 0.0f64;
    for (idx, (_, side)) in combined.iter().enumerate() {
        if *side == 0 {
            rank_sum_a += ranks[idx];
        }
    }
    let u_a = rank_sum_a - n_a * (n_a + 1.0) / 2.0;
    u_a / (n_a * n_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyWorkload(usize);
    impl Workload for DummyWorkload {
        fn iter(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            self.0 = self.0.wrapping_add(1);
            Ok(())
        }
    }

    #[test]
    fn time_then_summarize() {
        let cfg = RunConfig {
            warmup_rounds: 2,
            samples: 10,
            payload_bytes: 1024,
        };
        let timings = time_workload(&mut DummyWorkload(0), cfg).unwrap();
        assert_eq!(timings.len(), 10);
        let s = summarize("dummy", timings, cfg);
        assert_eq!(s.samples, 10);
        assert!(s.mean_ns >= 0.0);
        assert!(s.p99_ns >= s.p50_ns);
        assert!(s.throughput_gib_per_s > 0.0);
    }

    #[test]
    fn mann_whitney_obvious_difference() {
        let a = vec![100u128, 110, 105, 95, 102];
        let b = vec![1000u128, 1100, 1050, 950, 1020];
        let u = mann_whitney_u(&a, &b);
        assert!(u < 0.05, "a clearly dominates b but U was {u}");
    }
}
