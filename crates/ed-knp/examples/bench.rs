//! Release-mode step time on the default grid: `cargo run --release -p ed-knp --example bench`.
//!
//! Load: 200 compartments along a vertical line, with K⁺/Na⁺ currents and a capacitive
//! current that change every step (redeposited after `clear_sources`), as the neuron coupling
//! will do. As in a real neuron, each compartment carries a net current (the lower half
//! sources, the upper half sinks) that sums to zero over the cell. `STEPS` sets the number of
//! timed steps per run; the median of five runs is reported.
use ed_core::Simulation;
use ed_knp::{KnpConfig, KnpSolver};
use std::time::Instant;

fn step(s: &mut KnpSolver) {
    s.clear_sources();
    let t = s.time_ms();
    for c in 0..200 {
        let pos = [500.0, 500.0, 50.0 + 2.0 * c as f64];
        let i = 0.5 * (1.0 + (0.3 * t + 0.1 * c as f64).sin());
        let net = if c < 100 { 0.5 } else { -0.5 } * (0.3 * t).cos();
        s.add_ion_current(pos, 1, i);
        s.add_ion_current(pos, 0, -0.9 * i);
        s.add_capacitive_current(pos, net - 0.1 * i);
    }
    s.step(1);
}

fn main() {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    assert!(steps > 0, "STEPS must be positive");
    let mut s = KnpSolver::new(&KnpConfig::default());
    (0..20).for_each(|_| step(&mut s));
    let (mut times, mut iters) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..steps {
            step(&mut s);
            iters.push(s.last_solve().iterations);
        }
        times.push(t.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    times.sort_by(f64::total_cmp);
    let n = s.grid().n;
    println!(
        "{}x{}x{}: {:.2} ms/step (median of 5, range {:.2}–{:.2}), potential CG iterations max {}, mean {:.2}",
        n[0],
        n[1],
        n[2],
        times[2],
        times[0],
        times[4],
        iters.iter().max().unwrap(),
        iters.iter().sum::<usize>() as f64 / iters.len() as f64
    );
}
