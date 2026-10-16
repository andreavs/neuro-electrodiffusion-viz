//! Times the default-grid step: `cargo run --release -p ed-knp --example step_time`.

use ed_core::Simulation;
use ed_knp::{KnpSim, Params};
use std::time::Instant;

fn main() {
    let mut sim = KnpSim::new(Params::default());
    let centre = [500.0, 500.0, 250.0];
    sim.add_current_source(centre, &[-1.0, 1.0, 0.0, 0.0], 0.0);
    sim.step(20);
    let mut times: Vec<f64> = (0..200)
        .map(|_| {
            let start = Instant::now();
            sim.step(1);
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    let n = sim.grid().n;
    println!(
        "{}×{}×{} grid, {} substep(s): median {:.2} ms per step (min {:.2}, 90th percentile {:.2}), {} CG iteration(s)",
        n[0],
        n[1],
        n[2],
        sim.substeps(),
        times[100],
        times[0],
        times[180],
        sim.last_solve().iterations
    );
}
