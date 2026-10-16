//! Times `KnpSolver::step` on the default 64×64×32 grid, driven like the neuron coupling:
//! every step the sources are cleared and 200 compartments re-add spiking currents.
//!
//! Run with `cargo run --release -p ed-knp --example bench`.

use ed_knp::{KnpParams, KnpSolver};
use std::time::Instant;

fn drive(s: &mut KnpSolver, step: usize) {
    s.clear_sources();
    for c in 0..200 {
        let pos = [300.0 + 2.0 * c as f64, 500.0, 250.0];
        // Each compartment fires a 5 nA Na⁺ influx with K⁺ efflux every 250 steps.
        let i = if (step / 10 + c) % 25 == 0 { 5.0 } else { 0.0 };
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        s.add_ion_current(pos, 0, -i);
        s.add_ion_current(pos, 1, 0.5 * i);
        s.add_capacitive_current(pos, 0.5 * i * sign);
    }
}

fn main() {
    let mut s = KnpSolver::new(&KnpParams::default());
    for n in 0..10 {
        drive(&mut s, n);
        s.step();
    }
    let (steps, mut cg, mut cg_max) = (300, 0, 0);
    let start = Instant::now();
    for n in 10..10 + steps {
        drive(&mut s, n);
        s.step();
        cg += s.stats().cg_iterations;
        cg_max = cg_max.max(s.stats().cg_iterations);
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / steps as f64;
    let cg = cg as f64 / steps as f64;
    println!(
        "{} cells: {ms:.2} ms/step, CG iterations {cg:.2} mean / {cg_max} max",
        s.grid().len()
    );
}
