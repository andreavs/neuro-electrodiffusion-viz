//! Times the default KNP-EMI geometry: `cargo run --release -p ed-emi --example emi_bench`.

use ed_core::Simulation;
use ed_emi::{AxonBundle, EmiParams, EmiSolver};
use std::time::Instant;

fn main() {
    let bundle = AxonBundle::default();
    let (geo, axons) = bundle.build();
    let n = geo.grid.n;
    let mut sim = EmiSolver::new(geo, EmiParams::default());
    println!("grid {n:?}, {} membrane patches", sim.faces().len());
    let x0 = axons[0].lo[0] as f64 * bundle.hx;
    let p = sim.patches_near_start(1, x0 + 10.0);
    sim.add_synapse(1.0, 4.0, &p);
    let steps = 200;
    let t = Instant::now();
    let mut pot_it = 0;
    let mut max_pot = 0;
    for _ in 0..steps {
        sim.step(1);
        let s = sim.last_stats();
        pot_it += s.potential.iterations;
        max_pot = max_pot.max(s.potential.iterations);
    }
    let el = t.elapsed().as_secs_f64();
    ode_only(&sim);
    println!(
        "{steps} steps in {el:.3} s: {:.2} ms/step, {:.1} simulated ms per wall second; potential CG avg {:.1} max {max_pot}",
        1e3 * el / steps as f64,
        sim.time_ms() / el,
        pot_it as f64 / steps as f64
    );
}

#[allow(dead_code)]
fn ode_only(sim: &EmiSolver) {
    let mut p = sim.patches.clone();
    p.use_gate_table(true);
    let t = Instant::now();
    for s in 0..100 {
        p.advance(s as f64 * 0.1, 0.1, 25, None);
    }
    println!("ODE alone: {:.2} ms/step", t.elapsed().as_secs_f64() * 10.0);
}
