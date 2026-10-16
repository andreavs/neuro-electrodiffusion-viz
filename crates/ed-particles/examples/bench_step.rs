//! Times one step for 1e5 Ca²⁺ ions in the default spines.
//!
//! cargo run --release -p ed-particles --example bench_step
//!
//! The cloud is first spread from the head centre so particles reach the walls; each
//! configuration is timed in several rounds and the median and best round are reported.

use std::time::Instant;

use ed_particles::{NeckField, ParticleParams, ParticleSim, SpineGeometry};

fn main() {
    let n = 100_000;
    let rounds = 9;
    let steps_per_round = 100;
    for (name, geom) in [
        ("sharp", SpineGeometry::sharp()),
        ("funneled", SpineGeometry::funneled()),
    ] {
        for (label, dv) in [("no field", 0.0), ("10 mV neck drop", 10.0)] {
            let mut sim = ParticleSim::new(geom, ParticleParams::default(), 1);
            if dv != 0.0 {
                sim.set_field(NeckField::for_spine(&geom, dv));
            }
            sim.release(n);
            for _ in 0..2000 {
                sim.step_once();
            }
            let mut times: Vec<f64> = (0..rounds)
                .map(|_| {
                    let t0 = Instant::now();
                    for _ in 0..steps_per_round {
                        sim.step_once();
                    }
                    t0.elapsed().as_secs_f64() * 1e3 / steps_per_round as f64
                })
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "{name:>8}, {label:<16}: median {:.3} ms/step, best {:.3} ms/step ({} particles)",
                times[rounds / 2],
                times[0],
                sim.n_alive()
            );
        }
    }
}
