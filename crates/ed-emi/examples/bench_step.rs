//! Native step time for the default KNP-EMI geometry (issue #8):
//! `cargo run --release -p ed-emi --example bench_step [paper]`.
//!
//! Stimulates the centre axon of the 3×3 bundle, runs 20 ms and reports simulated ms per
//! wall second, linear-solver iterations and the conduction velocity.

use ed_emi::mesh::{Mesh, MeshSpec};
use ed_emi::solver::{EmiParams, EmiSolver};
use std::time::Instant;

fn main() {
    let paper = std::env::args().any(|a| a == "paper");
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let spec = if paper {
        let mut s = MeshSpec::paper_bundle();
        s.max_h = [5.0, 0.3, 0.3];
        s
    } else {
        MeshSpec::default_bundle()
    };
    let mesh = Mesh::build(&spec).expect("valid geometry");
    println!(
        "{} DOFs per field, {} tets, {} membrane nodes",
        mesh.n_dofs(),
        mesh.tets.len(),
        mesh.membrane.len()
    );
    let mut params = EmiParams::default();
    if let Some(r) = env("RTOL") {
        params.rtol = r;
    }
    if let Some(n) = env("SUBSTEPS") {
        params.ode_substeps = n as usize;
    }
    let mut s = EmiSolver::new(mesh, params);
    let centre = 5;
    s.stimulate(centre, 0.0, 25.0, 4.0, 0.0);

    // Membrane nodes of the centre axon at two probe positions.
    let probe = |s: &EmiSolver, x: f64| -> f64 {
        let (mut sum, mut n) = (0.0, 0);
        for (p, m) in s.mesh().membrane.iter().enumerate() {
            let px = s
                .mesh()
                .vertex_position(s.mesh().dof_vertex[m.inner as usize] as usize)[0];
            if m.cell == centre && (px - x).abs() < 1e-9 {
                sum += s.phi_m()[p];
                n += 1;
            }
        }
        sum / n as f64
    };
    let xs = [60.0, 140.0];
    let mut prev = xs.map(|x| probe(&s, x));
    let mut cross = [f64::NAN; 2];
    let (mut its_phi, mut its_c) = (0, 0);
    let start = Instant::now();
    while s.t() < 20.0 {
        s.step().expect("step failed");
        its_phi += s.stats().potential.map_or(0, |c| c.iterations);
        its_c += s
            .stats()
            .concentration
            .iter()
            .flatten()
            .map(|c| c.iterations)
            .sum::<usize>();
        for i in 0..2 {
            let v = probe(&s, xs[i]);
            if cross[i].is_nan() && v > -20.0 && prev[i] <= -20.0 {
                cross[i] = s.t() - s.params().dt * (v + 20.0) / (v - prev[i]);
            }
            prev[i] = v;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let steps = (s.t() / s.params().dt).round();
    println!(
        "{:.1} ms in {wall:.2} s: {:.1} simulated ms per wall second, {:.2} ms per step",
        s.t(),
        s.t() / wall,
        1e3 * wall / steps
    );
    println!(
        "mean CG iterations per step: potential {:.1}, concentrations {:.1}",
        its_phi as f64 / steps,
        its_c as f64 / steps
    );
    println!(
        "conduction velocity {:.0} µm/ms ({:.2} m/s)",
        (xs[1] - xs[0]) / (cross[1] - cross[0]),
        1e-3 * (xs[1] - xs[0]) / (cross[1] - cross[0])
    );
}
