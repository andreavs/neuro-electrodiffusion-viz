use super::*;
use ed_core::constants::psi_mv;
use ed_core::linalg::Grid;
use ed_core::Simulation;

fn species(list: &[(IonSpecies, f64)]) -> Vec<Species> {
    list.iter()
        .map(|&(ion, baseline)| Species { ion, baseline })
        .collect()
}

/// Adds `amp · exp(−|x − x0|² / (2 s²))` to `c`.
fn add_gaussian(g: &Grid, c: &mut [f64], x0: [f64; 3], s: f64, amp: f64) {
    g.for_each_cell(|i, j, k, idx| {
        let p = g.cell_center(i, j, k);
        let r2: f64 = (0..3).map(|a| (p[a] - x0[a]).powi(2)).sum();
        c[idx] += amp * (-r2 / (2.0 * s * s)).exp();
    });
}

fn max_charge_drift(sim: &KnpSim, initial: &[f64]) -> f64 {
    initial
        .iter()
        .enumerate()
        .map(|(i, q)| (sim.charge_density(i) - q).abs())
        .fold(0.0, f64::max)
}

fn charges(sim: &KnpSim) -> Vec<f64> {
    (0..sim.grid().len())
        .map(|i| sim.charge_density(i))
        .collect()
}

#[test]
fn no_flux_conserves_amounts_and_electroneutrality() {
    let mut sim = KnpSim::new(Params {
        cells: [16, 16, 8],
        extent: [160.0, 160.0, 80.0],
        boundary: OuterBoundary::NoFlux,
        ..Params::default()
    });
    let g = *sim.grid();
    // An electroneutral K⁺-for-Na⁺ swap plus extra NaX salt in another spot.
    add_gaussian(&g, sim.concentration_mut(1), [60.0, 80.0, 40.0], 15.0, 10.0);
    add_gaussian(
        &g,
        sim.concentration_mut(0),
        [60.0, 80.0, 40.0],
        15.0,
        -10.0,
    );
    add_gaussian(&g, sim.concentration_mut(0), [110.0, 70.0, 30.0], 20.0, 5.0);
    add_gaussian(&g, sim.concentration_mut(2), [110.0, 70.0, 30.0], 20.0, 5.0);
    let amounts: Vec<f64> = (0..4).map(|k| sim.total_amount(k)).collect();
    let q0 = charges(&sim);
    let peak0 = sim.concentration(1).iter().cloned().fold(0.0, f64::max);

    sim.step(500);

    for (k, m0) in amounts.iter().enumerate() {
        let rel = (sim.total_amount(k) - m0).abs() / m0;
        assert!(rel < 1e-10, "species {k}: relative change {rel:e}");
    }
    let drift = max_charge_drift(&sim, &q0);
    assert!(drift < 1e-9, "charge density drifted by {drift:e} mM");
    assert!(sim.potential().iter().any(|&p| p.abs() > 1e-3));
    // The K⁺ bump spread out.
    let peak = sim.concentration(1).iter().cloned().fold(0.0, f64::max);
    assert!(peak < peak0 - 1.0, "K⁺ peak {peak0} -> {peak}");
}

#[test]
fn neutral_species_spreads_like_a_gaussian() {
    const NEUTRAL: IonSpecies = IonSpecies {
        name: "N",
        z: 0,
        d: 2.0,
    };
    let (s0, t, lambda) = (2.0, 5.0, 1.6);
    let d = NEUTRAL.d / (lambda * lambda);
    let mut sim = KnpSim::new(Params {
        cells: [32, 32, 32],
        extent: [32.0, 32.0, 32.0],
        tortuosity: lambda,
        boundary: OuterBoundary::NoFlux,
        species: species(&[(NEUTRAL, 0.0)]),
        ..Params::default()
    });
    let g = *sim.grid();
    let x0 = [16.0; 3];
    add_gaussian(&g, sim.concentration_mut(0), x0, s0, 1.0);
    let variance = |sim: &KnpSim| {
        let c = sim.concentration(0);
        let mut m = [0.0; 2];
        g.for_each_cell(|i, _, _, idx| {
            let x = g.cell_center(i, 0, 0)[0] - x0[0];
            m[0] += c[idx];
            m[1] += c[idx] * x * x;
        });
        m[1] / m[0]
    };
    let var0 = variance(&sim);

    sim.step((t / sim.params().dt).round() as usize);

    let growth = variance(&sim) - var0;
    assert!(
        (growth - 2.0 * d * t).abs() < 1e-3 * 2.0 * d * t,
        "variance grew by {growth}"
    );
    // Peak of the continuous solution: (s0² / s²)^{3/2}, sampled half a cell off-centre.
    let s2 = s0 * s0 + 2.0 * d * t;
    let peak = (s0 * s0 / s2).powf(1.5) * (-3.0 * 0.25 / (2.0 * s2)).exp();
    let got = sim.concentration(0)[g.index(16, 16, 16)];
    assert!((got - peak).abs() < 0.01 * peak, "peak {got} vs {peak}");
}

#[test]
fn liquid_junction_gives_diffusion_potential() {
    // Publication 1: NaX at 150 mM next to 140 mM. X⁻ diffuses faster than Na⁺, so
    // the dilute side turns negative by ψ (D_X − D_Na)/(D_X + D_Na) ln(150/140).
    let mut sim = KnpSim::new(Params {
        cells: [200, 1, 1],
        extent: [200.0, 1.0, 1.0],
        tortuosity: 1.0,
        boundary: OuterBoundary::NoFlux,
        species: species(&[(NA, 150.0), (X, 150.0)]),
        ..Params::default()
    });
    assert!(
        sim.substeps() > 1,
        "dt should exceed the stability limit here"
    );
    for k in 0..2 {
        sim.concentration_mut(k)[100..].fill(140.0);
    }
    sim.step(100);

    let phi = sim.potential();
    let jump = phi[199] - phi[0];
    let psi = psi_mv(T_DEFAULT);
    let expected = -psi * (X.d - NA.d) / (X.d + NA.d) * (150.0f64 / 140.0).ln();
    assert!(jump < 0.0, "dilute side should be negative, got {jump}");
    assert!(
        (jump - expected).abs() < 1e-3 * expected.abs(),
        "junction potential {jump} mV, expected {expected} mV"
    );
}

#[test]
fn potassium_source_with_sodium_sink() {
    let mut sim = KnpSim::new(Params {
        cells: [32, 32, 16],
        extent: [500.0, 500.0, 250.0],
        ..Params::default()
    });
    let centre = [250.0, 250.0, 125.0];
    assert!(sim.add_current_source(centre, &[-1.0, 1.0, 0.0, 0.0], 0.0));
    let q0 = charges(&sim);
    sim.step(1000);

    let cell = sim.cell_at(centre).unwrap();
    assert!(sim.concentration(1)[cell] > 3.1, "K⁺ should rise locally");
    assert!(
        sim.concentration(0)[cell] < 149.9,
        "Na⁺ should fall locally"
    );
    // K⁺ leaves the source region faster than Na⁺ refills it, so the source region is
    // left negative relative to the grounded boundary.
    let phi = sim.potential();
    assert!(phi[cell] < -1e-4, "φ at the source = {} mV", phi[cell]);
    assert_eq!(phi[cell], phi.iter().cloned().fold(0.0, f64::min));
    let drift = max_charge_drift(&sim, &q0);
    assert!(drift < 1e-9, "charge density drifted by {drift:e} mM");
}

#[test]
fn capacitive_current_changes_local_charge() {
    let mut sim = KnpSim::new(Params {
        cells: [16, 16, 8],
        extent: [250.0, 250.0, 125.0],
        boundary: OuterBoundary::NoFlux,
        ..Params::default()
    });
    let pos = [100.0, 120.0, 60.0];
    // Total membrane current is zero: K⁺ out, an equal capacitive current in.
    assert!(sim.add_current_source(pos, &[0.0, 0.5, 0.0, 0.0], -0.5));
    let q0 = charges(&sim);
    let steps = 200;
    sim.step(steps);

    let cell = sim.cell_at(pos).unwrap();
    let h = sim.grid().h;
    let per_ms =
        0.5 * 1e6 / ed_core::constants::F / (h[0] * h[1] * h[2] * sim.params().volume_fraction);
    let expected = per_ms * steps as f64 * sim.params().dt;
    let got = sim.charge_density(cell) - q0[cell];
    assert!(
        (got - expected).abs() < 1e-6 * expected,
        "{got} vs {expected}"
    );
    let others = (0..q0.len())
        .filter(|&i| i != cell)
        .map(|i| (sim.charge_density(i) - q0[i]).abs())
        .fold(0.0, f64::max);
    assert!(others < 1e-9 * expected.max(1.0), "{others:e}");
}

#[test]
fn sources_outside_the_box_are_rejected() {
    let mut sim = KnpSim::new(Params {
        cells: [8, 8, 4],
        extent: [80.0, 80.0, 40.0],
        ..Params::default()
    });
    assert_eq!(sim.cell_at([0.0, 0.0, 0.0]), Some(0));
    assert_eq!(sim.cell_at([79.9, 79.9, 39.9]), Some(255));
    assert!(sim.cell_at([80.0, 1.0, 1.0]).is_none());
    assert!(!sim.add_species_source([-1.0, 1.0, 1.0], 0, 1.0));
    assert!(!sim.add_current_source([1.0, 1.0, 40.5], &[1.0; 4], 0.0));
}
