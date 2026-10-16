//! Conservation and physics checks for the KNP tissue solver.

use ed_core::constants::{psi_mv, F, T_DEFAULT};
use ed_core::ions::{IonSpecies, K, NA};
use ed_knp::{Ion, KnpParams, KnpSolver, OuterBoundary, CHARGE_TOL, X};
use std::f64::consts::PI;

fn ion(species: IonSpecies, baseline: f64) -> Ion {
    Ion { species, baseline }
}

/// Σ_k z_k c_k in every cell.
fn charge(s: &KnpSolver) -> Vec<f64> {
    (0..s.grid().len())
        .map(|cell| {
            s.ions()
                .iter()
                .enumerate()
                .map(|(k, i)| i.species.z as f64 * s.concentration(k)[cell])
                .sum()
        })
        .collect()
}

/// A small grid with electroneutral perturbations of Na⁺/X⁻ and K⁺/X⁻ pairs.
fn perturbed(boundary: OuterBoundary) -> KnpSolver {
    let params = KnpParams {
        cells: [16, 12, 8],
        extent: [160.0, 120.0, 80.0],
        boundary,
        dt: 1.0,
        ..KnpParams::default()
    };
    let mut s = KnpSolver::new(&params);
    let g = *s.grid();
    let mut dk = vec![0.0; g.len()];
    let mut dna = vec![0.0; g.len()];
    g.for_each_cell(|i, j, k, cell| {
        let p = g.cell_center(i, j, k);
        let r2 = (p[0] - 60.0).powi(2) + (p[1] - 70.0).powi(2) + (p[2] - 40.0).powi(2);
        dk[cell] = 10.0 * (-r2 / 400.0).exp();
        dna[cell] = 5.0 * (0.05 * p[0]).sin() * (0.07 * p[1]).cos();
    });
    for cell in 0..g.len() {
        s.concentration_mut(0)[cell] += dna[cell];
        s.concentration_mut(1)[cell] += dk[cell];
        s.concentration_mut(2)[cell] += dna[cell] + dk[cell];
    }
    s
}

#[test]
fn no_flux_conserves_every_species_and_charge() {
    let mut s = perturbed(OuterBoundary::NoFlux);
    let amounts: Vec<f64> = (0..4).map(|k| s.total_amount(k)).collect();
    let q0 = charge(&s);
    for _ in 0..200 {
        s.step();
        assert!(s.stats().cg_residual < 1e-3, "{:?}", s.stats());
    }
    assert!(
        s.potential().iter().any(|p| p.abs() > 1e-3),
        "diffusion potential expected"
    );
    for (k, a0) in amounts.iter().enumerate() {
        let rel = (s.total_amount(k) - a0).abs() / a0;
        assert!(rel < 1e-10, "species {k}: relative change {rel}");
    }
    for (q, q0) in charge(&s).iter().zip(&q0) {
        assert!(
            (q - q0).abs() < CHARGE_TOL,
            "charge drifted from {q0} to {q}"
        );
    }
}

#[test]
fn dirichlet_boundary_preserves_electroneutrality() {
    let mut s = perturbed(OuterBoundary::Dirichlet);
    let q0 = charge(&s);
    for _ in 0..200 {
        s.step();
    }
    for (q, q0) in charge(&s).iter().zip(&q0) {
        assert!(
            (q - q0).abs() < CHARGE_TOL,
            "charge drifted from {q0} to {q}"
        );
    }
    // Perturbations leak out through the boundary towards baseline.
    let k_excess: f64 = s.concentration(1).iter().map(|c| c - 3.0).sum();
    assert!(k_excess > 0.0 && k_excess < 10.0 * 400.0);
}

#[test]
fn neutral_tracer_spreads_like_a_gaussian() {
    let tracer = IonSpecies {
        name: "tracer",
        z: 0,
        d: 1.0,
    };
    let params = KnpParams {
        cells: [32, 32, 32],
        extent: [320.0, 320.0, 320.0],
        ions: vec![ion(tracer, 0.0), ion(NA, 100.0), ion(X, 100.0)],
        boundary: OuterBoundary::NoFlux,
        dt: 5.0,
        ..KnpParams::default()
    };
    let mut s = KnpSolver::new(&params);
    let g = *s.grid();
    let (centre, s0) = (160.0, 20.0);
    g.for_each_cell(|i, j, k, cell| {
        let p = g.cell_center(i, j, k);
        let r2: f64 = p.iter().map(|x| (x - centre) * (x - centre)).sum();
        s.concentration_mut(0)[cell] = (-r2 / (2.0 * s0 * s0)).exp();
    });
    let variance = |s: &KnpSolver| {
        let c = s.concentration(0);
        let (mut m, mut v) = (0.0, 0.0);
        g.for_each_cell(|i, _, _, cell| {
            let x = g.cell_center(i, 0, 0)[0] - centre;
            m += c[cell];
            v += c[cell] * x * x;
        });
        v / m
    };
    let v0 = variance(&s);
    // Effective diffusion coefficient in the porous medium, D/λ².
    let d = tracer.d / (params.tortuosity * params.tortuosity);
    let t = 500.0;
    for _ in 0..(t / params.dt) as usize {
        s.step();
    }
    // The discrete scheme grows the second moment by exactly 2 D t per axis.
    let v = variance(&s);
    assert!(
        (v - v0 - 2.0 * d * t).abs() < 1e-3 * 2.0 * d * t,
        "variance {v0} -> {v}"
    );
    // The peak follows the continuous solution (s0/s)³.
    let peak = s.concentration(0)[g.index(16, 16, 16)];
    let s1 = (s0 * s0 + 2.0 * d * t).sqrt();
    let exact = (s0 / s1).powi(3) * (-3.0 * 25.0 / (2.0 * s1 * s1)).exp();
    assert!(
        (peak / exact - 1.0).abs() < 0.02,
        "peak {peak}, exact {exact}"
    );
    assert!(s.potential().iter().all(|p| p.abs() < 1e-12));
}

#[test]
fn liquid_junction_potential_has_planck_value() {
    // Publication 1, application 1: NaX at 140 mM (left) and 150 mM (right).
    let params = KnpParams {
        cells: [200, 1, 1],
        extent: [1000.0, 10.0, 10.0],
        tortuosity: 1.0,
        ions: vec![ion(NA, 0.0), ion(X, 0.0)],
        boundary: OuterBoundary::NoFlux,
        dt: 10.0,
        ..KnpParams::default()
    };
    let mut s = KnpSolver::new(&params);
    for k in 0..2 {
        for (i, c) in s.concentration_mut(k).iter_mut().enumerate() {
            *c = if i < 100 { 140.0 } else { 150.0 };
        }
    }
    for _ in 0..100 {
        s.step();
    }
    let phi = s.potential();
    let dphi = phi[199] - phi[0];
    // Planck/Henderson value for a binary 1:1 electrolyte.
    let (dp, dm) = (NA.d, X.d);
    let expected = -psi_mv(T_DEFAULT) * (dp - dm) / (dp + dm) * (150.0f64 / 140.0).ln();
    assert!(expected > 0.3 && expected < 0.45);
    assert!(
        (dphi / expected - 1.0).abs() < 0.01,
        "Δφ = {dphi}, expected {expected}"
    );
}

#[test]
fn potassium_source_with_sodium_sink_gives_negative_potential() {
    let params = KnpParams {
        cells: [32, 32, 16],
        extent: [500.0, 500.0, 250.0],
        dt: 1.0,
        ..KnpParams::default()
    };
    let mut s = KnpSolver::new(&params);
    let centre = [250.0, 250.0, 125.0];
    // 1 nA of K⁺ leaving a neuron while 1 nA of Na⁺ enters it.
    assert!(s.add_ion_current(centre, 1, 1.0));
    assert!(s.add_ion_current(centre, 0, -1.0));
    assert!(!s.add_ion_current([600.0, 0.0, 0.0], 1, 1.0));
    for _ in 0..200 {
        s.step();
    }
    let cell = s.cell_at(centre).unwrap();
    assert!(
        s.concentration(1)[cell] > 3.1,
        "K = {}",
        s.concentration(1)[cell]
    );
    assert!(s.concentration(0)[cell] < 149.9);
    // K⁺ diffuses away faster than Na⁺ diffuses in, so the source region turns negative.
    let phi = s.potential()[cell];
    assert!(phi < -1e-3, "φ = {phi}");
    assert!(s.potential().iter().all(|&p| p >= phi));
}

#[test]
fn sources_enter_with_volume_fraction_and_current_units() {
    let tracer = IonSpecies {
        name: "tracer",
        z: 0,
        d: 1.0,
    };
    let params = KnpParams {
        ions: vec![ion(NA, 150.0), ion(K, 3.0), ion(X, 153.0), ion(tracer, 0.0)],
        boundary: OuterBoundary::NoFlux,
        ..KnpParams::default()
    };
    let mut s = KnpSolver::new(&params);
    let g = *s.grid();
    let centre = g.cell_center(32, 32, 16);
    let cell = s.cell_at(centre).unwrap();
    let k0 = s.total_amount(1);
    // 2 nA of K⁺ out of a neuron and 2 nA of Na⁺ into it; 5 amol/ms of tracer.
    assert!(s.add_ion_current(centre, 1, 2.0));
    assert!(s.add_ion_current(centre, 0, -2.0));
    assert!(!s.add_ion_current(centre, 3, 1.0));
    assert!(s.add_ion_source(centre, 3, 5.0));
    s.step();
    // The tracer has no drift and starts uniform, so only the source acts: 5 amol/ms
    // spread over the ECS part α h³ of the cell.
    let ecs_volume = params.volume_fraction * g.h.iter().product::<f64>();
    let expected = 5.0 * params.dt / ecs_volume;
    assert!((s.concentration(3)[cell] / expected - 1.0).abs() < 1e-12);
    // 2 nA of K⁺ is 2·10⁶/F amol/ms, and none leaves the box.
    let added = s.total_amount(1) - k0;
    let exact = 2.0 * 1e6 / F * params.dt;
    // Tolerance: round-off of summing ~10⁹ amol over the box.
    assert!((added / exact - 1.0).abs() < 1e-7, "{added} vs {exact}");
}

#[test]
fn capacitive_current_charges_the_ecs_and_gives_a_volume_conductor_field() {
    let params = KnpParams::default();
    let mut s = KnpSolver::new(&params);
    let g = *s.grid();
    let centre = g.cell_center(32, 32, 16);
    let cell = s.cell_at(centre).unwrap();
    let current = 1.0;
    assert!(s.add_capacitive_current(centre, current));
    let steps = 3;
    for _ in 0..steps {
        s.step();
    }
    // Outward capacitive current charges the membrane: ∂ρ/∂t = −ĩ_cap in the ECS.
    let ecs_volume = params.volume_fraction * g.h.iter().product::<f64>();
    let expected = -(steps as f64) * params.dt * current * 1e6 / F / ecs_volume;
    let q = charge(&s)[cell];
    assert!(
        (q / expected - 1.0).abs() < 1e-5,
        "Σ z c = {q}, expected {expected}"
    );
    // φ follows the point-source solution I / (4π α σ r) of the porous medium. Compare
    // the drop from three to six cells away, which cancels the boundary offset.
    let lambda2 = params.tortuosity * params.tortuosity;
    let sigma = (s.ions().iter())
        .map(|i| i.species.d / lambda2 * (i.species.z * i.species.z) as f64 * i.baseline)
        .sum::<f64>()
        / psi_mv(T_DEFAULT);
    let phi = s.potential();
    assert!(phi[cell] > 0.0);
    let h = g.h[0];
    let drop = phi[g.index(35, 32, 16)] - phi[g.index(38, 32, 16)];
    let exact = current * 1e6 / F / (4.0 * PI * params.volume_fraction * sigma)
        * (1.0 / (3.0 * h) - 1.0 / (6.0 * h));
    // The 7-point lattice Green's function exceeds 1/(4πr) by about h²/(4r²) along an
    // axis (≈ 5 % in this drop), and the grounded box adds about 1 %.
    assert!(
        (drop / exact - 1.0).abs() < 0.1,
        "Δφ = {drop}, expected {exact}"
    );
}
