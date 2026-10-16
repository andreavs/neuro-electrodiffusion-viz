mod common;

use common::*;
use ed_core::constants::{psi_mv, T_DEFAULT};
use ed_core::ions;
use ed_core::linalg::Boundary;
use ed_core::Simulation;
use ed_knp::{KnpConfig, KnpSolver};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// K⁺ released at the centre with a matching Na⁺ uptake, in mM·µm³/ms.
fn add_k_source_na_sink(s: &mut KnpSolver, rate: f64) {
    assert!(s.add_ion_source(CENTRE, K, rate));
    assert!(s.add_ion_source(CENTRE, NA, -rate));
}

#[test]
fn no_flux_conserves_ions_and_charge() {
    let mut s = KnpSolver::new(&KnpConfig {
        dt: 1.0,
        charge_tol: 1e-12,
        ..small(Boundary::Neumann)
    });
    add_bumps(&mut s);
    let q0 = charge(&s);
    let m0: Vec<f64> = (0..4).map(|k| amount(&s, k)).collect();
    s.step(200);
    for (k, m0) in m0.iter().enumerate() {
        let rel = (amount(&s, k) - m0).abs() / m0;
        assert!(rel < 1e-10, "species {k}: relative drift {rel:e}");
    }
    let dq = max_abs_diff(&charge(&s), &q0);
    assert!(dq < 1e-10, "charge drift {dq:e} mM");
    assert!(
        s.potential().iter().any(|p| p.abs() > 1e-3),
        "no diffusion potential"
    );
}

#[test]
fn default_tolerance_does_not_accumulate_charge_error() {
    let mut s = KnpSolver::new(&small(Boundary::Dirichlet));
    add_k_source_na_sink(&mut s, 500.0);
    s.step(400);
    let dq = charge(&s).iter().map(|q| q.abs()).fold(0.0, f64::max);
    assert!(dq < 1e-6, "charge {dq:e} mM");
}

#[test]
fn large_steps_stay_stable() {
    // 50 ms steps move ions much further than a cell per step.
    let mut s = KnpSolver::new(&KnpConfig {
        dt: 50.0,
        ..small(Boundary::Dirichlet)
    });
    add_k_source_na_sink(&mut s, 50.0);
    s.step(40);
    for k in 0..4 {
        let min = s
            .concentration(k)
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let finite = s.concentration(k).iter().all(|c| c.is_finite());
        assert!(min > 0.0 && finite, "species {k}: min {min}");
    }
    let k_max = s.concentration(K).iter().cloned().fold(0.0, f64::max);
    assert!(k_max > 3.0 && k_max < 150.0, "K max {k_max}");
}

#[test]
fn baseline_is_steady_with_zero_potential() {
    let mut s = KnpSolver::new(&small(Boundary::Dirichlet));
    s.step(10);
    for (k, base) in KnpConfig::default().baseline.iter().enumerate() {
        assert!(s.concentration(k).iter().all(|c| (c - base).abs() < 1e-10));
    }
    assert!(s.potential().iter().all(|p| p.abs() < 1e-10));
}

#[test]
fn salt_perturbation_spreads_with_ambipolar_diffusion() {
    // A small NaCl bump on a NaCl background spreads with D = 2 D₊ D₋ / (D₊ + D₋).
    let (dp, dn) = (ions::NA.d, ions::CL.d);
    let (d, t0, t1, n, dt) = (2.0 * dp * dn / (dp + dn), 10.0, 30.0, 200, 0.02);
    let mut s = KnpSolver::new(&KnpConfig {
        n: [n, 1, 1],
        size: [n as f64, 1.0, 1.0],
        species: vec![ions::NA, ions::CL],
        baseline: vec![150.0, 150.0],
        tortuosity: 1.0,
        dt,
        boundary: Boundary::Dirichlet,
        ..KnpConfig::default()
    });
    let x = |i: usize| i as f64 + 0.5;
    for k in 0..2 {
        for i in 0..n {
            s.concentration_mut(k)[i] += gauss(x(i), d, t0);
        }
    }
    s.step(((t1 - t0) / dt).round() as usize);
    let peak = gauss(100.0, d, t1);
    for i in 0..n {
        let err = (s.concentration(0)[i] - 150.0 - gauss(x(i), d, t1)).abs();
        assert!(err < 0.02 * peak, "x = {}: error {err:e}", x(i));
    }
}

#[test]
fn liquid_junction_gives_diffusion_potential() {
    // NaCl 140 mM | 150 mM: the faster anion runs ahead, leaving the dilute side negative.
    let n = 200;
    let mut s = KnpSolver::new(&KnpConfig {
        n: [n, 1, 1],
        size: [n as f64, 1.0, 1.0],
        species: vec![ions::NA, ions::CL],
        baseline: vec![140.0, 140.0],
        dt: 0.1,
        boundary: Boundary::Neumann,
        charge_tol: 1e-12,
        ..KnpConfig::default()
    });
    for k in 0..2 {
        s.concentration_mut(k)[n / 2..].fill(150.0);
    }
    s.step(200);
    let (dp, dn) = (ions::NA.d, ions::CL.d);
    let expected = psi_mv(T_DEFAULT) * (dn - dp) / (dp + dn) * (150.0f64 / 140.0).ln();
    let got = s.potential()[n - 1] - s.potential()[0];
    assert!(
        got > 0.0,
        "high-concentration side should be positive, got {got}"
    );
    assert!(
        (got - expected).abs() < 0.01 * expected,
        "Δφ = {got} mV, expected {expected} mV"
    );
}

#[test]
fn k_source_with_na_sink_lowers_potential() {
    let mut s = KnpSolver::new(&KnpConfig {
        dt: 0.5,
        ..small(Boundary::Neumann)
    });
    add_k_source_na_sink(&mut s, 200.0);
    s.step(40);
    let cell = s.cell_at(CENTRE).unwrap();
    assert!(s.concentration(K)[cell] > 3.5);
    // K⁺ diffuses away faster than Na⁺ refills, so the source region turns negative.
    let phi = s.potential();
    let phi_min = phi.iter().cloned().fold(0.0, f64::min);
    assert!(
        phi[cell] < 0.0 && phi[cell] <= phi_min + 1e-12,
        "φ = {}",
        phi[cell]
    );
}

#[test]
fn dirichlet_boundary_keeps_charge_and_pulls_towards_baseline() {
    // Coarse, so the bumps reach the boundary faces and their drift matters.
    let mut s = KnpSolver::new(&KnpConfig {
        n: [6, 5, 3],
        charge_tol: 1e-12,
        ..small(Boundary::Dirichlet)
    });
    add_bumps(&mut s);
    let q0 = charge(&s);
    s.step(50);
    let dq = max_abs_diff(&charge(&s), &q0);
    assert!(dq < 1e-10, "charge drift {dq:e} mM");
    s.set_dt(50.0);
    s.step(400);
    for (k, base) in KnpConfig::default().baseline.iter().enumerate() {
        let dev = s
            .concentration(k)
            .iter()
            .map(|c| (c - base).abs())
            .fold(0.0, f64::max);
        assert!(dev < 1e-3, "species {k}: max deviation {dev:e} mM");
    }
}
