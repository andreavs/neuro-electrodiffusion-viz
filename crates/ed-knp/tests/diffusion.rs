mod common;

use common::*;
use ed_core::ions::{self, IonSpecies};
use ed_core::linalg::Boundary;
use ed_core::Simulation;
use ed_knp::{KnpConfig, KnpSolver};

/// Spreads a unit Gaussian of a neutral species (D̃ = 1 µm²/ms) from t = 10 to 30 ms on a
/// 1-D grid of `n` cells of size `h` and compares it with the analytic solution.
fn gaussian_case(h: f64, dt: f64, tortuosity: f64, boundary: Boundary) {
    let neutral = IonSpecies {
        name: "N",
        z: 0,
        d: tortuosity * tortuosity,
    };
    let (t0, t1, n) = (10.0 * h * h, 30.0 * h * h, 200);
    let mut s = KnpSolver::new(&KnpConfig {
        n: [n, 1, 1],
        size: [n as f64 * h, 1.0, 1.0],
        species: vec![ions::NA, ions::CL, neutral],
        baseline: vec![150.0, 150.0, 0.0],
        tortuosity,
        dt,
        boundary,
        ..KnpConfig::default()
    });
    // Positions in units of h, so the profile is the unit-h one at times t / h².
    let x = |i: usize| i as f64 + 0.5;
    for i in 0..n {
        s.concentration_mut(2)[i] = gauss(x(i), 1.0, t0 / (h * h));
    }
    s.step(((t1 - t0) / dt).round() as usize);
    let peak = gauss(100.0, 1.0, t1 / (h * h));
    for i in 0..n {
        let err = (s.concentration(2)[i] - gauss(x(i), 1.0, t1 / (h * h))).abs();
        assert!(err < 0.01 * peak, "x = {}: error {err:e}", x(i));
    }
}

#[test]
fn neutral_perturbation_spreads_like_gaussian() {
    // 1 µm cells and 0.02 ms steps: solved with multigrid.
    gaussian_case(1.0, 0.02, 1.0, Boundary::Neumann);
}

#[test]
fn jacobi_path_and_tortuosity_match_gaussian() {
    // 10 µm cells and 1 ms steps: solved with Jacobi sweeps. D = 2.56 with λ = 1.6 gives
    // D̃ = 1, and the Dirichlet ends must not pin the thin y and z axes to the baseline.
    gaussian_case(10.0, 1.0, 1.6, Boundary::Dirichlet);
}

#[test]
fn diffusion_conserves_every_species() {
    let mut s = KnpSolver::new(&KnpConfig {
        dt: 1.0,
        ..small(Boundary::Neumann)
    });
    add_bumps(&mut s);
    let m0: Vec<f64> = (0..4).map(|k| amount(&s, k)).collect();
    let k_peak = s.concentration(K).iter().cloned().fold(0.0, f64::max);
    s.step(200);
    for (k, m0) in m0.iter().enumerate() {
        let rel = (amount(&s, k) - m0).abs() / m0;
        assert!(rel < 1e-10, "species {k}: relative drift {rel:e}");
    }
    assert!(s.concentration(K).iter().cloned().fold(0.0, f64::max) < k_peak - 1.0);
}

#[test]
fn ion_source_adds_its_amount() {
    // A rate of r mM·µm³/ms for t ms raises Σ c·V by r t / α in the ECS.
    let cfg = KnpConfig {
        dt: 0.5,
        ..small(Boundary::Neumann)
    };
    let mut s = KnpSolver::new(&cfg);
    assert!(s.add_ion_source(CENTRE, K, 1000.0));
    assert!(s.add_ion_source(CENTRE, NA, -1000.0));
    assert!(!s.add_ion_source([-1.0, 0.0, 0.0], K, 1000.0));
    let m0 = amount(&s, K);
    s.step(40);
    let h = s.grid().h;
    let added = (amount(&s, K) - m0) * h[0] * h[1] * h[2];
    let expected = 1000.0 * 20.0 / cfg.porosity;
    assert!(
        (added - expected).abs() < 1e-9 * expected,
        "{added} vs {expected}"
    );
    let cell = s.cell_at(CENTRE).unwrap();
    assert!(s.concentration(K)[cell] > 3.5);

    s.clear_sources();
    let m1 = amount(&s, K);
    s.step(10);
    assert!((amount(&s, K) - m1).abs() < 1e-10 * m1);
}
