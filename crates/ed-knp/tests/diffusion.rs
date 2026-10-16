mod common;

use common::*;
use ed_core::ions::{self, IonSpecies};
use ed_core::linalg::Boundary;
use ed_core::Simulation;
use ed_knp::{KnpConfig, KnpSolver};

#[test]
fn neutral_perturbation_spreads_like_gaussian() {
    let neutral = IonSpecies {
        name: "N",
        z: 0,
        d: 1.0,
    };
    let (d, t0, t1, n, dt) = (1.0, 10.0, 30.0, 200, 0.02);
    let mut s = KnpSolver::new(&KnpConfig {
        n: [n, 1, 1],
        size: [n as f64, 1.0, 1.0],
        species: vec![ions::NA, ions::CL, neutral],
        baseline: vec![150.0, 150.0, 0.0],
        tortuosity: 1.0,
        dt,
        boundary: Boundary::Neumann,
        ..KnpConfig::default()
    });
    let x = |i: usize| i as f64 + 0.5;
    for i in 0..n {
        s.concentration_mut(2)[i] = gauss(x(i), d, t0);
    }
    s.step(((t1 - t0) / dt).round() as usize);
    let peak = gauss(100.0, d, t1);
    for i in 0..n {
        let err = (s.concentration(2)[i] - gauss(x(i), d, t1)).abs();
        assert!(err < 0.01 * peak, "x = {}: error {err:e}", x(i));
    }
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
fn dirichlet_boundary_pulls_towards_baseline() {
    let mut s = KnpSolver::new(&small(Boundary::Dirichlet));
    add_bumps(&mut s);
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
