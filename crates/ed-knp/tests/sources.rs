mod common;

use common::*;
use ed_core::constants::F;
use ed_core::linalg::Boundary;
use ed_core::Simulation;
use ed_knp::{KnpConfig, KnpSolver};

/// 1 nA for 1 ms carries 1e-12 C, i.e. 1e-12/F mol of a monovalent ion; 1 mM·µm³ is
/// 1e-18 mol. That is 10.364 mM·µm³ (to within the rounding of F).
fn na_ms_in_mm_um3() -> f64 {
    let v = 1e6 / F;
    assert!((v - 10.364).abs() < 1e-2, "{v}");
    v
}

#[test]
fn ion_current_converts_nanoamperes() {
    let cfg = KnpConfig {
        dt: 0.5,
        ..small(Boundary::Neumann)
    };
    let mut s = KnpSolver::new(&cfg);
    let i_na = 20.0;
    // K⁺ out, Na⁺ in, Ca²⁺ out (half the ions per charge) and X⁻ in (an outward current).
    let currents = [(K, i_na), (NA, -i_na), (3, i_na), (2, i_na)];
    for (k, i) in currents {
        assert!(s.add_ion_current(CENTRE, k, i));
    }
    assert!(!s.add_ion_current([0.0, 0.0, -1.0], K, i_na));
    let m0: Vec<f64> = (0..4).map(|k| amount(&s, k)).collect();
    s.step(40);
    // The ECS holds a fraction α of each cell's volume.
    let h = s.grid().h;
    let z = [1.0, 1.0, -1.0, 2.0];
    for (k, i) in currents {
        let added = (amount(&s, k) - m0[k]) * h[0] * h[1] * h[2] * cfg.porosity;
        let expected = na_ms_in_mm_um3() * i / z[k] * 20.0;
        assert!(
            (added - expected).abs() < 1e-7 * expected.abs(),
            "species {k}: {added} vs {expected}"
        );
    }
}

#[test]
fn capacitive_current_changes_charge_density() {
    // ∂ρ/∂t = −q̃_cap: 1 nA of capacitive current for t ms removes 10.3648 t / (α V) mM.
    let cfg = KnpConfig {
        charge_tol: 1e-12,
        ..small(Boundary::Neumann)
    };
    let mut s = KnpSolver::new(&cfg);
    assert!(s.add_capacitive_current(CENTRE, 1.0));
    assert!(s.add_capacitive_current([20.0, 20.0, 20.0], -1.0));
    s.step(10);
    let h = s.grid().h;
    let expected = -na_ms_in_mm_um3() * 10.0 * cfg.dt / (cfg.porosity * h[0] * h[1] * h[2]);
    let got = charge(&s)[s.cell_at(CENTRE).unwrap()];
    assert!(
        (got - expected).abs() < 1e-9 * expected.abs(),
        "{got} vs {expected}"
    );

    s.clear_sources();
    let q = charge(&s);
    s.step(10);
    let dq = charge(&s)
        .iter()
        .zip(&q)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(dq < 1e-10, "charge moved by {dq:e} mM without sources");
}
