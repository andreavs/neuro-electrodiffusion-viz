use super::*;

/// Σ_k z_k c_k in every cell.
fn charge(s: &KnpSolver) -> Vec<f64> {
    let n = s.grid().len();
    (0..n)
        .map(|cell| {
            (0..s.config().species.len())
                .map(|k| s.config().species[k].ion.z as f64 * s.concentration(k)[cell])
                .sum()
        })
        .collect()
}

/// Adds `amp · bump(x)` to species `plus` and removes it from `minus` (same valence), so
/// the perturbation is electroneutral.
fn swap(s: &mut KnpSolver, plus: usize, minus: usize, amp: f64, center: [f64; 3], width: f64) {
    let grid = *s.grid();
    let mut bump = vec![0.0; grid.len()];
    grid.for_each_cell(|i, j, k, cell| {
        let p = grid.cell_center(i, j, k);
        let r2: f64 = (0..3).map(|a| (p[a] - center[a]).powi(2)).sum();
        bump[cell] = amp * (-r2 / (width * width)).exp();
    });
    s.concentration_mut(plus)
        .iter_mut()
        .zip(&bump)
        .for_each(|(c, b)| *c += b);
    s.concentration_mut(minus)
        .iter_mut()
        .zip(&bump)
        .for_each(|(c, b)| *c -= b);
}

#[test]
fn default_baseline_is_electroneutral() {
    let cfg = KnpConfig::default();
    let q: f64 = cfg
        .species
        .iter()
        .map(|s| s.ion.z as f64 * s.baseline)
        .sum();
    assert!(q.abs() < 1e-12, "Σ z c = {q}");
    assert_eq!(KnpSolver::new(cfg).substeps(), 1);
}

#[test]
fn no_flux_conserves_amount_and_electroneutrality() {
    let cfg = KnpConfig {
        cells: [16, 16, 8],
        extent: [160.0, 160.0, 80.0],
        dt: 0.5,
        ..KnpConfig::default()
    };
    let mut s = KnpSolver::new(cfg);
    // Swap Na⁺ for K⁺ in a blob: K⁺ is faster, so a diffusion potential builds up.
    swap(&mut s, 1, 0, 20.0, [60.0, 80.0, 40.0], 25.0);
    s.solve_potential();
    // K⁺ leaves the blob faster than Na⁺ refills it, so the blob is negative:
    // Δφ ≈ −ψ (D_K − D_Na) Δc / Σ D z² c ≈ −0.6 mV relative to the far corner.
    let far = s.grid().len() - 1;
    let centre = s.grid().index(6, 8, 4);
    let dphi = s.potential()[centre] - s.potential()[far];
    assert!(dphi < -0.35 && dphi > -0.75, "Δφ = {dphi} mV");
    let totals =
        |s: &KnpSolver| -> Vec<f64> { (0..4).map(|k| s.concentration(k).iter().sum()).collect() };
    let m0 = totals(&s);
    for _ in 0..50 {
        s.step(1);
        assert!(s.last_solve().converged);
        // Checked every step: the charge correction would hide a slowly growing error.
        let q = charge(&s).iter().fold(0.0f64, |m, q| m.max(q.abs()));
        assert!(
            q < CHARGE_TOLERANCE,
            "Σ z c reached {q} mM at t = {}",
            s.time_ms()
        );
    }
    for (k, (m, m0)) in totals(&s).iter().zip(&m0).enumerate() {
        assert!(((m - m0) / m0).abs() < 1e-10, "species {k}: {m0} → {m}");
    }
}

#[test]
fn fine_grids_take_substeps() {
    let cfg = KnpConfig {
        cells: [8, 8, 8],
        extent: [8.0, 8.0, 8.0],
        dt: 1.0,
        ..KnpConfig::default()
    };
    let s = KnpSolver::new(cfg);
    // h = 1 µm, D̃_max = 2.03 / 1.6²: the forward Euler limit is about 0.21 ms.
    assert_eq!(s.substeps(), 6);
    let mut s = s;
    s.step(2);
    assert!((s.time_ms() - 2.0).abs() < 1e-12);
}

#[test]
fn single_cell_axes_do_not_limit_the_step() {
    let cfg = KnpConfig {
        cells: [32, 32, 1],
        extent: [500.0, 500.0, 0.5],
        ..KnpConfig::default()
    };
    assert_eq!(KnpSolver::new(cfg).substeps(), 1);
}

#[test]
#[should_panic(expected = "not electroneutral")]
fn charged_baseline_is_rejected() {
    let mut cfg = KnpConfig::default();
    cfg.species.pop(); // drop Ca²⁺
    KnpSolver::new(cfg);
}
