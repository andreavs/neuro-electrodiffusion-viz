//! Helpers shared by the integration tests.
#![allow(dead_code)]

use ed_core::linalg::Boundary;
use ed_knp::{KnpConfig, KnpSolver};

pub const NA: usize = 0;
pub const K: usize = 1;
pub const CENTRE: [f64; 3] = [125.0, 125.0, 62.5];

/// The default species on a 16×16×8 grid of 15.6 µm cells.
pub fn small(boundary: Boundary) -> KnpConfig {
    KnpConfig {
        n: [16, 16, 8],
        size: [250.0, 250.0, 125.0],
        boundary,
        ..KnpConfig::default()
    }
}

pub fn amount(s: &KnpSolver, k: usize) -> f64 {
    s.concentration(k).iter().sum()
}

/// A KX bump next to a NaX dip (electroneutral).
pub fn add_bumps(s: &mut KnpSolver) {
    let g = *s.grid();
    g.for_each_cell(|i, j, k, cell| {
        let p = g.cell_center(i, j, k);
        let r2 = |x: f64, y: f64| (p[0] - x).powi(2) + (p[1] - y).powi(2) + (p[2] - 60.0).powi(2);
        let kx = 5.0 * (-r2(100.0, 120.0) / 900.0).exp();
        let nax = -4.0 * (-r2(150.0, 130.0) / 1600.0).exp();
        s.concentration_mut(K)[cell] += kx;
        s.concentration_mut(NA)[cell] += nax;
        s.concentration_mut(2)[cell] += kx + nax;
    });
}

/// Unit Gaussian centred at x = 100 µm after spreading for `t` ms with diffusivity `d`.
pub fn gauss(x: f64, d: f64, t: f64) -> f64 {
    (-(x - 100.0).powi(2) / (4.0 * d * t)).exp() / (4.0 * std::f64::consts::PI * d * t).sqrt()
}
