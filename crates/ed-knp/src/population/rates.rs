//! Tabulated gating functions for the [`Population`](super::Population) internal steps.

use super::params::p_inf;
use ed_core::membrane::hh;

/// Columns of a [`RateTable`] row.
pub(super) const M_INF: usize = 0;
pub(super) const M_SUM: usize = 1;
pub(super) const H_INF: usize = 2;
pub(super) const H_SUM: usize = 3;
pub(super) const N_INF: usize = 4;
pub(super) const N_SUM: usize = 5;
pub(super) const DM_INF: usize = 6;
pub(super) const P_INF: usize = 7;
pub(super) const DP_INF: usize = 8;
pub(super) const COLUMNS: usize = 9;
/// Range and spacing of the rate table, mV.
const TABLE_RANGE: (f64, f64) = (-150.0, 100.0);
const TABLE_DV: f64 = 0.05;

/// HH steady states and rate sums α + β (1/ms) of m, h and n, dm∞/dV, p∞ and dp∞/dV,
/// tabulated in V and interpolated linearly (as NEURON does), so an internal step costs
/// three exponentials instead of a dozen. Potentials outside the range are clamped.
#[derive(Debug, Clone)]
pub(super) struct RateTable {
    rows: Vec<[f64; COLUMNS]>,
}

impl RateTable {
    pub(super) fn new(v_rest: f64) -> Self {
        let exact = |v: f64| {
            let (m, h, n) = hh::rates_all(v, v_rest);
            let m_inf = |v| hh::rates_m(v, v_rest).steady_state();
            let p = p_inf(v);
            [
                m.steady_state(),
                m.alpha + m.beta,
                h.steady_state(),
                h.alpha + h.beta,
                n.steady_state(),
                n.alpha + n.beta,
                (m_inf(v + 1e-3) - m_inf(v - 1e-3)) / 2e-3,
                p,
                p * (1.0 - p) / 5.0,
            ]
        };
        let len = ((TABLE_RANGE.1 - TABLE_RANGE.0) / TABLE_DV).round() as usize + 1;
        Self {
            rows: (0..len)
                .map(|j| exact(TABLE_RANGE.0 + j as f64 * TABLE_DV))
                .collect(),
        }
    }

    #[inline]
    pub(super) fn at(&self, v: f64) -> [f64; COLUMNS] {
        let u = ((v - TABLE_RANGE.0) / TABLE_DV).clamp(0.0, (self.rows.len() - 2) as f64);
        // A NaN potential reads the first row rather than poisoning the gates.
        let j = u as usize;
        let w = (u - j as f64).max(0.0);
        let (a, b) = (&self.rows[j], &self.rows[j + 1]);
        std::array::from_fn(|c| a[c] + w * (b[c] - a[c]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_the_exact_rates() {
        let t = RateTable::new(-65.0);
        let mut v = -120.0;
        while v < 60.0 {
            let r = t.at(v);
            let (m, h, n) = hh::rates_all(v, -65.0);
            for (col, x) in [
                (M_INF, m.steady_state()),
                (M_SUM, m.alpha + m.beta),
                (H_INF, h.steady_state()),
                (H_SUM, h.alpha + h.beta),
                (N_INF, n.steady_state()),
                (N_SUM, n.alpha + n.beta),
                (P_INF, p_inf(v)),
            ] {
                assert!(
                    (r[col] - x).abs() < 1e-4 * x.abs().max(1.0),
                    "{v} mV col {col}"
                );
            }
            let dm = (hh::rates_m(v + 0.01, -65.0).steady_state()
                - hh::rates_m(v - 0.01, -65.0).steady_state())
                / 0.02;
            assert!(
                (r[DM_INF] - dm).abs() < 1e-4,
                "{v} mV: {} vs {dm}",
                r[DM_INF]
            );
            v += 0.137;
        }
        // Clamped outside the range, and NaN does not panic.
        assert_eq!(t.at(-1e3), t.at(-150.0));
        assert!(t.at(f64::NAN).iter().all(|x| x.is_finite()));
    }
}
