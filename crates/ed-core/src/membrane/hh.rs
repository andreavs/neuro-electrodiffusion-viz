//! Hodgkin-Huxley gating kinetics (Hodgkin & Huxley 1952), as used in
//! Publication 2 (KNP-EMI, eqs 15-21).
//!
//! Rates are in 1/ms and take the membrane potential `phi_m` in mV. The classic
//! formulation is written in terms of the depolarisation `v = phi_m - v_rest`;
//! with `v_rest = -65 mV` the steady states at the thesis initial potential
//! φ_M⁰ = −67.74 mV are m₀ ≈ 0.0379, h₀ ≈ 0.688, n₀ ≈ 0.276.

/// `x / (exp(x / y) - 1)`, continued smoothly through its removable
/// singularity at `x = 0`.
#[inline]
fn vtrap(x: f64, y: f64) -> f64 {
    let r = x / y;
    if r.abs() < 1e-6 {
        y * (1.0 - 0.5 * r)
    } else {
        x / r.exp_m1()
    }
}

/// Opening and closing rates (α, β) of one gate, in 1/ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    pub alpha: f64,
    pub beta: f64,
}

impl Rates {
    /// Steady-state open fraction α/(α+β).
    #[inline]
    pub fn steady_state(self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Rush-Larsen (exponential) update of gate `x` over `dt` ms. Exact for
    /// frozen rates, so the gate stays in [0, 1] for any step size.
    #[inline]
    pub fn rush_larsen(self, x: f64, dt: f64) -> f64 {
        let sum = self.alpha + self.beta;
        let x_inf = self.alpha / sum;
        x_inf + (x - x_inf) * (-dt * sum).exp()
    }
}

/// Sodium activation gate m.
#[inline]
pub fn rates_m(phi_m: f64, v_rest: f64) -> Rates {
    let v = phi_m - v_rest;
    Rates {
        alpha: 0.1 * vtrap(25.0 - v, 10.0),
        beta: 4.0 * (-v / 18.0).exp(),
    }
}

/// Sodium inactivation gate h.
#[inline]
pub fn rates_h(phi_m: f64, v_rest: f64) -> Rates {
    let v = phi_m - v_rest;
    Rates {
        alpha: 0.07 * (-v / 20.0).exp(),
        beta: 1.0 / (((30.0 - v) / 10.0).exp() + 1.0),
    }
}

/// Potassium activation gate n.
#[inline]
pub fn rates_n(phi_m: f64, v_rest: f64) -> Rates {
    let v = phi_m - v_rest;
    Rates {
        alpha: 0.01 * vtrap(10.0 - v, 10.0),
        beta: 0.125 * (-v / 80.0).exp(),
    }
}

/// Steady-state (m, h, n) at potential `phi_m`.
pub fn steady_state(phi_m: f64, v_rest: f64) -> (f64, f64, f64) {
    (
        rates_m(phi_m, v_rest).steady_state(),
        rates_h(phi_m, v_rest).steady_state(),
        rates_n(phi_m, v_rest).steady_state(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thesis_initial_gates_are_steady_state_at_initial_potential() {
        let (m, h, n) = steady_state(-67.74, -65.0);
        assert!((m - 0.0379).abs() < 5e-4, "m = {m}");
        assert!((h - 0.688).abs() < 1e-3, "h = {h}");
        assert!((n - 0.276).abs() < 1e-3, "n = {n}");
    }

    #[test]
    fn rates_are_finite_at_removable_singularities() {
        // α_m is 0/0 at v = 25 mV and α_n at v = 10 mV.
        for (phi, f) in [
            (-40.0, rates_m as fn(f64, f64) -> Rates),
            (-55.0, rates_n as fn(f64, f64) -> Rates),
        ] {
            let at = f(phi, -65.0).alpha;
            let near = f(phi + 1e-4, -65.0).alpha;
            assert!(at.is_finite());
            assert!((at - near).abs() < 1e-4, "{at} vs {near}");
        }
        assert!((rates_m(-40.0, -65.0).alpha - 1.0).abs() < 1e-12);
        assert!((rates_n(-55.0, -65.0).alpha - 0.1).abs() < 1e-12);
    }

    #[test]
    fn rush_larsen_stays_in_unit_interval_for_huge_steps() {
        for phi in [-100.0, -65.0, 0.0, 50.0] {
            for x in [0.0, 0.5, 1.0] {
                let y = rates_m(phi, -65.0).rush_larsen(x, 1e3);
                assert!((0.0..=1.0).contains(&y));
                assert!((y - rates_m(phi, -65.0).steady_state()).abs() < 1e-12);
            }
        }
    }
}
