//! Parameters of a [`Population`](super::Population) and the calibration that makes the
//! baseline a steady state.

use ed_core::constants::{psi_mv, T_DEFAULT};
use ed_core::membrane::hh;

/// Half-saturation of the pump by extracellular K⁺, mM (Kager et al. 2000).
pub const PUMP_KM_K: f64 = 2.0;
/// Half-saturation of the pump by intracellular Na⁺, mM (Kager et al. 2000).
pub const PUMP_KM_NA: f64 = 7.7;

/// Parameters shared by all neurons of a [`Population`].
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationParams {
    /// Membrane areas of the soma and the lumped dendrite, µm².
    pub area: [f64; 2],
    /// Intracellular volumes of the soma and dendrite, µm³.
    pub volume: [f64; 2],
    /// Axial conductance between soma and dendrite, µS.
    pub g_axial: f64,
    /// Membrane capacitance, µF/cm².
    pub c_m: f64,
    /// Somatic HH conductances ḡ_Na and ḡ_K, mS/cm².
    pub g_na_bar: f64,
    pub g_k_bar: f64,
    /// Persistent Na⁺ conductance (soma and dendrite), mS/cm².
    pub g_nap: f64,
    /// K⁺ leak of the soma and dendrite, mS/cm². Together with the pump it sets the resting
    /// ion turnover.
    pub g_leak_k: [f64; 2],
    /// Cl⁻ leak, mS/cm².
    pub g_leak_cl: f64,
    /// Multiplies the calibrated pump capacity (1 = baseline is a steady state).
    pub pump_scale: f64,
    /// Resting potential the leaks are calibrated for, mV. The HH rate functions are shifted
    /// to it.
    pub v_rest: f64,
    /// Baseline extracellular Na⁺, K⁺, X⁻ (Cl⁻), mM.
    pub ecs: [f64; 3],
    /// Baseline intracellular Na⁺ and K⁺, mM.
    pub ics: [f64; 2],
    /// Internal step bounds, ms, and the target change of V per internal step, mV.
    pub h_min: f64,
    pub h_max: f64,
    pub v_tol: f64,
    pub temperature: f64,
}

impl Default for PopulationParams {
    /// A small cortical neuron: a 20 µm soma and a lumped dendrite with four times its
    /// membrane, Publication 1 ECS baseline and Publication 2 intracellular values.
    fn default() -> Self {
        Self {
            area: [1500.0, 6000.0],
            volume: [4000.0, 3000.0],
            g_axial: 0.05,
            c_m: 1.0,
            g_na_bar: 120.0,
            g_k_bar: 36.0,
            g_nap: 0.0,
            g_leak_k: [0.1, 0.05],
            g_leak_cl: 0.02,
            pump_scale: 1.0,
            v_rest: -65.0,
            ecs: [150.0, 3.0, 155.8],
            ics: [12.0, 125.0],
            h_min: 0.025,
            h_max: 2.0,
            v_tol: 2.0,
            temperature: T_DEFAULT,
        }
    }
}

/// Persistent Na⁺ activation.
#[inline]
pub(super) fn p_inf(v: f64) -> f64 {
    1.0 / (1.0 + (-(v + 50.0) / 5.0).exp())
}

/// Pump rate relative to `P_max`.
#[inline]
pub(super) fn pump_fraction(k_e: f64, na_i: f64) -> f64 {
    let fk = k_e / (k_e + PUMP_KM_K);
    let fn_ = na_i / (na_i + PUMP_KM_NA);
    fk * fk * fn_ * fn_ * fn_
}

/// Leak and pump values that make the baseline a steady state, per compartment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// Na⁺ leak, mS/cm².
    pub g_leak_na: f64,
    /// Pump capacity P_max (before `pump_scale`), µA/cm².
    pub p_max: f64,
    /// Resting intracellular Cl⁻, mM.
    pub cl_i: f64,
}

impl PopulationParams {
    /// Calibrates compartment `c` (0 soma, 1 dendrite). Panics if the requested K⁺ leak is too
    /// small to balance the Na⁺ already entering through the voltage-gated channels.
    pub fn calibrate(&self, c: usize) -> Calibration {
        let psi = psi_mv(self.temperature);
        let v = self.v_rest;
        let [na_e, k_e, x_e] = self.ecs;
        let [na_i, k_i] = self.ics;
        let e_na = psi * (na_e / na_i).ln();
        let e_k = psi * (k_e / k_i).ln();
        let (mut g_na, mut g_k) = (self.g_nap * p_inf(v), self.g_leak_k[c]);
        if c == 0 {
            let (m, h, n) = hh::steady_state(v, v);
            g_na += self.g_na_bar * m * m * m * h;
            g_k += self.g_k_bar * n.powi(4);
        }
        // K⁺ out through the channels balances 2 K⁺ in per pump cycle; Na⁺ in, 3 out.
        let pump = g_k * (v - e_k) / 2.0;
        let g_leak_na = -3.0 * pump / (v - e_na) - g_na;
        assert!(
            pump > 0.0 && g_leak_na >= 0.0,
            "compartment {c}: K⁺ leak too small to balance the resting Na⁺ influx"
        );
        Calibration {
            g_leak_na,
            p_max: pump / pump_fraction(k_e, na_i),
            cl_i: x_e * (v / psi).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_balances_each_ion() {
        let p = PopulationParams {
            g_nap: 0.1,
            ..PopulationParams::default()
        };
        let psi = psi_mv(p.temperature);
        let v = p.v_rest;
        let e_na = psi * (p.ecs[0] / p.ics[0]).ln();
        let e_k = psi * (p.ecs[1] / p.ics[1]).ln();
        for c in 0..2 {
            let cal = p.calibrate(c);
            let pump = cal.p_max * pump_fraction(p.ecs[1], p.ics[0]);
            let (mut g_na, mut g_k) = (cal.g_leak_na + p.g_nap * p_inf(v), p.g_leak_k[c]);
            if c == 0 {
                let (m, h, n) = hh::steady_state(v, v);
                g_na += p.g_na_bar * m.powi(3) * h;
                g_k += p.g_k_bar * n.powi(4);
            }
            assert!(cal.g_leak_na > 0.0);
            assert!((g_na * (v - e_na) + 3.0 * pump).abs() < 1e-12);
            assert!((g_k * (v - e_k) - 2.0 * pump).abs() < 1e-12);
            let e_cl = -psi * (p.ecs[2] / cal.cl_i).ln();
            assert!((e_cl - v).abs() < 1e-9);
        }
    }

    #[test]
    #[should_panic(expected = "K⁺ leak too small")]
    fn too_much_persistent_sodium_is_rejected() {
        PopulationParams {
            g_nap: 2.0,
            ..PopulationParams::default()
        }
        .calibrate(1);
    }
}
