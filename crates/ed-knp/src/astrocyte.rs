//! Astrocyte membrane: the four mechanisms of the electrodiffusive astrocyte model of Halnes et
//! al. (2013), as used again in the edNEG model (Sætra et al. 2021).
//!
//! Densities in mS/cm² and µA/cm², outward positive (a conventional current, so outward Cl⁻
//! current is Cl⁻ entering the cell):
//! - K⁺ through Kir4.1: `g_Kir f (V − E_K)`, with
//!   `f = √([K]e/[K]e₀) · (1 + e^{18.4/42.4}) / (1 + e^{(V − E_K + 18.5)/42.5})
//!        · (1 + e^{−(118.6 + E_K₀)/44.1}) / (1 + e^{−(118.6 + V)/44.1})`,
//!   where `[K]e₀` and `E_K₀` are the baseline values. The conductance grows with [K⁺]e, which
//!   is what lets a depolarised astrocyte take up K⁺ where it is high.
//! - Na⁺ and Cl⁻ leaks: `g_Na (V − E_Na)`, `g_Cl (V − E_Cl)`.
//! - Na⁺/K⁺ pump: `I_p = P_max [Na]i^1.5 / ([Na]i^1.5 + K_Na^1.5) · [K]e / ([K]e + K_K)`,
//!   3 Na⁺ out and 2 K⁺ in per cycle.
//!
//! The conductances and pump constants are Halnes et al.'s. The resting potential and `P_max`
//! are not free: they are set so that the baseline concentrations are an exact ionic steady
//! state, and `[Cl⁻]i` so that `E_Cl` equals the resting potential (the neurons of
//! [`crate::population`] calibrate `P_max` and `[Cl⁻]i` the same way, but take V_rest as
//! given). With the default concentrations that gives V_rest ≈ −85 mV and
//! `P_max` ≈ 11.4 µA/cm², close to the −83.6 mV and 10.8 µA/cm² of Halnes et al.'s code.

use ed_core::constants::{psi_mv, T_DEFAULT};

/// Smallest concentration fed into a Nernst potential, mM.
const MIN_CONC: f64 = 1e-6;

/// Parameters of the astrocyte membrane.
#[derive(Debug, Clone, PartialEq)]
pub struct AstrocyteParams {
    /// Kir4.1 conductance at baseline, mS/cm².
    pub g_kir: f64,
    /// Na⁺ leak, mS/cm².
    pub g_leak_na: f64,
    /// Cl⁻ leak, mS/cm².
    pub g_leak_cl: f64,
    /// Membrane capacitance, µF/cm².
    pub c_m: f64,
    /// Pump half-saturation by intracellular Na⁺, mM.
    pub pump_km_na: f64,
    /// Pump half-saturation by extracellular K⁺, mM.
    pub pump_km_k: f64,
    /// Multiplies the calibrated pump capacity (1 = baseline is a steady state).
    pub pump_scale: f64,
    /// Baseline extracellular Na⁺, K⁺, X⁻ (Cl⁻), mM.
    pub ecs: [f64; 3],
    /// Baseline intracellular Na⁺ and K⁺, mM.
    pub ics: [f64; 2],
    /// Temperature for the Nernst potentials, K.
    pub temperature: f64,
}

impl Default for AstrocyteParams {
    /// Halnes et al. (2013) conductances and pump, the extracellular baseline of the KNP tissue
    /// and the astrocytic Na⁺ and K⁺ of the edNEG model.
    fn default() -> Self {
        Self {
            g_kir: 1.696,
            g_leak_na: 0.1,
            g_leak_cl: 0.05,
            c_m: 1.0,
            pump_km_na: 10.0,
            pump_km_k: 1.5,
            pump_scale: 1.0,
            ecs: [150.0, 3.0, 155.8],
            ics: [14.5, 101.2],
            temperature: T_DEFAULT,
        }
    }
}

/// Resting state that makes the baseline a steady state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AstrocyteCalibration {
    /// Resting membrane potential, mV.
    pub v_rest: f64,
    /// Pump capacity P_max (before `pump_scale`), µA/cm².
    pub p_max: f64,
    /// Resting intracellular Cl⁻, mM.
    pub cl_i: f64,
}

/// The astrocyte membrane with its calibration. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct AstrocyteMembrane {
    params: AstrocyteParams,
    cal: AstrocyteCalibration,
    psi: f64,
    /// Baseline E_K, mV.
    e_k0: f64,
}

impl AstrocyteParams {
    /// Finds the resting potential at which K⁺ and Na⁺ balance (Kir K⁺ out = 2 pump K⁺ in,
    /// leak Na⁺ in = 3 pump Na⁺ out), then the pump capacity and [Cl⁻]i that go with it.
    pub fn calibrate(&self) -> AstrocyteCalibration {
        assert!(
            self.g_kir > 0.0 && self.g_leak_na > 0.0,
            "the resting state needs both a K⁺ and a Na⁺ conductance"
        );
        assert!(
            self.ecs.iter().chain(&self.ics).all(|&c| c > 0.0),
            "baseline concentrations must be positive"
        );
        let psi = psi_mv(self.temperature);
        let [na_e, k_e, x_e] = self.ecs;
        let [na_i, k_i] = self.ics;
        let e_na = psi * (na_e / na_i).ln();
        let e_k = psi * (k_e / k_i).ln();
        assert!(
            e_k < e_na,
            "E_K ({e_k:.1} mV) must lie below E_Na ({e_na:.1} mV)"
        );
        // K⁺ out minus 2/3 of Na⁺ in: negative at E_K, positive at E_Na, so bisect.
        let imbalance = |v: f64| {
            self.g_kir * kir_factor(v, e_k, 1.0, e_k) * (v - e_k)
                + 2.0 / 3.0 * self.g_leak_na * (v - e_na)
        };
        let (mut lo, mut hi) = (e_k, e_na);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if imbalance(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let v = 0.5 * (lo + hi);
        let pump = -self.g_leak_na * (v - e_na) / 3.0;
        let p_max = pump / self.pump_fraction(k_e, na_i);
        assert!(p_max.is_finite() && p_max > 0.0, "pump capacity {p_max}");
        AstrocyteCalibration {
            v_rest: v,
            p_max,
            cl_i: x_e * (v / psi).exp(),
        }
    }

    /// Pump rate relative to `P_max`.
    pub fn pump_fraction(&self, k_e: f64, na_i: f64) -> f64 {
        let n = na_i.max(0.0).powf(1.5);
        let k = k_e.max(0.0);
        n / (n + self.pump_km_na.powf(1.5)) * k / (k + self.pump_km_k)
    }
}

/// The Kir4.1 factor `f` for potential `v`, current `E_K`, `[K]e/[K]e₀` and baseline `E_K₀`.
pub fn kir_factor(v: f64, e_k: f64, k_ratio: f64, e_k0: f64) -> f64 {
    let f1 = (1.0 + (18.4f64 / 42.4).exp()) / (1.0 + ((v - e_k + 18.5) / 42.5).exp());
    let f2 = (1.0 + (-(118.6 + e_k0) / 44.1).exp()) / (1.0 + (-(118.6 + v) / 44.1).exp());
    k_ratio.max(0.0).sqrt() * f1 * f2
}

impl AstrocyteMembrane {
    /// The membrane with `params`, calibrated (see [`AstrocyteParams::calibrate`]).
    pub fn new(params: AstrocyteParams) -> Self {
        let cal = params.calibrate();
        let psi = psi_mv(params.temperature);
        let e_k0 = psi * (params.ecs[1] / params.ics[1]).ln();
        Self {
            params,
            cal,
            psi,
            e_k0,
        }
    }

    pub fn params(&self) -> &AstrocyteParams {
        &self.params
    }

    /// The resting state the membrane was calibrated to.
    pub fn calibration(&self) -> AstrocyteCalibration {
        self.cal
    }

    /// Reversal potentials of Na⁺, K⁺ and Cl⁻, mV, for extracellular Na⁺, K⁺, X⁻ and
    /// intracellular Na⁺, K⁺, Cl⁻ (mM).
    pub fn reversal(&self, ecs: [f64; 3], ics: [f64; 3]) -> [f64; 3] {
        let ln = |o: f64, i: f64| (o.max(MIN_CONC) / i.max(MIN_CONC)).ln();
        [
            self.psi * ln(ecs[0], ics[0]),
            self.psi * ln(ecs[1], ics[1]),
            -self.psi * ln(ecs[2], ics[2]),
        ]
    }

    /// Chord conductances of Na⁺, K⁺ and Cl⁻ (mS/cm²) at potential `v`, with the reversal
    /// potentials `e` from [`reversal`](Self::reversal) and extracellular K⁺ `k_e` (mM).
    pub fn conductances(&self, v: f64, e: [f64; 3], k_e: f64) -> [f64; 3] {
        let p = &self.params;
        let f = kir_factor(v, e[1], k_e / p.ecs[1], self.e_k0);
        [p.g_leak_na, p.g_kir * f, p.g_leak_cl]
    }

    /// Pump current I_p, µA/cm².
    pub fn pump(&self, k_e: f64, na_i: f64) -> f64 {
        self.params.pump_scale * self.cal.p_max * self.params.pump_fraction(k_e, na_i)
    }

    /// Na⁺, K⁺ and Cl⁻ membrane currents (µA/cm², outward positive) at potential `v`.
    pub fn currents(&self, v: f64, ecs: [f64; 3], ics: [f64; 3]) -> [f64; 3] {
        let e = self.reversal(ecs, ics);
        let g = self.conductances(v, e, ecs[1]);
        let pump = self.pump(ecs[1], ics[0]);
        [
            g[0] * (v - e[0]) + 3.0 * pump,
            g[1] * (v - e[1]) - 2.0 * pump,
            g[2] * (v - e[2]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed_core::constants::F;

    /// One astrocyte with its own concentrations in a fixed extracellular bath: backward Euler
    /// on V with the chord conductances frozen over each step.
    struct Point {
        m: AstrocyteMembrane,
        v: f64,
        ics: [f64; 3],
        /// Intracellular volume per membrane area, µm.
        vol_per_area: f64,
    }

    impl Point {
        fn new(params: AstrocyteParams) -> Self {
            let m = AstrocyteMembrane::new(params);
            let c = m.calibration();
            let ics = [m.params.ics[0], m.params.ics[1], c.cl_i];
            Self {
                v: c.v_rest,
                m,
                ics,
                vol_per_area: 1.0,
            }
        }

        fn step(&mut self, dt: f64, ecs: [f64; 3]) {
            let e = self.m.reversal(ecs, self.ics);
            let g = self.m.conductances(self.v, e, ecs[1]);
            let pump = self.m.pump(ecs[1], self.ics[0]);
            let c = self.m.params.c_m;
            let gs: f64 = g.iter().sum();
            let ge: f64 = (0..3).map(|k| g[k] * e[k]).sum();
            self.v = (c / dt * self.v + ge - pump) / (c / dt + gs);
            let i = self.m.currents(self.v, ecs, self.ics);
            // 1 µA/cm² = 1e-17 C/(ms·µm²) and 1 mM = 1e-18 mol/µm³.
            let to_mm = 10.0 * dt / (F * self.vol_per_area);
            self.ics[0] -= i[0] * to_mm;
            self.ics[1] -= i[1] * to_mm;
            self.ics[2] += i[2] * to_mm;
        }
    }

    #[test]
    fn calibration_balances_each_ion() {
        let p = AstrocyteParams::default();
        let m = AstrocyteMembrane::new(p.clone());
        let c = m.calibration();
        let i = m.currents(c.v_rest, p.ecs, [p.ics[0], p.ics[1], c.cl_i]);
        for ik in i {
            assert!(ik.abs() < 1e-9, "{i:?}");
        }
        // About −85 mV, a few mV above E_K, and a pump near Halnes et al.'s 10.8 µA/cm².
        let e_k = psi_mv(p.temperature) * (p.ecs[1] / p.ics[1]).ln();
        assert!(
            c.v_rest > e_k && c.v_rest < e_k + 10.0,
            "V_rest {}",
            c.v_rest
        );
        assert!((c.v_rest + 85.0).abs() < 1.0, "V_rest {}", c.v_rest);
        assert!(c.p_max > 8.0 && c.p_max < 15.0, "P_max {}", c.p_max);
    }

    #[test]
    fn kir_grows_with_extracellular_potassium() {
        let m = AstrocyteMembrane::new(AstrocyteParams::default());
        let ics = [14.5, 101.2, m.calibration().cl_i];
        let g = |k_e: f64| {
            let e = m.reversal([150.0, k_e, 155.8], ics);
            m.conductances(-80.0, e, k_e)[1]
        };
        assert!(g(10.0) > g(6.0) && g(6.0) > g(3.0));
        // Inward rectification: less conductance far above E_K.
        let e = m.reversal([150.0, 3.0, 155.8], ics);
        assert!(m.conductances(-40.0, e, 3.0)[1] < m.conductances(-90.0, e, 3.0)[1]);
    }

    #[test]
    fn baseline_is_a_steady_state() {
        let p = AstrocyteParams::default();
        let mut a = Point::new(p.clone());
        let v0 = a.v;
        for _ in 0..1000 {
            a.step(1.0, p.ecs);
        }
        assert!((a.v - v0).abs() < 1e-9, "V {} vs {v0}", a.v);
        let cl_i = a.m.calibration().cl_i;
        for (c, c0) in a.ics.iter().zip([p.ics[0], p.ics[1], cl_i]) {
            assert!((c - c0).abs() < 1e-9, "{:?}", a.ics);
        }
    }

    #[test]
    fn raised_potassium_depolarises_and_is_taken_up() {
        let p = AstrocyteParams::default();
        let mut a = Point::new(p.clone());
        let ecs = [143.0, 10.0, 155.8];
        let (k0, cl0) = (a.ics[1], a.ics[2]);
        for _ in 0..20_000 {
            a.step(1.0, ecs);
        }
        let i = a.m.currents(a.v, ecs, a.ics);
        assert!(a.v > -75.0 && a.v < -50.0, "V {}", a.v);
        assert!(a.ics[1] > k0 + 1.0, "K⁺i {} vs {k0}", a.ics[1]);
        // The depolarisation pulls Cl⁻ in along with the K⁺.
        assert!(a.ics[2] > cl0 + 1.0, "Cl⁻i {} vs {cl0}", a.ics[2]);
        // Still taking up K⁺ (inward current) after 20 s.
        assert!(i[1] < 0.0, "I_K {}", i[1]);
    }

    #[test]
    fn pump_scale_scales_the_pump() {
        let p = AstrocyteParams {
            pump_scale: 0.5,
            ..AstrocyteParams::default()
        };
        let m = AstrocyteMembrane::new(p);
        let full = m.calibration().p_max * m.params().pump_fraction(3.0, 14.5);
        assert!((m.pump(3.0, 14.5) - 0.5 * full).abs() < 1e-12);
    }
}
