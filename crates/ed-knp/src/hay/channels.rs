//! Channel kinetics of the Hay et al. (2011) model, transcribed from the ModelDB 139653
//! NMODL files, with the densities of `L5PCbiophys3.hoc` (the model constrained for both
//! BAC firing and current-step firing).
//!
//! Units: mV, ms, mM; conductance densities mS/cm², currents µA/cm² (outward positive),
//! capacitance µF/cm². The NMODL files hard-code a temperature factor for 34 °C
//! (`qt = 2.3^((34 − 21)/10)`) on most time constants, so the kinetics are those of the
//! paper's 34 °C simulations. Gates use `cnexp`'s exact exponential update for a fixed
//! potential (Rush-Larsen), as NEURON does.

use super::Region;
use ed_core::constants::{F, R};

/// `2.3^((34 − 21)/10)`.
const QT: f64 = 2.952_882_641_412_121;
/// Temperature of the paper's simulations, K.
const CELSIUS_34: f64 = 307.15;

/// Fixed reversal potentials of the model (mV) and the extracellular Ca²⁺ (mM) used for
/// E_Ca. Hay et al. set E_Na and E_K by hand; E_Ca follows the internal Ca²⁺.
pub const E_NA: f64 = 50.0;
pub const E_K: f64 = -85.0;
pub const E_PAS: f64 = -90.0;
pub const E_H: f64 = -45.0;
pub const CA_OUT: f64 = 2.0;

/// Fraction of a current reversing at `e` that Na⁺ carries when it is split into a Na⁺
/// and a K⁺ part reversing at E_Na and E_K: the two parts sum to the original current at
/// every potential.
pub fn na_fraction(e: f64) -> f64 {
    (e - E_K) / (E_NA - E_K)
}

/// Maximal conductances (mS/cm²) and the other per-compartment parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Densities {
    pub na_t: f64,
    pub na_p: f64,
    pub k_p: f64,
    pub k_t: f64,
    pub sk: f64,
    pub kv3: f64,
    pub ca_hva: f64,
    pub ca_lva: f64,
    pub im: f64,
    pub ih: f64,
    pub pas: f64,
    /// µF/cm² (2 in the dendrites, to account for spines).
    pub cm: f64,
    /// Ca²⁺ removal time constant, ms, and free fraction of Ca²⁺ influx.
    pub ca_decay: f64,
    pub ca_gamma: f64,
}

impl Densities {
    /// `L5PCbiophys3` at path distance `dist` (µm) from the soma; `max_apical` normalises
    /// the exponential apical Ih gradient. Values in the .hoc file are S/cm².
    pub fn hay(region: Region, dist: f64, max_apical: f64) -> Self {
        let s = |g: f64| 1e3 * g;
        let base = Self {
            cm: 1.0,
            ca_decay: 80.0,
            ca_gamma: 0.05,
            ..Self::default()
        };
        match region {
            Region::Soma => Self {
                ca_lva: s(0.00343),
                ca_hva: s(0.000992),
                kv3: s(0.693),
                sk: s(0.0441),
                k_t: s(0.0812),
                k_p: s(0.00223),
                na_p: s(0.00172),
                na_t: s(2.04),
                ih: s(0.0002),
                pas: s(0.0000338),
                ca_decay: 460.0,
                ca_gamma: 0.000501,
                ..base
            },
            Region::Apical => {
                // Ca²⁺ hot zone 685–885 µm from the soma, around the main bifurcation.
                let hot = dist > 685.0 && dist < 885.0;
                Self {
                    cm: 2.0,
                    ih: s(0.0002 * (-0.8696 + 2.087 * (3.6161 * dist / max_apical).exp())),
                    sk: s(0.0012),
                    ca_lva: s(0.0187 * if hot { 1.0 } else { 0.01 }),
                    ca_hva: s(0.000555 * if hot { 1.0 } else { 0.1 }),
                    kv3: s(0.000261),
                    na_t: s(0.0213),
                    im: s(0.0000675),
                    pas: s(0.0000589),
                    ca_decay: 122.0,
                    ca_gamma: 0.000509,
                    ..base
                }
            }
            Region::Basal => Self {
                cm: 2.0,
                ih: s(0.0002),
                pas: s(0.0000467),
                ..base
            },
            Region::Axon => Self {
                pas: s(0.0000325),
                ..base
            },
        }
    }
}

/// Gate states and internal Ca²⁺ (mM) of one compartment.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct State {
    pub nat: [f64; 2],
    pub nap: [f64; 2],
    pub kp: [f64; 2],
    pub kt: [f64; 2],
    pub sk: f64,
    pub kv3: f64,
    pub hva: [f64; 2],
    pub lva: [f64; 2],
    pub im: f64,
    pub ih: f64,
    pub cai: f64,
}

/// Membrane currents of one compartment, µA/cm², and the total conductance (mS/cm²)
/// with the gates frozen, so `I(V') ≈ total + g (V' − V)` is exact for the ionic part.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Currents {
    pub na: f64,
    pub k: f64,
    pub ca: f64,
    pub g: f64,
}

impl Currents {
    pub fn total(&self) -> f64 {
        self.na + self.k + self.ca
    }
}

/// `x / (1 − exp(−x/k))`, continuous at x = 0 (where NMODL nudges v by 1e-4 mV).
fn efun(x: f64, k: f64) -> f64 {
    if x.abs() < 1e-6 * k {
        k + 0.5 * x
    } else {
        x / (1.0 - (-x / k).exp())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// (steady state, time constant ms) of every voltage-gated gate, in [`State`] order.
type Gate = (f64, f64);

fn from_rates(a: f64, b: f64, q: f64) -> Gate {
    (a / (a + b), 1.0 / ((a + b) * q))
}

fn nat_m(v: f64) -> Gate {
    from_rates(
        0.182 * efun(v + 38.0, 6.0),
        0.124 * efun(-v - 38.0, 6.0),
        QT,
    )
}
fn nat_h(v: f64) -> Gate {
    from_rates(
        0.015 * efun(-v - 66.0, 6.0),
        0.015 * efun(v + 66.0, 6.0),
        QT,
    )
}
fn nap_m(v: f64) -> Gate {
    (sigmoid((v + 52.6) / 4.6), 6.0 * nat_m(v).1)
}
fn nap_h(v: f64) -> Gate {
    let a = 2.88e-6 * efun(-v - 17.0, 4.63);
    let b = 6.94e-6 * efun(v + 64.4, 2.63);
    (sigmoid(-(v + 48.8) / 10.0), 1.0 / ((a + b) * QT))
}
fn kp_m(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = if v < -50.0 {
        1.25 + 175.03 * (0.026 * v).exp()
    } else {
        1.25 + 13.0 * (-0.026 * v).exp()
    };
    (sigmoid((v + 1.0) / 12.0), tau / QT)
}
fn kp_h(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = 360.0 + (1010.0 + 24.0 * (v + 55.0)) * (-((v + 75.0) / 48.0).powi(2)).exp();
    (sigmoid(-(v + 54.0) / 11.0), tau / QT)
}
fn kt_m(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = 0.34 + 0.92 * (-((v + 71.0) / 59.0).powi(2)).exp();
    (sigmoid(v / 19.0), tau / QT)
}
fn kt_h(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = 8.0 + 49.0 * (-((v + 73.0) / 23.0).powi(2)).exp();
    (sigmoid(-(v + 66.0) / 10.0), tau / QT)
}
fn kv3_m(v: f64) -> Gate {
    (
        sigmoid((v - 18.7) / 9.7),
        4.0 * sigmoid((v + 46.56) / 44.14),
    )
}
fn hva_m(v: f64) -> Gate {
    from_rates(
        0.055 * efun(v + 27.0, 3.8),
        0.94 * ((-75.0 - v) / 17.0).exp(),
        1.0,
    )
}
fn hva_h(v: f64) -> Gate {
    let a = 0.000457 * ((-13.0 - v) / 50.0).exp();
    let b = 0.0065 * sigmoid((v + 15.0) / 28.0);
    from_rates(a, b, 1.0)
}
fn lva_m(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = 5.0 + 20.0 * sigmoid(-(v + 25.0) / 5.0);
    (sigmoid((v + 30.0) / 6.0), tau / QT)
}
fn lva_h(v: f64) -> Gate {
    let v = v + 10.0;
    let tau = 20.0 + 50.0 * sigmoid(-(v + 40.0) / 7.0);
    (sigmoid(-(v + 80.0) / 6.4), tau / QT)
}
fn im_m(v: f64) -> Gate {
    let x = 0.1 * (v + 35.0);
    from_rates(3.3e-3 * x.exp(), 3.3e-3 * (-x).exp(), QT)
}
fn ih_m(v: f64) -> Gate {
    from_rates(
        0.00643 * efun(-v - 154.9, 11.9),
        0.193 * (v / 33.1).exp(),
        1.0,
    )
}
/// SK_E2 activation by internal Ca²⁺ (mM); τ = 1 ms.
fn sk_z(cai: f64) -> f64 {
    let ca = if cai < 1e-7 { cai + 1e-7 } else { cai };
    1.0 / (1.0 + (0.00043 / ca).powf(4.8))
}

/// Nernst potential of Ca²⁺ at 34 °C, mV.
pub fn e_ca(cai: f64) -> f64 {
    1e3 * R * CELSIUS_34 / (2.0 * F) * (CA_OUT / cai).ln()
}

/// Exact update of `x' = (inf − x)/τ` over `dt`.
fn relax(x: &mut f64, (inf, tau): Gate, dt: f64) {
    *x = inf + (*x - inf) * (-dt / tau).exp();
}

/// Floor of internal Ca²⁺, mM.
const MIN_CAI: f64 = 1e-4;
/// Depth of the Ca²⁺ shell, µm.
const CA_DEPTH: f64 = 0.1;

impl State {
    /// Every gate at its steady state for `v`, internal Ca²⁺ at its floor.
    pub fn steady(v: f64) -> Self {
        let inf = |g: Gate| g.0;
        Self {
            nat: [inf(nat_m(v)), inf(nat_h(v))],
            nap: [inf(nap_m(v)), inf(nap_h(v))],
            kp: [inf(kp_m(v)), inf(kp_h(v))],
            kt: [inf(kt_m(v)), inf(kt_h(v))],
            sk: sk_z(MIN_CAI),
            kv3: inf(kv3_m(v)),
            hva: [inf(hva_m(v)), inf(hva_h(v))],
            lva: [inf(lva_m(v)), inf(lva_h(v))],
            im: inf(im_m(v)),
            ih: inf(ih_m(v)),
            cai: MIN_CAI,
        }
    }

    /// Currents at `v` with the gates as they are.
    pub fn currents(&self, d: &Densities, v: f64) -> Currents {
        let g_na =
            d.na_t * self.nat[0].powi(3) * self.nat[1] + d.na_p * self.nap[0].powi(3) * self.nap[1];
        let g_k = d.k_p * self.kp[0].powi(2) * self.kp[1]
            + d.k_t * self.kt[0].powi(4) * self.kt[1]
            + d.sk * self.sk
            + d.kv3 * self.kv3
            + d.im * self.im;
        let g_ca = d.ca_hva * self.hva[0].powi(2) * self.hva[1]
            + d.ca_lva * self.lva[0].powi(2) * self.lva[1];
        let g_h = d.ih * self.ih;
        let f = na_fraction(E_H);
        let ca = if g_ca > 0.0 {
            g_ca * (v - e_ca(self.cai))
        } else {
            0.0
        };
        Currents {
            // Ih splits into Na⁺ and K⁺; the leak is carried by K⁺.
            na: (g_na + f * g_h) * (v - E_NA),
            k: (g_k + (1.0 - f) * g_h) * (v - E_K) + d.pas * (v - E_PAS),
            ca,
            g: g_na + g_k + g_ca + g_h + d.pas,
        }
    }

    /// Advances the gates over `dt` at the potential `v`, and internal Ca²⁺ with the Ca²⁺
    /// current `ica` (µA/cm²) held over the step.
    pub fn advance(&mut self, d: &Densities, v: f64, ica: f64, dt: f64) {
        if d.na_t > 0.0 {
            relax(&mut self.nat[0], nat_m(v), dt);
            relax(&mut self.nat[1], nat_h(v), dt);
        }
        if d.na_p > 0.0 {
            relax(&mut self.nap[0], nap_m(v), dt);
            relax(&mut self.nap[1], nap_h(v), dt);
        }
        if d.k_p > 0.0 {
            relax(&mut self.kp[0], kp_m(v), dt);
            relax(&mut self.kp[1], kp_h(v), dt);
        }
        if d.k_t > 0.0 {
            relax(&mut self.kt[0], kt_m(v), dt);
            relax(&mut self.kt[1], kt_h(v), dt);
        }
        if d.kv3 > 0.0 {
            relax(&mut self.kv3, kv3_m(v), dt);
        }
        if d.ca_hva > 0.0 {
            relax(&mut self.hva[0], hva_m(v), dt);
            relax(&mut self.hva[1], hva_h(v), dt);
        }
        if d.ca_lva > 0.0 {
            relax(&mut self.lva[0], lva_m(v), dt);
            relax(&mut self.lva[1], lva_h(v), dt);
        }
        if d.im > 0.0 {
            relax(&mut self.im, im_m(v), dt);
        }
        if d.ih > 0.0 {
            relax(&mut self.ih, ih_m(v), dt);
        }
        // SK follows the Ca²⁺ at the start of the step (τ = 1 ms).
        if d.sk > 0.0 {
            relax(&mut self.sk, (sk_z(self.cai), 1.0), dt);
        }
        // cai' = −10⁴ ica γ / (2 F depth) − (cai − min)/decay with ica in mA/cm²: exact
        // for an ica held over the step.
        let influx = -1e4 * 1e-3 * ica * d.ca_gamma / (2.0 * F * CA_DEPTH);
        relax(
            &mut self.cai,
            (MIN_CAI + influx * d.ca_decay, d.ca_decay),
            dt,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_match_the_nmodl_formulas_at_sample_points() {
        // NaTa_t m at −20 mV, straight from the .mod expressions.
        let v: f64 = -20.0;
        let a = 0.182 * (v + 38.0) / (1.0 - (-(v + 38.0) / 6.0).exp());
        let b = 0.124 * (-v - 38.0) / (1.0 - (-(-v - 38.0) / 6.0).exp());
        let (inf, tau) = nat_m(v);
        assert!((inf - a / (a + b)).abs() < 1e-12 && (tau - 1.0 / (a + b) / QT).abs() < 1e-12);
        // Ih at −20 mV.
        let a = 0.001 * 6.43 * (v + 154.9) / (((v + 154.9) / 11.9).exp() - 1.0);
        let b = 0.001 * 193.0 * (v / 33.1).exp();
        let (inf, tau) = ih_m(v);
        assert!((inf - a / (a + b)).abs() < 1e-12 && (tau - 1.0 / (a + b)).abs() < 1e-9);
        // Removable singularities are continuous.
        for (f, v0) in [
            (nat_m as fn(f64) -> Gate, -38.0),
            (nat_h, -66.0),
            (hva_m, -27.0),
            (ih_m, -154.9),
        ] {
            let (a, b) = (f(v0), f(v0 + 1e-4));
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-3 * a.1);
        }
        // Gates span 0..1 and time constants are positive over the physiological range.
        for i in -150..=60 {
            let v = i as f64;
            for (inf, tau) in [
                nat_m(v),
                nat_h(v),
                nap_m(v),
                nap_h(v),
                kp_m(v),
                kp_h(v),
                kt_m(v),
                kt_h(v),
                kv3_m(v),
                hva_m(v),
                hva_h(v),
                lva_m(v),
                lva_h(v),
                im_m(v),
                ih_m(v),
            ] {
                assert!(
                    (0.0..=1.0).contains(&inf) && tau > 0.0 && tau.is_finite(),
                    "{v}"
                );
            }
        }
    }

    #[test]
    fn densities_follow_l5pcbiophys3() {
        let soma = Densities::hay(Region::Soma, 0.0, 1300.0);
        assert_eq!((soma.na_t, soma.kv3, soma.cm), (2040.0, 693.0, 1.0));
        let apical = |d| Densities::hay(Region::Apical, d, 1000.0);
        // Ih grows exponentially along the apical dendrite, ×(−0.87 + 2.09 e^3.6) at the tip.
        assert!((apical(0.0).ih - 0.2 * (2.087 - 0.8696)).abs() < 1e-9);
        assert!(apical(1000.0).ih > 15.0 * apical(0.0).ih);
        // The Ca²⁺ hot zone has 100× the LVA and 10× the HVA conductance.
        assert!((apical(700.0).ca_lva / apical(600.0).ca_lva - 100.0).abs() < 1e-9);
        assert!((apical(700.0).ca_hva / apical(900.0).ca_hva - 10.0).abs() < 1e-9);
        let basal = Densities::hay(Region::Basal, 50.0, 1000.0);
        assert_eq!((basal.na_t, basal.cm), (0.0, 2.0));
        assert_eq!(Densities::hay(Region::Axon, 15.0, 1000.0).na_t, 0.0);
    }

    #[test]
    fn currents_are_linear_in_v_with_frozen_gates() {
        let d = Densities::hay(Region::Soma, 0.0, 1300.0);
        let s = State::steady(-70.0);
        let (i0, i1) = (s.currents(&d, -70.0), s.currents(&d, -60.0));
        assert!((i1.total() - i0.total() - 10.0 * i0.g).abs() < 1e-9 * i0.g);
        // The Ih split reverses at E_h: at −45 mV a basal compartment carries only leak.
        let d = Densities::hay(Region::Basal, 0.0, 1000.0);
        let c = State::steady(E_H).currents(&d, E_H);
        assert!((c.total() - d.pas * (E_H - E_PAS)).abs() < 1e-9);
        assert!(c.na < 0.0 && c.k > 0.0);
    }

    #[test]
    fn calcium_influx_raises_cai_and_it_decays_back() {
        let d = Densities::hay(Region::Soma, 0.0, 1300.0);
        let mut s = State::steady(-70.0);
        assert!(e_ca(s.cai) > 100.0);
        s.advance(&d, -70.0, -10.0, 1.0);
        let peak = s.cai;
        assert!(peak > MIN_CAI);
        s.advance(&d, -70.0, 0.0, 5000.0);
        assert!((s.cai - MIN_CAI).abs() < 1e-4 * (peak - MIN_CAI));
    }
}
