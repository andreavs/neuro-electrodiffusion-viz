//! Membrane models shared by the KNP point neuron and the KNP-EMI interface.
//!
//! Units follow the membrane convention used in the thesis (Publication 2):
//! potential mV, time ms, conductance mS/cm², capacitance µF/cm², current
//! density µA/cm² (so `I / C_M` is in mV/ms), concentrations mM.
//!
//! Sign convention: channel currents `I_ch^k = g_k (φ_M − E_k)` are positive
//! outward, and an isolated patch obeys `C_M dφ_M/dt = I_ext − I_ch`, where
//! `I_ext` is an injected (inward-positive, depolarising) stimulus. A coupled
//! solver can instead pass `I_M` as `I_ext`, which matches the KNP-EMI ODE
//! `C_M ∂φ_M/∂t = I_M − I_ch`.
//!
//! Model, per patch:
//! - Na⁺: `(ḡ_Na m³h + g_L^Na + g_syn(t)) (φ_M − E_Na)`
//! - K⁺:  `(ḡ_K n⁴ + g_L^K) (φ_M − E_K)`
//! - Cl⁻: `g_L^Cl (φ_M − E_Cl)`
//!
//! with Nernst potentials from the patch's own concentrations. The synapse
//! `g_syn(t) = ḡ_syn e^{−(t−t₀)/τ}` for `t ≥ t₀` opens Na⁺ channels, as in
//! KNP-EMI, so `E_syn = E_Na` and its current is carried by Na⁺. Setting
//! [`MembraneParams::hodgkin_huxley`] to `false` gives the passive leak model.
//!
//! State is stored as a struct of arrays so thousands of patches update in a
//! tight, allocation-free loop.

pub mod hh;

use crate::constants::psi_mv;

/// Membrane parameters shared by all patches in a [`MembranePatches`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MembraneParams {
    /// Membrane capacitance C_M, µF/cm².
    pub c_m: f64,
    /// Na⁺ leak conductance g_L^Na, mS/cm².
    pub g_leak_na: f64,
    /// K⁺ leak conductance g_L^K, mS/cm².
    pub g_leak_k: f64,
    /// Cl⁻ leak conductance g_L^Cl, mS/cm².
    pub g_leak_cl: f64,
    /// Whether the Hodgkin-Huxley Na⁺ and K⁺ channels are present.
    pub hodgkin_huxley: bool,
    /// Maximal HH Na⁺ conductance ḡ_Na, mS/cm².
    pub g_na_bar: f64,
    /// Maximal HH K⁺ conductance ḡ_K, mS/cm².
    pub g_k_bar: f64,
    /// Reference potential the HH rate functions are shifted to, mV.
    pub v_rest_hh: f64,
    /// Synaptic decay time constant τ, ms.
    pub syn_tau: f64,
    /// Temperature for the Nernst potentials, K.
    pub temperature: f64,
}

impl MembraneParams {
    /// Hodgkin-Huxley membrane with the KNP-EMI parameters (Publication 2).
    pub const HODGKIN_HUXLEY: Self = Self {
        c_m: 1.0,
        g_leak_na: 0.2,
        g_leak_k: 0.8,
        g_leak_cl: 0.0,
        hodgkin_huxley: true,
        g_na_bar: 120.0,
        g_k_bar: 36.0,
        v_rest_hh: -65.0,
        syn_tau: 2.0,
        temperature: crate::constants::T_DEFAULT,
    };

    /// Passive leak membrane with the KNP-EMI leak conductances.
    pub const PASSIVE: Self = Self {
        hodgkin_huxley: false,
        ..Self::HODGKIN_HUXLEY
    };
}

impl Default for MembraneParams {
    fn default() -> Self {
        Self::HODGKIN_HUXLEY
    }
}

/// Intra- and extracellular concentrations (mM) seen by one patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonConcentrations {
    pub na_i: f64,
    pub na_e: f64,
    pub k_i: f64,
    pub k_e: f64,
    pub cl_i: f64,
    pub cl_e: f64,
}

impl IonConcentrations {
    /// KNP-EMI initial concentrations (Publication 2).
    pub const KNP_EMI: Self = Self {
        na_i: 12.0,
        na_e: 100.0,
        k_i: 125.0,
        k_e: 4.0,
        cl_i: 137.0,
        cl_e: 104.0,
    };
}

impl Default for IonConcentrations {
    fn default() -> Self {
        Self::KNP_EMI
    }
}

/// Membrane potential and HH gates of one patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchState {
    /// φ_M, mV.
    pub phi_m: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
}

impl PatchState {
    /// Initial state from Publication 2: φ_M⁰ = −67.74 mV, m₀ 0.0379,
    /// h₀ 0.688, n₀ 0.276. With the default parameters and concentrations this
    /// is close to, but not exactly, an equilibrium (net current ≈ −3.6 µA/cm²):
    /// an isolated patch overshoots to ≈ −65.5 mV within 1 ms and settles at
    /// [`rest_state`] (≈ −66.1 mV) after about 10 ms.
    pub const KNP_EMI: Self = Self {
        phi_m: -67.74,
        m: 0.0379,
        h: 0.688,
        n: 0.276,
    };
}

/// Nernst potentials (E_Na, E_K, E_Cl) in mV.
#[inline]
fn nernst_all(c: &IonConcentrations, psi: f64) -> (f64, f64, f64) {
    (
        psi * (c.na_e / c.na_i).ln(),
        psi * (c.k_e / c.k_i).ln(),
        -psi * (c.cl_e / c.cl_i).ln(),
    )
}

/// Steady-state channel current (µA/cm², outward positive) at potential `phi`
/// with the gates at their steady state and no synaptic input.
fn steady_current(p: &MembraneParams, e: (f64, f64, f64), phi: f64) -> f64 {
    let (g_na, g_k) = if p.hodgkin_huxley {
        let (m, h, n) = hh::steady_state(phi, p.v_rest_hh);
        (p.g_na_bar * m * m * m * h, p.g_k_bar * n.powi(4))
    } else {
        (0.0, 0.0)
    };
    (g_na + p.g_leak_na) * (phi - e.0)
        + (g_k + p.g_leak_k) * (phi - e.1)
        + p.g_leak_cl * (phi - e.2)
}

/// Resting state of an isolated patch: the most hyperpolarised potential where
/// the steady-state channel current vanishes, with gates at steady state.
/// Returns `None` if no zero crossing is found in [−150, 50] mV.
pub fn rest_state(p: &MembraneParams, c: &IonConcentrations) -> Option<PatchState> {
    let e = nernst_all(c, psi_mv(p.temperature));
    let f = |phi| steady_current(p, e, phi);
    // Steady-state current is increasing at a stable rest; scan upwards for
    // the first sign change from negative to positive, then bisect.
    let step = 0.25;
    let mut lo = -150.0;
    let mut f_lo = f(lo);
    while lo < 50.0 {
        let hi = lo + step;
        let f_hi = f(hi);
        if f_lo <= 0.0 && f_hi > 0.0 {
            let (mut a, mut b) = (lo, hi);
            for _ in 0..60 {
                let mid = 0.5 * (a + b);
                if f(mid) > 0.0 {
                    b = mid;
                } else {
                    a = mid;
                }
            }
            let phi_m = 0.5 * (a + b);
            let (m, h, n) = hh::steady_state(phi_m, p.v_rest_hh);
            return Some(PatchState { phi_m, m, h, n });
        }
        lo = hi;
        f_lo = f_hi;
    }
    None
}

/// A population of membrane patches in struct-of-arrays layout.
///
/// Callers may write the concentration arrays between calls (e.g. from a KNP
/// solver); Nernst potentials are recomputed at the start of every
/// [`advance`](Self::advance) and [`compute_currents`](Self::compute_currents).
/// The arrays are public for zero-copy access, but their lengths must not
/// change after construction.
#[derive(Debug, Clone)]
pub struct MembranePatches {
    pub params: MembraneParams,

    /// Membrane potential φ_M, mV.
    pub phi_m: Vec<f64>,
    /// HH Na⁺ activation.
    pub m: Vec<f64>,
    /// HH Na⁺ inactivation.
    pub h: Vec<f64>,
    /// HH K⁺ activation.
    pub n: Vec<f64>,

    /// Peak synaptic conductance ḡ_syn per patch, mS/cm² (0 = no synapse).
    pub syn_g: Vec<f64>,
    /// Synaptic onset time t₀ per patch, ms.
    pub syn_t0: Vec<f64>,

    /// Concentrations, mM.
    pub na_i: Vec<f64>,
    pub na_e: Vec<f64>,
    pub k_i: Vec<f64>,
    pub k_e: Vec<f64>,
    pub cl_i: Vec<f64>,
    pub cl_e: Vec<f64>,

    /// Na⁺ channel current (including synaptic), µA/cm², outward positive.
    pub i_na: Vec<f64>,
    /// K⁺ channel current, µA/cm², outward positive.
    pub i_k: Vec<f64>,
    /// Cl⁻ channel current, µA/cm², outward positive.
    pub i_cl: Vec<f64>,
    /// Total channel current I_ch = I_Na + I_K + I_Cl, µA/cm².
    pub i_ch: Vec<f64>,
    /// Total channel conductance Σ g_k = ∂I_ch/∂φ_M, mS/cm², for
    /// semi-implicit coupling.
    pub g_ch: Vec<f64>,

    /// Tabulated gate updates, when enabled with [`use_gate_table`](Self::use_gate_table).
    gate_table: Option<hh::GateTable>,
    use_table: bool,

    // Nernst potentials, refreshed from the concentrations on each call.
    e_na: Vec<f64>,
    e_k: Vec<f64>,
    e_cl: Vec<f64>,
}

impl MembranePatches {
    /// `len` patches with uniform concentrations, initialised at [`rest_state`]
    /// (falling back to [`PatchState::KNP_EMI`] if no rest exists).
    pub fn new(len: usize, params: MembraneParams, conc: IonConcentrations) -> Self {
        let state = rest_state(&params, &conc).unwrap_or(PatchState::KNP_EMI);
        Self::with_state(len, params, conc, state)
    }

    /// `len` patches with uniform concentrations and the given initial state.
    pub fn with_state(
        len: usize,
        params: MembraneParams,
        conc: IonConcentrations,
        state: PatchState,
    ) -> Self {
        let mut patches = Self {
            params,
            phi_m: vec![state.phi_m; len],
            m: vec![state.m; len],
            h: vec![state.h; len],
            n: vec![state.n; len],
            syn_g: vec![0.0; len],
            syn_t0: vec![0.0; len],
            na_i: vec![conc.na_i; len],
            na_e: vec![conc.na_e; len],
            k_i: vec![conc.k_i; len],
            k_e: vec![conc.k_e; len],
            cl_i: vec![conc.cl_i; len],
            cl_e: vec![conc.cl_e; len],
            i_na: vec![0.0; len],
            i_k: vec![0.0; len],
            i_cl: vec![0.0; len],
            i_ch: vec![0.0; len],
            g_ch: vec![0.0; len],
            gate_table: None,
            use_table: false,
            e_na: vec![0.0; len],
            e_k: vec![0.0; len],
            e_cl: vec![0.0; len],
        };
        patches.compute_currents(0.0);
        patches
    }

    /// Number of patches.
    pub fn len(&self) -> usize {
        self.phi_m.len()
    }

    /// Whether there are no patches.
    pub fn is_empty(&self) -> bool {
        self.phi_m.is_empty()
    }

    /// Concentrations of patch `i`.
    pub fn concentrations(&self, i: usize) -> IonConcentrations {
        IonConcentrations {
            na_i: self.na_i[i],
            na_e: self.na_e[i],
            k_i: self.k_i[i],
            k_e: self.k_e[i],
            cl_i: self.cl_i[i],
            cl_e: self.cl_e[i],
        }
    }

    /// Use a [`hh::GateTable`] for the gate updates in [`advance`](Self::advance): several
    /// times faster for large populations, with gate values within ~10⁻⁵ of the exact
    /// update. Off by default.
    pub fn use_gate_table(&mut self, enabled: bool) {
        self.use_table = enabled;
    }

    /// Schedule a synaptic input with peak conductance `g` (mS/cm²) starting at
    /// `t0` (ms) on patch `i`.
    pub fn set_synapse(&mut self, i: usize, g: f64, t0: f64) {
        self.syn_g[i] = g;
        self.syn_t0[i] = t0;
    }

    /// Synaptic conductance of patch `i` at time `t`, mS/cm².
    #[inline]
    pub fn synaptic_conductance(&self, i: usize, t: f64) -> f64 {
        let g = self.syn_g[i];
        let dt = t - self.syn_t0[i];
        if g == 0.0 || dt < 0.0 {
            0.0
        } else {
            g * (-dt / self.params.syn_tau).exp()
        }
    }

    fn refresh_nernst(&mut self) {
        let psi = psi_mv(self.params.temperature);
        for i in 0..self.len() {
            let c = self.concentrations(i);
            debug_assert!(
                [c.na_i, c.na_e, c.k_i, c.k_e, c.cl_i, c.cl_e]
                    .iter()
                    .all(|&x| x > 0.0),
                "non-positive concentration at patch {i}: {c:?}"
            );
            let (e_na, e_k, e_cl) = nernst_all(&c, psi);
            self.e_na[i] = e_na;
            self.e_k[i] = e_k;
            self.e_cl[i] = e_cl;
        }
    }

    /// Conductances (g_Na, g_K, g_Cl) of patch `i` at time `t`, mS/cm².
    #[inline]
    fn conductances(&self, i: usize, t: f64) -> (f64, f64, f64) {
        let p = &self.params;
        let mut g_na = p.g_leak_na + self.synaptic_conductance(i, t);
        let mut g_k = p.g_leak_k;
        if p.hodgkin_huxley {
            let (m, h, n) = (self.m[i], self.h[i], self.n[i]);
            g_na += p.g_na_bar * m * m * m * h;
            let n2 = n * n;
            g_k += p.g_k_bar * n2 * n2;
        }
        (g_na, g_k, p.g_leak_cl)
    }

    /// Fill the per-ion and total channel currents (and conductance) from the current state at
    /// time `t` (ms).
    pub fn compute_currents(&mut self, t: f64) {
        self.refresh_nernst();
        self.fill_currents(t);
    }

    fn fill_currents(&mut self, t: f64) {
        for i in 0..self.len() {
            let (g_na, g_k, g_cl) = self.conductances(i, t);
            let phi = self.phi_m[i];
            let i_na = g_na * (phi - self.e_na[i]);
            let i_k = g_k * (phi - self.e_k[i]);
            let i_cl = g_cl * (phi - self.e_cl[i]);
            self.i_na[i] = i_na;
            self.i_k[i] = i_k;
            self.i_cl[i] = i_cl;
            self.i_ch[i] = i_na + i_k + i_cl;
            self.g_ch[i] = g_na + g_k + g_cl;
        }
    }

    /// Advance all patches from `t` to `t + dt` (ms) using `substeps` equal
    /// substeps, with optional extra current `i_ext` (µA/cm², inward
    /// positive, held constant over `dt`). Concentrations are held fixed.
    ///
    /// Each substep updates φ_M exactly for frozen conductances (an
    /// exponential relaxation towards the conductance-weighted reversal
    /// potential) and then the gates with Rush-Larsen, so the scheme is
    /// stable for any step size. The synaptic conductance is sampled at the
    /// start of each substep, so an onset inside a substep takes effect at the
    /// next one. On return the current arrays hold the currents at `t + dt`.
    pub fn advance(&mut self, t: f64, dt: f64, substeps: usize, i_ext: Option<&[f64]>) {
        assert!(substeps > 0, "substeps must be positive");
        if let Some(ext) = i_ext {
            assert_eq!(ext.len(), self.len(), "i_ext length mismatch");
        }
        self.refresh_nernst();
        let h = dt / substeps as f64;
        let c_m = self.params.c_m;
        let hh = self.params.hodgkin_huxley;
        let v_rest = self.params.v_rest_hh;
        if hh
            && self.use_table
            && !self
                .gate_table
                .as_ref()
                .is_some_and(|t| t.matches(v_rest, h))
        {
            self.gate_table = Some(hh::GateTable::new(v_rest, h));
        }
        let table = if self.use_table {
            self.gate_table.take()
        } else {
            None
        };
        for s in 0..substeps {
            let ts = t + s as f64 * h;
            for i in 0..self.len() {
                let (g_na, g_k, g_cl) = self.conductances(i, ts);
                let g = g_na + g_k + g_cl;
                let src = i_ext.map_or(0.0, |e| e[i]);
                let phi = self.phi_m[i];
                let new_phi = if g > 0.0 {
                    let phi_inf =
                        (g_na * self.e_na[i] + g_k * self.e_k[i] + g_cl * self.e_cl[i] + src) / g;
                    phi_inf + (phi - phi_inf) * (-h * g / c_m).exp()
                } else {
                    phi + h * src / c_m
                };
                self.phi_m[i] = new_phi;
                if let Some(t) = &table {
                    [self.m[i], self.h[i], self.n[i]] =
                        t.step(new_phi, [self.m[i], self.h[i], self.n[i]]);
                } else if hh {
                    self.m[i] = hh::rates_m(new_phi, v_rest).rush_larsen(self.m[i], h);
                    self.h[i] = hh::rates_h(new_phi, v_rest).rush_larsen(self.h[i], h);
                    self.n[i] = hh::rates_n(new_phi, v_rest).rush_larsen(self.n[i], h);
                }
            }
        }
        if table.is_some() {
            self.gate_table = table;
        }
        self.fill_currents(t + dt);
    }
}

#[cfg(test)]
mod tests;
