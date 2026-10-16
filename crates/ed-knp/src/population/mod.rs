//! Many reduced neurons that both release ions into, and respond to, the extracellular space.
//!
//! Each neuron has two compartments, a soma (Hodgkin-Huxley Na⁺ and K⁺ channels) and a lumped
//! passive dendrite, joined by an axial conductance. Unlike [`crate::neuron::Neuron`], the
//! coupling to the tissue goes both ways: before each [`Population::step`] the caller writes
//! the extracellular Na⁺, K⁺ and X⁻ concentrations at every compartment, the reversal
//! potentials follow them, and each compartment keeps its own intracellular Na⁺, K⁺ and Cl⁻,
//! updated from its currents. This is what spreading depression needs: raised [K⁺]e
//! depolarises a cell, which then releases more K⁺.
//!
//! Membrane, per compartment (densities in mS/cm² and µA/cm², outward positive):
//! - Na⁺: `(g_L^Na + ḡ_Na m³h + g_NaP p∞(V)) (V − E_Na)` (+ the drive conductance on the soma),
//! - K⁺: `(g_L^K + ḡ_K n⁴) (V − E_K)`, Cl⁻: `g_L^Cl (V − E_Cl)`,
//! - Na⁺/K⁺ pump: `I_p = P_max (1 + K_mK/[K]e)⁻² (1 + K_mNa/[Na]i)⁻³`, moving 3 Na⁺ out and
//!   2 K⁺ in per cycle (Kager, Wadman & Somjen 2000).
//!
//! The HH channels are on the soma only. `p∞(V) = 1/(1 + e^{−(V+50)/5})` is a generic
//! persistent Na⁺ current, off by default. The Na⁺ leak and `P_max` are not free: they are
//! set so that the baseline concentrations at [`PopulationParams::v_rest`] are an exact ionic
//! steady state (each ion's net flux is zero), and `[Cl⁻]i` so that `E_Cl = V_rest`.
//! [`PopulationParams::pump_scale`] then weakens or strengthens the pump from there.
//!
//! Time stepping: within one call to [`Population::step`] the concentrations, reversal
//! potentials and pump currents are frozen (they change on the scale of seconds). Each neuron
//! then takes its own internal steps: backward Euler for the two potentials (a 2×2 solve
//! with the conductances frozen at the start of the step) and Rush-Larsen for the gates. The
//! internal step adapts so that V moves by about [`PopulationParams::v_tol`] per step, between
//! `h_min` while spiking and `h_max` at rest or in depolarisation block, so a quiet neuron
//! costs one or two updates per call. Both schemes are stable at any step size, but frozen
//! gates over a long step damp out spike initiation, so where the Na⁺ feedback is strong (near
//! threshold) the step is also held to a quarter of the Na⁺ activation time. Steps are never
//! rejected: the controller only sets the next step from the last change of V, and the
//! threshold cap is what resolves the spike upstroke. Gating functions
//! come from a table in V (linear interpolation, 0.05 mV spacing), as in NEURON.
//!
//! Outputs, per compartment: mean Na⁺, K⁺, X⁻ (Cl⁻) and capacitive currents over the last
//! step, in nA, positive out of the cell. Axial currents cancel, so the four sum to zero over
//! each neuron (to round-off), as a KNP solver with closed boundaries requires.

mod params;
mod rates;
#[cfg(test)]
mod tests;

pub use params::{Calibration, PopulationParams, PUMP_KM_K, PUMP_KM_NA};

use ed_core::constants::{psi_mv, F};
use ed_core::membrane::hh;
use params::pump_fraction;
use rates::*;

/// Smallest concentration fed into a Nernst potential, mM.
const MIN_CONC: f64 = 1e-6;
/// A compartment counts as excitable when its Na⁺ feedback conductance exceeds this fraction
/// of its other conductances; its internal step is then at most `TAU_FRACTION` of the Na⁺
/// activation time constant (`NAP_TAU` for the persistent current), in ms.
const EXCITABLE: f64 = 0.5;
const TAU_FRACTION: f64 = 0.25;
const NAP_TAU: f64 = 0.5;

/// `N` two-compartment neurons. Compartment `2i` is neuron `i`'s soma, `2i + 1` its dendrite.
/// See the [module docs](self).
#[derive(Debug, Clone)]
pub struct Population {
    params: PopulationParams,
    cal: [Calibration; 2],
    table: RateTable,
    /// Membrane area in units of 1e5 µm² (densities → nA, µS, nF), per compartment kind.
    scale: [f64; 2],
    time: f64,

    /// Membrane potential per compartment, mV.
    v: Vec<f64>,
    /// Somatic HH gates per neuron.
    m: Vec<f64>,
    h: Vec<f64>,
    n: Vec<f64>,
    /// Next internal step per neuron, ms.
    h_next: Vec<f64>,
    /// Extra Na⁺ conductance on each soma, µS.
    drive: Vec<f64>,

    /// Extracellular Na⁺, K⁺ and X⁻ at each compartment, mM. Written by the caller.
    pub na_e: Vec<f64>,
    pub k_e: Vec<f64>,
    pub x_e: Vec<f64>,
    /// Intracellular Na⁺, K⁺ and Cl⁻ per compartment, mM.
    na_i: Vec<f64>,
    k_i: Vec<f64>,
    cl_i: Vec<f64>,

    i_na: Vec<f64>,
    i_k: Vec<f64>,
    i_x: Vec<f64>,
    i_cap: Vec<f64>,
    /// Internal steps taken over all neurons by the last `step`.
    substeps: usize,
}

impl Population {
    /// `count` neurons at rest at the baseline concentrations.
    pub fn new(count: usize, params: PopulationParams) -> Self {
        let p = &params;
        assert!(
            p.h_min > 0.0 && p.h_max >= p.h_min && p.v_tol > 0.0,
            "invalid internal step bounds"
        );
        assert!(p.area.iter().chain(&p.volume).all(|&x| x > 0.0) && p.g_axial >= 0.0);
        let cal = [p.calibrate(0), p.calibrate(1)];
        let len = 2 * count;
        let (m, h, n) = hh::steady_state(p.v_rest, p.v_rest);
        let per_comp = |f: fn(&Calibration) -> f64| (0..len).map(|c| f(&cal[c % 2])).collect();
        Self {
            scale: p.area.map(|a| a * 1e-5),
            time: 0.0,
            v: vec![p.v_rest; len],
            m: vec![m; count],
            h: vec![h; count],
            n: vec![n; count],
            h_next: vec![p.h_max; count],
            drive: vec![0.0; count],
            na_e: vec![p.ecs[0]; len],
            k_e: vec![p.ecs[1]; len],
            x_e: vec![p.ecs[2]; len],
            na_i: vec![p.ics[0]; len],
            k_i: vec![p.ics[1]; len],
            cl_i: per_comp(|c| c.cl_i),
            i_na: vec![0.0; len],
            i_k: vec![0.0; len],
            i_x: vec![0.0; len],
            i_cap: vec![0.0; len],
            substeps: 0,
            cal,
            table: RateTable::new(p.v_rest),
            params,
        }
    }

    pub fn params(&self) -> &PopulationParams {
        &self.params
    }
    pub fn calibration(&self) -> [Calibration; 2] {
        self.cal
    }
    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.h.len()
    }
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }
    pub fn time_ms(&self) -> f64 {
        self.time
    }
    /// Membrane potential per compartment, mV.
    pub fn v(&self) -> &[f64] {
        &self.v
    }
    /// Intracellular Na⁺, K⁺ and Cl⁻ per compartment, mM.
    pub fn na_i(&self) -> &[f64] {
        &self.na_i
    }
    pub fn k_i(&self) -> &[f64] {
        &self.k_i
    }
    pub fn cl_i(&self) -> &[f64] {
        &self.cl_i
    }
    /// Mean currents per compartment over the last step, nA, outward positive.
    pub fn i_na(&self) -> &[f64] {
        &self.i_na
    }
    pub fn i_k(&self) -> &[f64] {
        &self.i_k
    }
    pub fn i_x(&self) -> &[f64] {
        &self.i_x
    }
    pub fn i_cap(&self) -> &[f64] {
        &self.i_cap
    }
    /// Internal steps taken by all neurons together in the last step.
    pub fn substeps(&self) -> usize {
        self.substeps
    }

    /// Sets the excitatory drive of neuron `i`: a Na⁺ conductance on its soma, nS.
    pub fn set_drive(&mut self, i: usize, g_ns: f64) {
        assert!(
            g_ns >= 0.0 && g_ns.is_finite(),
            "drive must be finite and ≥ 0"
        );
        self.drive[i] = g_ns * 1e-3;
    }

    /// Changes the pump capacity relative to the calibrated one.
    pub fn set_pump_scale(&mut self, s: f64) {
        assert!(
            s >= 0.0 && s.is_finite(),
            "pump scale must be finite and ≥ 0"
        );
        self.params.pump_scale = s;
    }

    /// Advances every neuron by `dt` ms with the current extracellular concentrations, then
    /// updates the intracellular ones.
    pub fn step(&mut self, dt: f64) {
        assert!(dt > 0.0 && dt.is_finite(), "time step must be positive");
        let p = &self.params;
        let psi = psi_mv(p.temperature);
        let mut substeps = 0;
        for i in 0..self.len() {
            // Per compartment: reversal potentials and pump current (nA), frozen over dt.
            let mut e = [[0.0; 3]; 2];
            let mut pump = [0.0; 2];
            let mut gl = [[0.0; 3]; 2];
            let mut c_m = [0.0; 2];
            for s in 0..2 {
                let c = 2 * i + s;
                let ln = |o: f64, i: f64| (o.max(MIN_CONC) / i.max(MIN_CONC)).ln();
                e[s] = [
                    psi * ln(self.na_e[c], self.na_i[c]),
                    psi * ln(self.k_e[c], self.k_i[c]),
                    -psi * ln(self.x_e[c], self.cl_i[c]),
                ];
                let sc = self.scale[s];
                let rate = pump_fraction(self.k_e[c].max(0.0), self.na_i[c].max(0.0));
                pump[s] = p.pump_scale * self.cal[s].p_max * rate * sc;
                gl[s] = [
                    self.cal[s].g_leak_na * sc,
                    p.g_leak_k[s] * sc,
                    p.g_leak_cl * sc,
                ];
                c_m[s] = p.c_m * sc;
            }
            let (s0, d0) = (2 * i, 2 * i + 1);
            let ga = p.g_axial;
            let mut v = [self.v[s0], self.v[d0]];
            let v_start = v;
            // Charge moved per ion and compartment, nA·ms.
            let mut q = [[0.0; 3]; 2];
            let mut t = 0.0;
            // Rate table rows at the current potentials, reused by the next internal step.
            let mut row = [self.table.at(v[0]), self.table.at(v[1])];
            while t < dt * (1.0 - 1e-12) {
                let (m, hg, n) = (self.m[i], self.h[i], self.n[i]);
                let mut g = gl;
                g[0][0] += self.scale[0] * (p.g_na_bar * m * m * m * hg) + self.drive[i];
                g[0][1] += self.scale[0] * p.g_k_bar * (n * n) * (n * n);
                let mut h = self.h_next[i];
                for s in 0..2 {
                    let r = &row[s];
                    g[s][0] += self.scale[s] * p.g_nap * r[P_INF];
                    // Near threshold the Na⁺ feedback (−∂I_Na/∂V with m at m∞) competes with
                    // the other conductances. Frozen gates would damp it out over a long step,
                    // so resolve the activation time there.
                    let mut slope = p.g_nap * r[DP_INF];
                    if s == 0 {
                        let mi = r[M_INF];
                        slope += p.g_na_bar * 3.0 * mi * mi * r[DM_INF] * hg;
                    }
                    let feedback = -self.scale[s] * slope * (v[s] - e[s][0]);
                    if feedback > EXCITABLE * (g[s].iter().sum::<f64>() + ga) {
                        let tau = if s == 0 { 1.0 / r[M_SUM] } else { NAP_TAU };
                        h = h.min(TAU_FRACTION * tau);
                    }
                }
                // The last step may be cut short to end on dt; the growth limit below still
                // refers to the planned step, so a short remainder does not shrink the next one.
                let planned = h.max(p.h_min);
                let h = planned.min(dt - t);
                // Backward Euler: (C/h + G + g_a) V' − g_a V'_other = C/h V + Σ g E − P.
                let mut a = [0.0; 2];
                let mut b = [0.0; 2];
                for s in 0..2 {
                    let gs: f64 = g[s].iter().sum();
                    let ge: f64 = (0..3).map(|k| g[s][k] * e[s][k]).sum();
                    a[s] = c_m[s] / h + gs + ga;
                    b[s] = c_m[s] / h * v[s] + ge - pump[s];
                }
                let det = a[0] * a[1] - ga * ga;
                let vn = [
                    (b[0] * a[1] + ga * b[1]) / det,
                    (b[1] * a[0] + ga * b[0]) / det,
                ];
                for s in 0..2 {
                    for k in 0..3 {
                        q[s][k] += h * g[s][k] * (vn[s] - e[s][k]);
                    }
                }
                row = [self.table.at(vn[0]), self.table.at(vn[1])];
                let r = &row[0];
                let rl = |x: f64, inf: f64, sum: f64| inf + (x - inf) * (-h * sum).exp();
                self.m[i] = rl(m, r[M_INF], r[M_SUM]);
                self.h[i] = rl(hg, r[H_INF], r[H_SUM]);
                self.n[i] = rl(n, r[N_INF], r[N_SUM]);
                let dv = (vn[0] - v[0]).abs().max((vn[1] - v[1]).abs());
                let next = if dv > 0.0 { p.v_tol * h / dv } else { p.h_max };
                self.h_next[i] = next.min(2.0 * planned).clamp(p.h_min, p.h_max);
                v = vn;
                t += h;
                substeps += 1;
            }
            for s in 0..2 {
                let c = 2 * i + s;
                self.v[c] = v[s];
                self.i_na[c] = q[s][0] / dt + 3.0 * pump[s];
                self.i_k[c] = q[s][1] / dt - 2.0 * pump[s];
                self.i_x[c] = q[s][2] / dt;
                self.i_cap[c] = c_m[s] * (v[s] - v_start[s]) / dt;
                // Outward current removes cations from the cell and brings anions in.
                let to_mm = dt * 1e6 / (F * p.volume[s]);
                self.na_i[c] -= self.i_na[c] * to_mm;
                self.k_i[c] -= self.i_k[c] * to_mm;
                self.cl_i[c] += self.i_x[c] * to_mm;
                debug_assert!(
                    self.na_i[c] > 0.0 && self.k_i[c] > 0.0 && self.cl_i[c] > 0.0,
                    "intracellular concentration of compartment {c} went negative: step too long"
                );
            }
        }
        self.substeps = substeps;
        self.time += dt;
    }
}
