//! The Hay et al. cell as a current source: the cable equation on [`HayMorphology`] with
//! the [`channels`](super::channels) of `L5PCbiophys3`.
//!
//! The scheme and conventions are those of [`crate::neuron::Neuron`]: each internal step
//! (Δt = 0.025 ms, as NEURON in Publication 1) is backward Euler for the potentials with the
//! channel currents linearised for frozen gates (exact for conductance-based currents),
//! solved by Hines elimination, followed by the exact exponential gate update at the new
//! potential. Branches meet at zero-area nodes and soma children attach at the soma centre,
//! as in NEURON, and channel densities follow NEURON's `distribute_channels` exactly; the
//! cell then matches NEURON's resting potential and firing rates (see the tests). Outputs are per-compartment currents in nA, positive out of the cell,
//! averaged over the internal steps of the last [`Simulation::step`] call. Ionic,
//! capacitive and electrode currents sum to zero over the cell to round-off.
//!
//! Ion assignment of the currents that NEURON does not tie to an ion:
//! - Ih (E_h = −45 mV) and the synapses (E = 0 mV) are split into a Na⁺ and a K⁺ current
//!   reversing at E_Na and E_K whose sum is the original current at every potential.
//! - The leak (E_pas = −90 mV, below E_K) is carried by K⁺.
//! - Ca²⁺ currents are reported separately ([`HayNeuron::i_ca`]).
//!
//! Inputs as for the simple cell: a somatic current [`HayNeuron::i_stim`] and Poisson
//! excitatory events at [`HayNeuron::syn_rate_hz`], placed uniformly over the membrane area
//! as in Publication 1, each adding `syn_weight_ns` to an exponential synapse (τ = 2 ms).

use super::channels::{self, Densities, State, E_K, E_NA};
use super::HayMorphology;
use crate::neuron::Morphology;
use ed_core::Simulation;

/// Parameters of a [`HayNeuron`].
#[derive(Debug, Clone, PartialEq)]
pub struct HayParams {
    /// Time step, ms.
    pub dt: f64,
    /// Axial resistivity, Ω·cm (100 in Hay et al.).
    pub ra: f64,
    /// Peak conductance added by one synaptic event, nS.
    pub syn_weight_ns: f64,
    /// Synaptic decay time constant, ms.
    pub syn_tau: f64,
    /// RNG seed for the Poisson input.
    pub seed: u64,
    /// Simulated time to relax to rest during construction (with Δt = 0.5 ms), ms.
    pub settle_ms: f64,
}

impl Default for HayParams {
    fn default() -> Self {
        Self {
            dt: 0.025,
            ra: 100.0,
            syn_weight_ns: 1.0,
            syn_tau: 2.0,
            seed: 1,
            settle_ms: 500.0,
        }
    }
}

/// Initial potential before settling, mV.
const V_INIT: f64 = -75.0;

/// The detailed pyramidal cell. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct HayNeuron {
    params: HayParams,
    /// Total rate of excitatory synaptic events over the whole membrane, Hz.
    pub syn_rate_hz: f64,
    /// Current injected into the soma, nA (depolarising positive).
    pub i_stim: f64,

    cell: HayMorphology,
    centres: Vec<[f64; 3]>,
    /// Membrane area in units of 1e-3 cm² (densities in µA/cm², mS/cm² → nA, µS).
    scale: Vec<f64>,
    /// Cable nodes: the compartments plus a zero-area node at every branch point, each
    /// after its parent, as NEURON joins child sections at parent(1).
    node: Vec<usize>,
    parent: Vec<usize>,
    /// Axial conductance of each node to its parent, µS, and the sum over its neighbours.
    g_axial: Vec<f64>,
    g_axial_sum: Vec<f64>,
    dens: Vec<Densities>,
    state: Vec<State>,
    /// Synaptic conductance, µS.
    g_syn: Vec<f64>,
    syn_cdf: Vec<f64>,
    rng: u64,
    time: f64,

    v: Vec<f64>,
    i_na: Vec<f64>,
    i_k: Vec<f64>,
    i_ca: Vec<f64>,
    i_cap: Vec<f64>,
    i_electrode: f64,
    diag: Vec<f64>,
    rhs: Vec<f64>,
}

impl HayNeuron {
    /// Build the cell and relax it to rest for `params.settle_ms`; time starts at 0 after.
    pub fn new(cell: HayMorphology, params: HayParams) -> Self {
        assert!(params.dt > 0.0 && params.ra > 0.0 && params.syn_tau > 0.0);
        let morph = &cell.morph;
        let n = morph.len();
        let comps = &morph.compartments;
        let r: Vec<f64> = comps
            .iter()
            .map(|c| c.axial_resistance(params.ra))
            .collect();
        let mut kids = vec![0usize; n];
        comps
            .iter()
            .filter_map(|c| c.parent)
            .for_each(|p| kids[p] += 1);
        // Children of the soma attach at its centre (soma(0.5)) through their own half
        // resistance; children of a branch point meet at a zero-area node at the parent's
        // end; consecutive segments of a section couple through both half resistances.
        let (mut node, mut parent, mut g_axial) = (vec![0; n], vec![0], vec![0.0]);
        let mut junction = vec![usize::MAX; n];
        for i in 1..n {
            let p = comps[i].parent.unwrap();
            let (to, g) = if p == 0 {
                (node[0], 2.0 / r[i])
            } else if kids[p] >= 2 {
                (junction[p], 2.0 / r[i])
            } else {
                (node[p], 2.0 / (r[i] + r[p]))
            };
            node[i] = parent.len();
            parent.push(to);
            g_axial.push(g);
            if kids[i] >= 2 {
                junction[i] = parent.len();
                parent.push(node[i]);
                g_axial.push(2.0 / r[i]);
            }
        }
        let mut g_axial_sum = vec![0.0; parent.len()];
        for j in 1..parent.len() {
            g_axial_sum[j] += g_axial[j];
            g_axial_sum[parent[j]] += g_axial[j];
        }
        // NEURON's `distribute_channels` also sets x = 1 of each section, which writes the
        // distal end's density into the section's last segment; do the same.
        let d = &cell.distance;
        let mut d_end = vec![0.0; n];
        for i in 1..n {
            let p = comps[i].parent.unwrap();
            d_end[i] = if p != 0 && kids[p] == 1 {
                d[i] + 0.5 * (d[i] - d[p])
            } else {
                2.0 * d[i] - d_end[p]
            };
        }
        let dens: Vec<Densities> = (0..n)
            .map(|i| {
                let dist = if kids[i] == 1 { d[i] } else { d_end[i] };
                Densities::hay(cell.region[i], dist, cell.max_apical)
            })
            .collect();
        let nodes = parent.len();
        let mut acc = 0.0;
        let mut syn_cdf: Vec<f64> = comps
            .iter()
            .map(|c| {
                acc += c.area();
                acc
            })
            .collect();
        syn_cdf.iter_mut().for_each(|x| *x /= acc);
        let mut neuron = Self {
            syn_rate_hz: 0.0,
            i_stim: 0.0,
            centres: comps.iter().map(|c| c.centre()).collect(),
            scale: comps.iter().map(|c| c.area() * 1e-5).collect(),
            node,
            parent,
            g_axial,
            g_axial_sum,
            dens,
            state: vec![State::steady(V_INIT); n],
            g_syn: vec![0.0; n],
            syn_cdf,
            rng: splitmix64(params.seed) | 1,
            time: 0.0,
            v: vec![V_INIT; n],
            i_na: vec![0.0; n],
            i_k: vec![0.0; n],
            i_ca: vec![0.0; n],
            i_cap: vec![0.0; n],
            i_electrode: 0.0,
            diag: vec![0.0; nodes],
            rhs: vec![0.0; nodes],
            cell,
            params,
        };
        let dt = std::mem::replace(&mut neuron.params.dt, 0.5);
        neuron.step((neuron.params.settle_ms / 0.5).round() as usize);
        neuron.params.dt = dt;
        neuron.time = 0.0;
        neuron
    }

    /// The `cell1` morphology with default parameters.
    pub fn cell1() -> Self {
        Self::new(HayMorphology::cell1(), HayParams::default())
    }

    pub fn params(&self) -> &HayParams {
        &self.params
    }
    pub fn cell(&self) -> &HayMorphology {
        &self.cell
    }
    pub fn morphology(&self) -> &Morphology {
        &self.cell.morph
    }
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// Compartment centres (point-source positions), µm.
    pub fn centres(&self) -> &[[f64; 3]] {
        &self.centres
    }
    /// Membrane potential per compartment, mV.
    pub fn v(&self) -> &[f64] {
        &self.v
    }
    /// Na⁺ current per compartment over the last step, nA, outward positive.
    pub fn i_na(&self) -> &[f64] {
        &self.i_na
    }
    /// K⁺ current per compartment over the last step, nA, outward positive.
    pub fn i_k(&self) -> &[f64] {
        &self.i_k
    }
    /// Ca²⁺ current per compartment over the last step, nA, outward positive.
    pub fn i_ca(&self) -> &[f64] {
        &self.i_ca
    }
    /// Capacitive current per compartment over the last step, nA, outward positive.
    pub fn i_cap(&self) -> &[f64] {
        &self.i_cap
    }
    /// Outward-positive source (nA) at the soma that closes the electrode circuit.
    pub fn electrode_return(&self) -> f64 {
        self.i_electrode
    }
    /// Internal Ca²⁺ per compartment, mM.
    pub fn cai(&self) -> impl Iterator<Item = f64> + '_ {
        self.state.iter().map(|s| s.cai)
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Poisson events this step, each on a compartment chosen by membrane area.
    fn deliver_synapses(&mut self) {
        let mean = self.syn_rate_hz * 1e-3 * self.params.dt;
        if mean.is_nan() || mean <= 0.0 {
            return;
        }
        let limit = (-mean).exp();
        let mut prod = self.uniform();
        while prod > limit {
            let u = self.uniform();
            let i = self
                .syn_cdf
                .partition_point(|&c| c <= u)
                .min(self.len() - 1);
            self.g_syn[i] += 1e-3 * self.params.syn_weight_ns;
            prod *= self.uniform();
        }
    }

    fn step_once(&mut self, w: f64) {
        let dt = self.params.dt;
        self.deliver_synapses();
        let n = self.len();
        // Branch nodes carry no membrane.
        self.diag.copy_from_slice(&self.g_axial_sum);
        self.rhs.fill(0.0);
        for i in 0..n {
            let (d, s) = (&self.dens[i], self.scale[i]);
            let c = self.state[i].currents(d, self.v[i]);
            let cap = d.cm * s / dt;
            let g = c.g * s;
            let stim = if i == 0 { self.i_stim } else { 0.0 };
            let k = self.node[i];
            self.diag[k] += cap + g + self.g_syn[i];
            self.rhs[k] = (cap + g) * self.v[i] - c.total() * s + stim;
        }
        // Hines elimination: every node comes after its parent.
        for k in (1..self.parent.len()).rev() {
            let p = self.parent[k];
            let f = self.g_axial[k] / self.diag[k];
            self.diag[p] -= f * self.g_axial[k];
            self.rhs[p] += f * self.rhs[k];
        }
        self.rhs[0] /= self.diag[0];
        for k in 1..self.parent.len() {
            self.rhs[k] = (self.rhs[k] + self.g_axial[k] * self.rhs[self.parent[k]]) / self.diag[k];
        }
        let f_syn = channels::na_fraction(0.0);
        for i in 0..n {
            let (d, s) = (&self.dens[i], self.scale[i]);
            let v_new = self.rhs[self.node[i]];
            let c = self.state[i].currents(d, v_new);
            // Synaptic current g (V − 0) as a Na⁺ part reversing at E_Na plus a K⁺ part
            // reversing at E_K.
            let (gs, f) = (self.g_syn[i], f_syn);
            self.i_cap[i] += w * d.cm * s / dt * (v_new - self.v[i]);
            self.i_na[i] += w * (c.na * s + f * gs * (v_new - E_NA));
            self.i_k[i] += w * (c.k * s + (1.0 - f) * gs * (v_new - E_K));
            self.i_ca[i] += w * c.ca * s;
            self.state[i].advance(d, v_new, c.ca, dt);
            self.v[i] = v_new;
        }
        let decay = (-dt / self.params.syn_tau).exp();
        self.g_syn.iter_mut().for_each(|g| *g *= decay);
        self.i_electrode -= w * self.i_stim;
        self.time += dt;
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Simulation for HayNeuron {
    fn step(&mut self, n: usize) {
        let w = 1.0 / n.max(1) as f64;
        for out in [
            &mut self.i_na,
            &mut self.i_k,
            &mut self.i_ca,
            &mut self.i_cap,
        ] {
            out.fill(0.0);
        }
        self.i_electrode = 0.0;
        for _ in 0..n {
            self.step_once(w);
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spikes(n: &mut HayNeuron, ms: f64) -> usize {
        let (mut count, mut above) = (0, n.v()[0] > 0.0);
        for _ in 0..(ms / n.params().dt).round() as usize {
            n.step(1);
            let up = n.v()[0] > 0.0;
            count += usize::from(up && !above);
            above = up;
        }
        count
    }

    #[test]
    fn stays_at_rest() {
        let mut n = HayNeuron::cell1();
        let v0 = n.v()[0];
        assert!((-80.0..-70.0).contains(&v0), "{v0}");
        assert_eq!(spikes(&mut n, 100.0), 0);
        assert!((n.v()[0] - v0).abs() < 0.5);
    }

    #[test]
    fn matches_neuron_rest_and_firing_rate() {
        // NEURON 9 with the ModelDB 139653 files (L5PCbiophys3, 34 °C, Δt = 0.025 ms,
        // 500 ms settle, cai0 = 1e-4 mM): rest −77.25 mV, 40 spikes in 2 s at 1.9 nA.
        let mut n = HayNeuron::cell1();
        assert!((n.v()[0] + 77.25).abs() < 0.05, "{}", n.v()[0]);
        n.i_stim = 1.9;
        assert!((spikes(&mut n, 2000.0) as i64 - 40).abs() <= 1);
    }

    #[test]
    fn synapses_let_na_in_and_k_out() {
        let mut quiet = HayNeuron::cell1();
        let mut driven = quiet.clone();
        driven.g_syn[100] = 0.01;
        quiet.step(1);
        driven.step(1);
        let (dna, dk) = (
            driven.i_na[100] - quiet.i_na[100],
            driven.i_k[100] - quiet.i_k[100],
        );
        assert!(dna < 0.0 && dk > 0.0, "{dna} {dk}");
    }

    #[test]
    fn currents_sum_to_the_electrode_current() {
        let mut n = HayNeuron::cell1();
        n.i_stim = 1.5;
        n.syn_rate_hz = 20_000.0;
        for _ in 0..50 {
            n.step(8);
            let sum: f64 = [n.i_na(), n.i_k(), n.i_ca(), n.i_cap()]
                .iter()
                .map(|a| a.iter().sum::<f64>())
                .sum();
            assert!((sum + n.electrode_return()).abs() < 1e-9, "{sum}");
        }
        assert!(n.i_ca().iter().sum::<f64>() < 0.0, "Ca²⁺ flows in");
    }

    /// `cargo test --release -p ed-knp hay_step_time -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn hay_step_time() {
        let mut n = HayNeuron::cell1();
        n.syn_rate_hz = 5000.0;
        let start = std::time::Instant::now();
        n.step(4000);
        let us = start.elapsed().as_secs_f64() * 1e6 / 4000.0;
        println!("{us:.1} µs per 0.025 ms step");
    }
}
