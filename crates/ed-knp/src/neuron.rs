//! Multi-compartment neuron that supplies the KNP point sources.
//!
//! Replaces the NEURON simulation of Publication 1: a tree of cylindrical
//! compartments ([`Morphology`]) obeys the cable equation, with
//! Hodgkin-Huxley channels in the soma and axon and a passive leak in the
//! dendrites (both from [`ed_core::membrane`]). The coupling to the
//! extracellular space is one-way, as in the paper: concentrations seen by
//! the membrane are fixed.
//!
//! Each step (backward Euler, Δt = 0.025 ms by default as in the paper) solves
//!
//! `C_j (V_j' − V_j)/Δt + I_ch,j(V_j') = Σ_neighbours g_jl (V_l' − V_j') + I_stim,j`
//!
//! with the channel current linearised around `V_j` for frozen gates (which is
//! exact for conductance-based currents), using the O(N) Hines elimination on
//! the tree. Gates are then advanced with Rush-Larsen at the new potential.
//!
//! Inputs: a current injected into the soma ([`Neuron::i_stim`]) and
//! excitatory Poisson input at [`Neuron::syn_rate_hz`], placed by dendritic
//! area; each event adds `syn_weight_ns` to an exponential Na⁺ synapse
//! (τ = `passive.syn_tau`). A clone shares the RNG state, so it receives the
//! same input.
//!
//! Outputs, per compartment, in nA with **positive = positive charge leaving
//! the cell** (the Publication 1 convention): [`Neuron::i_na`], [`Neuron::i_k`],
//! [`Neuron::i_x`] and [`Neuron::i_cap`], plus [`Neuron::centres`] in µm. They
//! are averages over the internal steps of the last [`Simulation::step`] call,
//! so a KNP solver with a longer time step gets the charge actually moved;
//! within a step, ionic currents are evaluated at the new potential with the
//! gates frozen at their start-of-step values. Since axial currents cancel
//! pairwise, the sum of all four over all compartments equals the electrode
//! current, `−`[`Neuron::electrode_return`], to round-off, which keeps the
//! KNP solver electroneutral. A KNP solver turns them into sources with
//! `f_k = I_k / (F z_k α)` (Publication 1, eq 38).
//!
//! Without ion pumps, the resting membrane is not ion-neutral: it carries a
//! steady inward Na⁺ and equal outward K⁺ current (≈ 0.4 nA in total for the
//! default cell), so extracellular K⁺ slowly builds up even without input.
//!
//! Ion assignment:
//! - Leak currents are ion specific (g_L^Na, g_L^K, g_L^Cl of the membrane
//!   model), so no non-specific leak needs splitting.
//! - Synapses open Na⁺ channels (E_syn = E_Na), so synaptic current is Na⁺.
//! - Cl⁻ currents are reported as the generic anion X⁻ of the KNP model.
//! - The somatic current electrode is not a transmembrane current and is not
//!   in any of the arrays. [`Neuron::electrode_return`] gives the
//!   source the KNP solver must add for its return path to keep the total
//!   zero, e.g. as Cl⁻ released by a Ag/AgCl ground at the soma.

mod morphology;
#[cfg(test)]
mod tests;

pub use morphology::{Compartment, Kind, Morphology};

use ed_core::membrane::{hh, IonConcentrations, MembraneParams, MembranePatches};
use ed_core::Simulation;

/// Fixed concentrations seen by the neuron (mM): the Publication 1 ECS
/// baseline (Na⁺ 150, K⁺ 3, X⁻ 155.8) and Publication 2 intracellular values.
pub const TISSUE_CONCENTRATIONS: IonConcentrations = IonConcentrations {
    na_i: 12.0,
    na_e: 150.0,
    k_i: 125.0,
    k_e: 3.0,
    cl_i: 137.0,
    cl_e: 155.8,
};

/// Soma and axon membrane: the KNP-EMI Hodgkin-Huxley channels, with the leak
/// scaled to the classic HH total of 0.3 mS/cm² (Na⁺/K⁺ ratio kept). With the
/// KNP-EMI leak of 1 mS/cm², the dendrite-loaded soma fires tonically only in
/// a narrow band of input currents (around 2 nA) and gives single spikes elsewhere.
pub const SOMA_AXON: MembraneParams = MembraneParams {
    g_leak_na: 0.06,
    g_leak_k: 0.24,
    ..MembraneParams::HODGKIN_HUXLEY
};

/// Dendritic membrane: the KNP-EMI passive leak scaled down 20× (Na⁺/K⁺ ratio
/// kept), giving a specific membrane resistance of 20 kΩ·cm² and τ_m = 20 ms,
/// typical of cortical dendrites.
pub const DENDRITE: MembraneParams = MembraneParams {
    g_leak_na: 0.01,
    g_leak_k: 0.04,
    ..MembraneParams::PASSIVE
};

/// Parameters of a [`Neuron`].
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronParams {
    /// Time step, ms.
    pub dt: f64,
    /// Axial resistivity, Ω·cm.
    pub ra: f64,
    /// Membrane of the soma and axon.
    pub active: MembraneParams,
    /// Membrane of the dendrites.
    pub passive: MembraneParams,
    /// Fixed intra- and extracellular concentrations.
    pub concentrations: IonConcentrations,
    /// Peak conductance added by one synaptic event, nS (≥ 0).
    pub syn_weight_ns: f64,
    /// RNG seed for the Poisson input.
    pub seed: u64,
    /// Simulated time to relax to rest during construction (with Δt = 0.5 ms), ms.
    pub settle_ms: f64,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            dt: 0.025,
            ra: 150.0,
            active: SOMA_AXON,
            passive: DENDRITE,
            concentrations: TISSUE_CONCENTRATIONS,
            syn_weight_ns: 1.0,
            seed: 1,
            settle_ms: 200.0,
        }
    }
}

/// A multi-compartment neuron. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct Neuron {
    params: NeuronParams,
    /// Total rate of excitatory synaptic events over all dendrites, Hz,
    /// placed in proportion to membrane area (uniform density, as in
    /// Publication 1). Must be finite and non-negative; can be changed at any
    /// time. With the default cell and weight, about 2 kHz gives the paper's
    /// ~5 spikes/s, 10–20 kHz gives ~90–110 spikes/s, and above ~30 kHz the
    /// soma goes into depolarisation block.
    pub syn_rate_hz: f64,
    /// Current injected into the root (soma), nA (depolarising positive). Can
    /// be changed at any time.
    pub i_stim: f64,

    morph: Morphology,
    centres: Vec<[f64; 3]>,
    parent: Vec<usize>,
    /// Membrane area in units of 1e5 µm² = 1e-3 cm², which converts densities
    /// (µA/cm², mS/cm², µF/cm²) into nA, µS and nF.
    scale: Vec<f64>,
    /// Axial conductance to the parent, µS (0 for the root).
    g_axial: Vec<f64>,
    /// Sum of axial conductances to all neighbours, µS.
    g_axial_sum: Vec<f64>,
    /// (is_active, index within that population) per compartment.
    slot: Vec<(bool, usize)>,
    active: MembranePatches,
    passive: MembranePatches,
    /// Cumulative dendritic area fraction per compartment (1.0 at the last
    /// dendrite, 0 contribution from non-dendrites), for placing events.
    syn_cdf: Vec<f64>,
    rng: u64,
    time: f64,

    v: Vec<f64>,
    i_na: Vec<f64>,
    i_k: Vec<f64>,
    i_x: Vec<f64>,
    i_cap: Vec<f64>,
    i_electrode: f64,
    // Hines scratch.
    diag: Vec<f64>,
    rhs: Vec<f64>,
}

impl Neuron {
    /// Build a neuron and relax it to rest for `params.settle_ms`. Time starts
    /// at 0 after settling.
    pub fn new(morph: Morphology, params: NeuronParams) -> Self {
        let n = morph.len();
        if let Err(e) = morph.validate() {
            panic!("invalid morphology: {e}");
        }
        assert!(
            params.dt > 0.0 && params.ra > 0.0,
            "dt and ra must be positive"
        );
        let comps = &morph.compartments;
        let mut parent = vec![0; n];
        let mut g_axial = vec![0.0; n];
        let mut g_axial_sum = vec![0.0; n];
        for i in 1..n {
            let (p, g) = (
                comps[i].parent.unwrap(),
                morph.coupling_conductance(i, params.ra),
            );
            parent[i] = p;
            g_axial[i] = g;
            g_axial_sum[i] += g;
            g_axial_sum[p] += g;
        }
        let mut slot = Vec::with_capacity(n);
        let (mut n_act, mut n_pas) = (0, 0);
        for c in comps {
            if c.kind.is_active() {
                slot.push((true, n_act));
                n_act += 1;
            } else {
                slot.push((false, n_pas));
                n_pas += 1;
            }
        }
        let mut acc = 0.0;
        let mut syn_cdf: Vec<f64> = comps
            .iter()
            .map(|c| {
                acc += if c.kind == Kind::Dendrite {
                    c.area()
                } else {
                    0.0
                };
                acc
            })
            .collect();
        syn_cdf
            .iter_mut()
            .for_each(|x| *x /= acc.max(f64::MIN_POSITIVE));
        let conc = params.concentrations;
        let mut neuron = Self {
            i_stim: 0.0,
            syn_rate_hz: 0.0,
            syn_cdf,
            rng: splitmix64(params.seed) | 1,
            centres: comps.iter().map(Compartment::centre).collect(),
            scale: comps.iter().map(|c| c.area() * 1e-5).collect(),
            parent,
            g_axial,
            g_axial_sum,
            active: MembranePatches::new(n_act, params.active, conc),
            passive: MembranePatches::new(n_pas, params.passive, conc),
            v: vec![0.0; n],
            slot,
            time: 0.0,
            i_na: vec![0.0; n],
            i_k: vec![0.0; n],
            i_x: vec![0.0; n],
            i_cap: vec![0.0; n],
            i_electrode: 0.0,
            diag: vec![0.0; n],
            rhs: vec![0.0; n],
            morph,
            params,
        };
        for i in 0..n {
            let (act, j) = neuron.slot[i];
            neuron.v[i] = if act { &neuron.active } else { &neuron.passive }.phi_m[j];
        }
        // Backward Euler and Rush-Larsen are stable at large steps and share
        // the fixed point of the fine scheme, so relax with a coarse step.
        let dt = std::mem::replace(&mut neuron.params.dt, 0.5);
        neuron.step((neuron.params.settle_ms / 0.5).round() as usize);
        neuron.params.dt = dt;
        neuron.time = 0.0;
        neuron
    }

    /// The default pyramidal cell with default parameters.
    pub fn pyramidal() -> Self {
        Self::new(Morphology::pyramidal(), NeuronParams::default())
    }

    pub fn params(&self) -> &NeuronParams {
        &self.params
    }
    pub fn morphology(&self) -> &Morphology {
        &self.morph
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
    /// X⁻ (Cl⁻) current per compartment over the last step, nA, outward
    /// positive.
    pub fn i_x(&self) -> &[f64] {
        &self.i_x
    }
    /// Outward-positive source (nA) at the root compartment's centre that
    /// closes the electrode circuit: `−i_stim`, averaged like the currents.
    pub fn electrode_return(&self) -> f64 {
        self.i_electrode
    }
    /// Capacitive current per compartment over the last step, nA, outward positive.
    pub fn i_cap(&self) -> &[f64] {
        &self.i_cap
    }

    /// Uniform random number in [0, 1) (xorshift64*).
    fn uniform(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draw this step's Poisson number of synaptic events, place each on a
    /// dendritic compartment with probability proportional to its area, and
    /// add its conductance to that compartment's exponential synapse.
    fn deliver_synapses(&mut self, t: f64) {
        let mean = self.syn_rate_hz * 1e-3 * self.params.dt;
        debug_assert!(
            mean.is_nan() || mean < 700.0,
            "synaptic rate {} Hz too high",
            self.syn_rate_hz
        );
        if mean.is_nan() || mean <= 0.0 || !matches!(self.syn_cdf.last(), Some(&x) if x > 0.0) {
            return;
        }
        // Knuth's method; the mean per step is small (≈ 0.5 at 20 kHz).
        let limit = (-mean).exp();
        let mut prod = self.uniform();
        while prod > limit {
            let u = self.uniform();
            let i = self.syn_cdf.partition_point(|&c| c <= u);
            let (act, j) = self.slot[i];
            debug_assert!(!act, "synaptic event placed on a non-dendrite");
            // nS → mS/cm²: 1e-6 mS / (A · 1e-8 cm²) = 1e-3 / scale.
            let dg = self.params.syn_weight_ns * 1e-3 / self.scale[i];
            let pop = if act {
                &mut self.active
            } else {
                &mut self.passive
            };
            let g = pop.synaptic_conductance(j, t) + dg;
            pop.set_synapse(j, g, t);
            prod *= self.uniform();
        }
    }

    /// One internal step; adds `w` times its currents to the outputs.
    fn step_once(&mut self, w: f64) {
        let (t, dt) = (self.time, self.params.dt);
        self.deliver_synapses(t);
        self.active.compute_currents(t);
        self.passive.compute_currents(t);
        let n = self.len();
        for i in 0..n {
            let (act, j) = self.slot[i];
            let pop = if act { &self.active } else { &self.passive };
            let s = self.scale[i];
            let cap = pop.params.c_m * s / dt;
            let g = pop.g_ch[j] * s;
            let stim = if i == 0 { self.i_stim } else { 0.0 };
            self.diag[i] = cap + g + self.g_axial_sum[i];
            self.rhs[i] = (cap + g) * self.v[i] - pop.i_ch[j] * s + stim;
        }
        // Hines elimination: children have larger indices than parents.
        for i in (1..n).rev() {
            let p = self.parent[i];
            let f = -self.g_axial[i] / self.diag[i];
            self.diag[p] += f * self.g_axial[i];
            self.rhs[p] -= f * self.rhs[i];
        }
        self.rhs[0] /= self.diag[0];
        for i in 1..n {
            self.rhs[i] = (self.rhs[i] + self.g_axial[i] * self.rhs[self.parent[i]]) / self.diag[i];
        }
        // rhs now holds V'. Ionic currents at V' with frozen gates.
        for i in 0..n {
            let (act, j) = self.slot[i];
            let pop = if act {
                &mut self.active
            } else {
                &mut self.passive
            };
            pop.phi_m[j] = self.rhs[i];
        }
        self.active.compute_currents(t);
        self.passive.compute_currents(t);
        for i in 0..n {
            let (act, j) = self.slot[i];
            let pop = if act { &self.active } else { &self.passive };
            let s = self.scale[i];
            let v_new = self.rhs[i];
            self.i_cap[i] += w * pop.params.c_m * s / dt * (v_new - self.v[i]);
            self.i_na[i] += w * pop.i_na[j] * s;
            self.i_k[i] += w * pop.i_k[j] * s;
            self.i_x[i] += w * pop.i_cl[j] * s;
            self.v[i] = v_new;
        }
        self.i_electrode -= w * self.i_stim;
        update_gates(&mut self.active, dt);
        self.time += dt;
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Rush-Larsen update of the HH gates at the patches' current potential.
fn update_gates(p: &mut MembranePatches, dt: f64) {
    if !p.params.hodgkin_huxley {
        return;
    }
    let vr = p.params.v_rest_hh;
    for j in 0..p.len() {
        let v = p.phi_m[j];
        p.m[j] = hh::rates_m(v, vr).rush_larsen(p.m[j], dt);
        p.h[j] = hh::rates_h(v, vr).rush_larsen(p.h[j], dt);
        p.n[j] = hh::rates_n(v, vr).rush_larsen(p.n[j], dt);
    }
}

impl Simulation for Neuron {
    fn step(&mut self, n: usize) {
        let w = 1.0 / n.max(1) as f64;
        for out in [
            &mut self.i_na,
            &mut self.i_k,
            &mut self.i_x,
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
