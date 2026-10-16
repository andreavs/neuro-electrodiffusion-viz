//! What [`crate::tissue::Tissue`] needs from a neuron model: membrane currents per
//! compartment and where they enter the extracellular space.

use crate::hay::HayNeuron;
use crate::neuron::{Morphology, Neuron};
use ed_core::Simulation;

/// A multi-compartment neuron that releases ions. Currents are in nA, positive out of the
/// cell, averaged over the internal steps of the last [`Simulation::step`] call, and,
/// with [`electrode_return`](Cell::electrode_return) at the root, sum to zero.
pub trait Cell: Simulation {
    /// Internal time step, ms.
    fn dt(&self) -> f64;
    fn morphology(&self) -> &Morphology;
    /// Compartment centres relative to the soma, µm.
    fn centres(&self) -> &[[f64; 3]];
    /// Membrane potential per compartment, mV; index 0 is the soma.
    fn v(&self) -> &[f64];
    /// Ionic currents in the KNP species order Na⁺, K⁺, X⁻, Ca²⁺; a species the model
    /// does not carry gives an empty slice.
    fn ion_currents(&self) -> [&[f64]; 4];
    fn i_cap(&self) -> &[f64];
    /// Source at the root that closes the electrode circuit (carried by X⁻).
    fn electrode_return(&self) -> f64;
    /// Current injected into the soma, nA.
    fn set_stim(&mut self, i_na: f64);
    /// Total excitatory synaptic rate, Hz.
    fn set_syn_rate(&mut self, hz: f64);
    fn len(&self) -> usize {
        self.v().len()
    }
    fn is_empty(&self) -> bool {
        self.v().is_empty()
    }
}

impl Cell for Neuron {
    fn dt(&self) -> f64 {
        self.params().dt
    }
    fn morphology(&self) -> &Morphology {
        Neuron::morphology(self)
    }
    fn centres(&self) -> &[[f64; 3]] {
        Neuron::centres(self)
    }
    fn v(&self) -> &[f64] {
        Neuron::v(self)
    }
    fn ion_currents(&self) -> [&[f64]; 4] {
        [self.i_na(), self.i_k(), self.i_x(), &[]]
    }
    fn i_cap(&self) -> &[f64] {
        Neuron::i_cap(self)
    }
    fn electrode_return(&self) -> f64 {
        Neuron::electrode_return(self)
    }
    fn set_stim(&mut self, i_na: f64) {
        self.i_stim = i_na;
    }
    fn set_syn_rate(&mut self, hz: f64) {
        self.syn_rate_hz = hz;
    }
}

impl Cell for HayNeuron {
    fn dt(&self) -> f64 {
        self.params().dt
    }
    fn morphology(&self) -> &Morphology {
        HayNeuron::morphology(self)
    }
    fn centres(&self) -> &[[f64; 3]] {
        HayNeuron::centres(self)
    }
    fn v(&self) -> &[f64] {
        HayNeuron::v(self)
    }
    fn ion_currents(&self) -> [&[f64]; 4] {
        [self.i_na(), self.i_k(), &[], self.i_ca()]
    }
    fn i_cap(&self) -> &[f64] {
        HayNeuron::i_cap(self)
    }
    fn electrode_return(&self) -> f64 {
        HayNeuron::electrode_return(self)
    }
    fn set_stim(&mut self, i_na: f64) {
        self.i_stim = i_na;
    }
    fn set_syn_rate(&mut self, hz: f64) {
        self.syn_rate_hz = hz;
    }
}

impl Simulation for Box<dyn Cell> {
    fn step(&mut self, n: usize) {
        (**self).step(n)
    }
    fn time_ms(&self) -> f64 {
        (**self).time_ms()
    }
}

impl Cell for Box<dyn Cell> {
    fn dt(&self) -> f64 {
        (**self).dt()
    }
    fn morphology(&self) -> &Morphology {
        (**self).morphology()
    }
    fn centres(&self) -> &[[f64; 3]] {
        (**self).centres()
    }
    fn v(&self) -> &[f64] {
        (**self).v()
    }
    fn ion_currents(&self) -> [&[f64]; 4] {
        (**self).ion_currents()
    }
    fn i_cap(&self) -> &[f64] {
        (**self).i_cap()
    }
    fn electrode_return(&self) -> f64 {
        (**self).electrode_return()
    }
    fn set_stim(&mut self, i_na: f64) {
        (**self).set_stim(i_na)
    }
    fn set_syn_rate(&mut self, hz: f64) {
        (**self).set_syn_rate(hz)
    }
}
