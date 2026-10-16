//! Multi-compartment neuron that supplies the KNP point sources
//! (Publication 1 used a NEURON simulation for this).
//!
//! This module currently holds the morphology; the cable-equation solver
//! that produces per-compartment ion currents builds on it.

mod morphology;

pub use morphology::{Compartment, Kind, Morphology};
