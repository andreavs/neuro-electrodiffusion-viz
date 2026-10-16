//! Many reduced neurons that both release ions into, and respond to, the extracellular space.
//!
//! Each neuron has two compartments, a soma (Hodgkin-Huxley Na⁺ and K⁺ channels) and a lumped
//! passive dendrite, joined by an axial conductance. Unlike [`crate::neuron::Neuron`], the
//! coupling to the tissue goes both ways: before each `Population::step` (#54) the caller writes
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
//! Time stepping: within one call to `Population::step` (#54) the concentrations, reversal
//! potentials and pump currents are frozen (they change on the scale of seconds). Each neuron
//! then takes its own internal steps: backward Euler for the two potentials (a 2×2 solve
//! with the conductances frozen at the start of the step) and Rush-Larsen for the gates. The
//! internal step adapts so that V moves by about [`PopulationParams::v_tol`] per step, between
//! `h_min` while spiking and `h_max` at rest or in depolarisation block, so a quiet neuron
//! costs one or two updates per call. Both schemes are stable at any step size, but frozen
//! gates over a long step damp out spike initiation, so where the Na⁺ feedback is strong (near
//! threshold) the step is also held to a quarter of the Na⁺ activation time. Gating functions
//! come from a table in V (linear interpolation, 0.05 mV spacing), as in NEURON.
//!
//! Outputs, per compartment: mean Na⁺, K⁺, X⁻ (Cl⁻) and capacitive currents over the last
//! step, in nA, positive out of the cell. Axial currents cancel, so the four sum to zero over
//! each neuron (to round-off), as a KNP solver with closed boundaries requires.

mod params;
// The stepper that reads the table lands with `Population` (#54).
#[allow(dead_code)]
mod rates;

pub use params::{Calibration, PopulationParams, PUMP_KM_K, PUMP_KM_NA};
