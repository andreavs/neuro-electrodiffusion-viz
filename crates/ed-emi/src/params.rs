//! Parameters, units and species shared by the KNP-EMI solver.

use ed_core::constants::F;
use ed_core::ions::{IonSpecies, CL, K, NA};
use ed_core::linalg::CgStats;
use ed_core::membrane::{IonConcentrations, MembraneParams};

/// Species order used for every per-species array: Na⁺, K⁺, Cl⁻.
pub const SPECIES: [IonSpecies; 3] = [NA, K, CL];

/// Converts a current density in µA/cm² to a charge flux in mM·µm/ms (per unit valence).
/// `I = F Σ z J`, with 1 mM·µm/ms = 10⁻³ mol/(m² s) and 1 A/m² = 100 µA/cm².
pub const UA_PER_CM2_TO_FLUX: f64 = 10.0 / F;

/// Solver parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmiParams {
    /// PDE time step Δt, ms.
    pub dt: f64,
    /// Membrane ODE substeps per PDE step.
    pub substeps: usize,
    pub membrane: MembraneParams,
    /// Initial intra- and extracellular concentrations, mM.
    pub initial: IonConcentrations,
    /// Tolerance of the potential solve, relative to the size of the step's update (the
    /// leftover residual is corrected in the next step).
    pub potential_rtol: f64,
}

impl Default for EmiParams {
    /// Publication 2: Δt = 0.1 ms with 25 ODE substeps, Hodgkin-Huxley membrane.
    fn default() -> Self {
        Self {
            dt: 0.1,
            substeps: 25,
            membrane: MembraneParams::HODGKIN_HUXLEY,
            initial: IonConcentrations::KNP_EMI,
            potential_rtol: 1e-3,
        }
    }
}

/// Linear-solver work of the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    pub potential: CgStats,
    /// Largest iteration count of the three concentration solves.
    pub concentration_iterations: usize,
}
