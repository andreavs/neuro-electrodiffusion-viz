//! Cell-scale KNP-EMI electrodiffusion with explicit membranes.
//!
//! See docs/ARCHITECTURE.md for the model and the GitHub issues for the plan.

pub mod geometry;
pub mod params;
pub mod precond;
pub mod solver;

pub use geometry::{AxonBundle, AxonExtent, MembraneFace, VoxelGeometry, ECS};
pub use params::{EmiParams, StepStats, SPECIES, UA_PER_CM2_TO_FLUX};
pub use solver::EmiSolver;
