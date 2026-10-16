//! Cell-scale KNP-EMI electrodiffusion with explicit membranes.
//!
//! See docs/ARCHITECTURE.md for the model and the GitHub issues for the plan.

pub mod geometry;
pub mod solver;

pub use geometry::{AxonBundle, Geometry, MembraneFace};
pub use solver::{EmiParams, KnpEmi, StepStats, SPECIES};
