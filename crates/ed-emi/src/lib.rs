//! Cell-scale KNP-EMI electrodiffusion with explicit membranes.
//!
//! See docs/ARCHITECTURE.md for the model and the GitHub issues for the plan.

pub mod geometry;
pub mod mesh;

pub use geometry::{AxonBundle, Geometry, MembraneFace};
