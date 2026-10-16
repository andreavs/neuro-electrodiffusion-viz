//! The detailed layer-5b pyramidal cell of Hay et al. (2011), ModelDB 139653, used in
//! Publication 1: morphology `cell1.asc` with the `L5PCbiophys3` channels.

pub mod channels;
mod morphology;

pub use morphology::{HayMorphology, Region};
