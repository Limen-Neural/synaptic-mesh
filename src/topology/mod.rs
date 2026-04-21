//! Topology module — network graph construction and wiring.
//!
//! Provides the [`SynapticGraph`] adjacency structure and deterministic
//! topology generators inspired by classic network science models.

mod generators;
mod graph;
mod wiring_rules;

pub use generators::{
    generate_layered, generate_random, generate_scale_free, generate_small_world,
};
pub use graph::SynapticGraph;
pub use wiring_rules::{apply_dale_polarity, assign_delays};
