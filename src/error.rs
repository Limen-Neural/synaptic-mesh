//! Error types for `synaptic-mesh`.
//!
//! Follows the `corinth-canal` pattern of a single unified error enum
//! using `thiserror` for ergonomic `Display` and `From` implementations.

use std::fmt;

/// Unified error type for synaptic-mesh operations.
#[derive(Debug, Clone)]
pub enum MeshError {
    /// A required parameter was out of range or invalid.
    InvalidConfig(String),

    /// Neuron count mismatch between components.
    NeuronCountMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Attempted to access a neuron or synapse that doesn't exist.
    IndexOutOfBounds { index: usize, max: usize },

    /// Topology generation failed (e.g., impossible wiring constraints).
    TopologyError(String),

    /// Delay buffer overflow or misconfiguration.
    DelayError(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            MeshError::NeuronCountMismatch {
                expected,
                got,
                context,
            } => write!(
                f,
                "neuron count mismatch in {context}: expected {expected}, got {got}"
            ),
            MeshError::IndexOutOfBounds { index, max } => {
                write!(f, "index {index} out of bounds (max {max})")
            }
            MeshError::TopologyError(msg) => write!(f, "topology error: {msg}"),
            MeshError::DelayError(msg) => write!(f, "delay error: {msg}"),
        }
    }
}

impl std::error::Error for MeshError {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, MeshError>;
