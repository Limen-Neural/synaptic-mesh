//! Core types for `synaptic-mesh`.
//!
//! Shared scalar types, measurement units, and configuration structs used
//! across the topology, delay, and mesh modules.

use serde::{Deserialize, Serialize};

// ── Scalar aliases ────────────────────────────────────────────────────────────

/// Unique identifier for a neuron within a mesh.
pub type NeuronId = u32;

/// Axonal propagation delay in discrete simulation ticks.
/// A delay of 0 means same-tick delivery (instantaneous).
pub type DelayTicks = u16;

// ── Polarity ──────────────────────────────────────────────────────────────────

/// Synaptic polarity following Dale's principle: a neuron's outgoing
/// synapses are either all excitatory or all inhibitory.
///
/// # References
///
/// Dale, H. H. (1935). *Pharmacology and Nerve-endings.* Proceedings of
/// the Royal Society of Medicine, 28(3), 319–332.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarity {
    /// Positive synaptic weight → depolarises the postsynaptic neuron.
    Excitatory,
    /// Negative synaptic weight → hyperpolarises the postsynaptic neuron.
    Inhibitory,
}

impl Polarity {
    /// Sign multiplier: +1.0 for excitatory, −1.0 for inhibitory.
    pub fn sign(self) -> f32 {
        match self {
            Self::Excitatory => 1.0,
            Self::Inhibitory => -1.0,
        }
    }
}

impl Default for Polarity {
    fn default() -> Self {
        Self::Excitatory
    }
}

// ── Synapse descriptor ────────────────────────────────────────────────────────

/// A fully-described synaptic connection with weight, delay, and polarity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynapseDescriptor {
    /// Source neuron.
    pub source: NeuronId,
    /// Target neuron.
    pub target: NeuronId,
    /// Absolute synaptic weight (always ≥ 0; sign comes from polarity).
    pub weight: f32,
    /// Axonal propagation delay in ticks.
    pub delay: DelayTicks,
    /// Excitatory or inhibitory.
    pub polarity: Polarity,
}

impl SynapseDescriptor {
    /// Effective signed weight: `|weight| × polarity.sign()`.
    pub fn effective_weight(&self) -> f32 {
        self.weight * self.polarity.sign()
    }
}

// ── Topology parameters ──────────────────────────────────────────────────────

/// Configuration for network topology generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Number of neurons in the network.
    pub neuron_count: usize,
    /// Fraction of neurons that are inhibitory (Dale's law).
    /// Typical cortical value: ~0.20 (80% excitatory, 20% inhibitory).
    pub inhibitory_fraction: f32,
    /// Maximum axonal delay in ticks.
    pub max_delay: DelayTicks,
    /// Minimum absolute weight to retain (sparsity pruning threshold).
    pub sparsity_threshold: f32,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            neuron_count: 2048,
            inhibitory_fraction: 0.20,
            max_delay: 20,
            sparsity_threshold: 0.01,
        }
    }
}

// ── Connection probability models ─────────────────────────────────────────────

/// Strategy for determining connection probability between neuron pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionModel {
    /// Erdős–Rényi: each pair connected independently with probability `p`.
    Uniform { p: f32 },
    /// Distance-dependent: probability decays with Euclidean distance.
    /// `p(d) = p_max × exp(−d / lambda)`.
    DistanceDependent { p_max: f32, lambda: f32 },
    /// Watts–Strogatz small-world: each neuron connected to `k` nearest
    /// neighbours in a ring, then each edge rewired with probability `beta`.
    SmallWorld { k: usize, beta: f32 },
    /// Barabási–Albert preferential attachment: start with `m0` connected
    /// nodes, each new node attaches to `m` existing nodes proportional
    /// to their degree.
    ScaleFree { m0: usize, m: usize },
    /// Feed-forward layered: neurons arranged in layers, each layer
    /// fully connected to the next with given probability.
    Layered {
        layer_sizes: Vec<usize>,
        inter_layer_p: f32,
    },
}

impl Default for ConnectionModel {
    fn default() -> Self {
        Self::Uniform { p: 0.05 }
    }
}

// ── Delay model ───────────────────────────────────────────────────────────────

/// Strategy for assigning axonal delays to synapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelayModel {
    /// All synapses have the same fixed delay.
    Fixed { delay: DelayTicks },
    /// Delay is proportional to Euclidean distance between neurons.
    /// `delay = clamp(round(distance / speed), min_delay, max_delay)`.
    DistanceProportional {
        speed: f32,
        min_delay: DelayTicks,
        max_delay: DelayTicks,
    },
    /// Delay drawn uniformly from `[min_delay, max_delay]`.
    UniformRandom {
        min_delay: DelayTicks,
        max_delay: DelayTicks,
    },
}

impl Default for DelayModel {
    fn default() -> Self {
        Self::Fixed { delay: 1 }
    }
}
