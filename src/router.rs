//! SNN-based Anti-Hallucination Layer Router.
//!
//! A generic multi-channel SNN router that integration signal pulses
//! across a bank of neuromodulatory neurons to produce a sparse routing mask.
//!
//! This version is domain-agnostic and expects raw signal strengths as input.

use serde::{Deserialize, Serialize};
use crate::neuromod::NeuromodNeuron;

/// Number of input channels (default 3).
pub const AHL_NUM_CHANNELS: usize = 3;

/// Integration timesteps per routing decision (more → more stable).
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / `ROUTING_TIMESTEPS`) to activate a channel.
const MIN_FIRE_RATE: f32 = 0.1875;

/// Sparse activation decision from the SNN router.
#[derive(Debug, Clone, Default)]
pub struct RoutingDecision {
    /// Indices of the channels that were activated.
    pub active_channels: Vec<usize>,
    /// Per-channel firing rates (for diagnostics and feedback).
    pub firing_rates: [f32; AHL_NUM_CHANNELS],
    /// Raw input signals fed into the router.
    pub input_signals: [f32; AHL_NUM_CHANNELS],
}

impl RoutingDecision {
    pub fn is_active(&self, channel: usize) -> bool {
        self.active_channels.contains(&channel)
    }

    /// True when no channel was activated.
    pub fn is_empty(&self) -> bool {
        self.active_channels.is_empty()
    }
}

/// Anti-Hallucination Layer SNN Router.
///
/// Contains `AHL_NUM_CHANNELS` neurons that integrate multi-channel
/// signals over `ROUTING_TIMESTEPS` to produce a sparse activation mask.
#[derive(Clone, Serialize, Deserialize)]
pub struct AhlRouter {
    neurons: Vec<NeuromodNeuron>,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
}

impl Default for AhlRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AhlRouter {
    /// Create a new router with default synaptic weights.
    pub fn new() -> Self {
        let neurons = (0..AHL_NUM_CHANNELS).map(|i| {
            let mut n = NeuromodNeuron::new();
            // Strong self-affinity; weak cross-channel inhibition.
            n.weights = vec![-0.15; AHL_NUM_CHANNELS];
            n.weights[i] = 0.9;
            n.threshold = 0.22;
            n.leak = 0.12;
            n
        }).collect();

        Self { neurons, total_routes: 0 }
    }

    /// Route raw channel signals through the SNN.
    ///
    /// `signals` must have length `AHL_NUM_CHANNELS`.
    pub fn route(&mut self, signals: [f32; AHL_NUM_CHANNELS]) -> RoutingDecision {
        let mut spike_counts = [0u32; AHL_NUM_CHANNELS];

        // Reset membrane potentials for a fresh routing decision.
        for n in &mut self.neurons {
            n.v = 0.0;
        }

        // Integrate over ROUTING_TIMESTEPS.
        for _ in 0..ROUTING_TIMESTEPS {
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                let stimulus: f32 = signals.iter()
                    .zip(neuron.weights.iter())
                    .map(|(sig, w)| sig * w)
                    .sum();
                
                neuron.integrate(stimulus);
                
                if neuron.check_fire().is_some() {
                    spike_counts[i] += 1;
                }
            }
        }

        let mut firing_rates = [0.0f32; AHL_NUM_CHANNELS];
        let mut active_channels = Vec::new();
        for i in 0..AHL_NUM_CHANNELS {
            firing_rates[i] = spike_counts[i] as f32 / ROUTING_TIMESTEPS as f32;
            if firing_rates[i] >= MIN_FIRE_RATE {
                active_channels.push(i);
            }
        }

        self.total_routes += 1;
        RoutingDecision { 
            active_channels, 
            firing_rates, 
            input_signals: signals 
        }
    }

    /// Apply feedback to adjust synaptic weights for a specific channel.
    pub fn apply_feedback(&mut self, channel_idx: usize, reward: f32) {
        if channel_idx >= AHL_NUM_CHANNELS { return; }
        
        let delta = reward * 0.01; // small learning rate

        // Potentiate/Depress self-affinity
        self.neurons[channel_idx].weights[channel_idx] =
            (self.neurons[channel_idx].weights[channel_idx] + delta).clamp(0.1, 2.0);

        // Lateral inhibition adjustment
        if reward > 0.0 {
            for j in 0..AHL_NUM_CHANNELS {
                if j != channel_idx {
                    self.neurons[j].weights[channel_idx] =
                        (self.neurons[j].weights[channel_idx] - delta * 0.3).clamp(-1.0, 1.5);
                }
            }
        }
    }

    /// Apply global neuromodulatory gain to all neurons.
    pub fn set_global_gain(&mut self, gain: f32) {
        for n in &mut self.neurons {
            n.set_gain(gain);
        }
    }

    /// Current routing weight matrix (row = neuron, col = input channel).
    pub fn weight_matrix(&self) -> [[f32; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS] {
        let mut m = [[0.0; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS];
        for (i, n) in self.neurons.iter().enumerate() {
            for (j, &w) in n.weights.iter().enumerate() {
                if j < AHL_NUM_CHANNELS {
                    m[i][j] = w;
                }
            }
        }
        m
    }
}
