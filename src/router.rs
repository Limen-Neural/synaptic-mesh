//! Generic multi-channel SNN router.
//!
//! A domain-agnostic SNN router that integrates signal pulses across a bank
//! of neuromodulatory neurons to produce a sparse routing mask.
//!
//! The router is generic over channel count and expects raw signal strengths as input.

use serde::{Deserialize, Serialize};
use crate::neuromod::NeuromodNeuron;

/// Number of input channels for the default 3-channel router (backward compatible).
pub const AHL_NUM_CHANNELS: usize = 3;

/// Integration timesteps per routing decision (more → more stable).
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / `ROUTING_TIMESTEPS`) to activate a channel.
const MIN_FIRE_RATE: f32 = 0.1875;

/// Configuration for a generic channel router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Number of input/output channels.
    pub channel_count: usize,
    /// Self-affinity weight (diagonal of weight matrix).
    pub self_weight: f32,
    /// Cross-channel inhibition weight (off-diagonal).
    pub cross_weight: f32,
    /// Firing threshold for neuromodulatory neurons.
    pub threshold: f32,
    /// Passive leak rate per timestep.
    pub leak: f32,
    /// Integration timesteps per routing decision.
    pub routing_timesteps: usize,
    /// Minimum firing rate to activate a channel.
    pub min_fire_rate: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            channel_count: 3,
            self_weight: 0.9,
            cross_weight: -0.15,
            threshold: 0.22,
            leak: 0.12,
            routing_timesteps: ROUTING_TIMESTEPS,
            min_fire_rate: MIN_FIRE_RATE,
        }
    }
}

/// Sparse activation decision from the SNN router.
#[derive(Debug, Clone, Default)]
pub struct RoutingDecision {
    /// Indices of the channels that were activated.
    pub active_channels: Vec<usize>,
    /// Per-channel firing rates (for diagnostics and feedback).
    pub firing_rates: Vec<f32>,
    /// Raw input signals fed into the router.
    pub input_signals: Vec<f32>,
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

/// Generic multi-channel SNN Router.
///
/// Integrates multi-channel signals over `ROUTING_TIMESTEPS` to produce
/// a sparse activation mask. The number of channels is configurable at
/// construction time via [`RouterConfig`].
#[derive(Clone, Serialize, Deserialize)]
pub struct ChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    #[serde(default)]
    config: RouterConfig,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
}

/// Backward-compatible alias for the default 3-channel router.
///
/// Deprecated: use [`ChannelRouter`] with [`RouterConfig::default()`] instead.
pub type AhlRouter = ChannelRouter;

impl Default for ChannelRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelRouter {
    /// Create a new router with default configuration (3 channels).
    pub fn new() -> Self {
        Self::with_config(RouterConfig::default())
    }

    /// Create a new router with a custom configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.routing_timesteps` is zero.
    pub fn with_config(config: RouterConfig) -> Self {
        assert!(config.routing_timesteps > 0, "routing_timesteps must be > 0");
        let n = config.channel_count;
        let neurons = (0..n).map(|i| {
            let mut neu = NeuromodNeuron::new();
            // Strong self-affinity; weak cross-channel inhibition.
            neu.weights = vec![config.cross_weight; n];
            neu.weights[i] = config.self_weight;
            neu.threshold = config.threshold;
            neu.leak = config.leak;
            neu
        }).collect();

        Self { neurons, config, total_routes: 0 }
    }

    /// Route raw channel signals through the SNN.
    ///
    /// `signals` must have length equal to `config.channel_count`.
    pub fn route<S: AsRef<[f32]>>(&mut self, signals: S) -> Result<RoutingDecision, crate::error::MeshError> {
        let signals = signals.as_ref();
        let n = self.config.channel_count;
        if signals.len() != n {
            return Err(crate::error::MeshError::NeuronCountMismatch {
                expected: n,
                got: signals.len(),
                context: "route signals".into(),
            });
        }

        let mut spike_counts = vec![0u32; n];
        let timesteps = self.config.routing_timesteps;
        let min_rate = self.config.min_fire_rate;

        // Reset membrane potentials for a fresh routing decision.
        for neu in &mut self.neurons {
            neu.v = 0.0;
        }

        // Integrate over routing_timesteps.
        for _ in 0..timesteps {
            for (i, neu) in self.neurons.iter_mut().enumerate() {
                let stimulus: f32 = signals.iter()
                    .zip(neu.weights.iter())
                    .map(|(sig, w)| sig * w)
                    .sum();

                neu.integrate(stimulus);

                if neu.check_fire().is_some() {
                    spike_counts[i] += 1;
                }
            }
        }

        let mut firing_rates = vec![0.0f32; n];
        let mut active_channels = Vec::new();
        for i in 0..n {
            firing_rates[i] = spike_counts[i] as f32 / timesteps as f32;
            if firing_rates[i] >= min_rate {
                active_channels.push(i);
            }
        }

        self.total_routes += 1;
        Ok(RoutingDecision {
            active_channels,
            firing_rates,
            input_signals: signals.to_vec(),
        })
    }

    /// Apply feedback to adjust synaptic weights for a specific channel.
    pub fn apply_feedback(&mut self, channel_idx: usize, reward: f32) {
        let n = self.config.channel_count;
        if channel_idx >= n { return; }

        let delta = reward * 0.01; // small learning rate

        // Potentiate/Depress self-affinity
        self.neurons[channel_idx].weights[channel_idx] =
            (self.neurons[channel_idx].weights[channel_idx] + delta).clamp(0.1, 2.0);

        // Lateral inhibition adjustment
        if reward > 0.0 {
            for j in 0..n {
                if j != channel_idx {
                    self.neurons[j].weights[channel_idx] =
                        (self.neurons[j].weights[channel_idx] - delta * 0.3).clamp(-1.0, 1.5);
                }
            }
        }
    }

    /// Apply global neuromodulatory gain to all neurons.
    pub fn set_global_gain(&mut self, gain: f32) {
        for neu in &mut self.neurons {
            neu.set_gain(gain);
        }
    }

    /// Current routing weight matrix (row = neuron, col = input channel).
    pub fn weight_matrix(&self) -> Vec<Vec<f32>> {
        self.neurons.iter().map(|neu| neu.weights.clone()).collect()
    }

    /// Access the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }
}
