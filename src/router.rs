//! Generic multi-channel SNN router with neuromodulatory adaptation.
//!
//! A domain-agnostic SNN router that integrates signal pulses across a bank
//! of neuromodulatory neurons to produce a sparse routing mask.
//!
//! The router is generic over channel count and supports adaptive
//! neuromodulatory routing — channels strengthen with use (dopamine-gated)
//! and weaken when idle (use-it-or-lose-it plasticity).

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
    /// Weight decay rate for inactive channels (use-it-or-lose-it).
    pub plasticity_decay: f32,
    /// Weight potentiation rate for active channels (dopamine-gated).
    pub plasticity_potentiate: f32,
    /// Fatigue accumulation rate per activation.
    pub fatigue_accumulation: f32,
    /// Fatigue recovery rate per tick.
    pub fatigue_recovery: f32,
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
            plasticity_decay: 0.02,
            plasticity_potentiate: 0.05,
            fatigue_accumulation: 0.15,
            fatigue_recovery: 0.05,
        }
    }
}

/// Neuromodulatory state for adaptive routing.
///
/// Cortisol (stress) increases resistance — channels become harder to activate.
/// Dopamine (reward) increases conductance — channels become easier to activate.
/// Serotonin (patience) reduces persistence — faster decay of activation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NeuromodState {
    /// Stress level (0.0 = calm, 1.0 = max stress).
    /// Raises firing thresholds and amplifies fatigue.
    pub cortisol: f32,
    /// Reward level (0.0 = no reward, 1.0 = high reward).
    /// Lowers thresholds, strengthens active synapses, counteracts fatigue.
    pub dopamine: f32,
    /// Patience/risk-aversion level (0.0 = impulsive, 1.0 = patient).
    /// Increases leak/decay rate, making activations less persistent.
    pub serotonin: f32,
}

impl NeuromodState {
    /// Create a balanced neuromodulatory state (no modulation).
    pub fn balanced() -> Self {
        Self {
            cortisol: 0.0,
            dopamine: 0.0,
            serotonin: 0.0,
        }
    }

    /// Create a stressed state (high cortisol).
    pub fn stressed() -> Self {
        Self {
            cortisol: 0.8,
            dopamine: 0.0,
            serotonin: 0.0,
        }
    }

    /// Create a rewarded state (high dopamine).
    pub fn rewarded() -> Self {
        Self {
            cortisol: 0.0,
            dopamine: 0.8,
            serotonin: 0.0,
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
///
/// Supports adaptive neuromodulatory routing via [`route_modulated`]:
/// - Channels strengthen with use (dopamine-gated potentiation)
/// - Channels weaken when idle (use-it-or-lose-it decay)
/// - Fatigue accumulates with activation, cortisol amplifies it
/// - The router naturally seeks the least-resistance pathway
#[derive(Clone, Serialize, Deserialize)]
pub struct ChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    #[serde(default)]
    config: RouterConfig,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
    /// Per-channel fatigue (0.0 = fresh, 1.0 = fully exhausted).
    pub channel_fatigue: Vec<f32>,
    /// Baseline weights for plasticity decay reference.
    baseline_weights: Vec<Vec<f32>>,
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
        let neurons: Vec<NeuromodNeuron> = (0..n).map(|i| {
            let mut neu = NeuromodNeuron::new();
            // Strong self-affinity; weak cross-channel inhibition.
            neu.weights = vec![config.cross_weight; n];
            neu.weights[i] = config.self_weight;
            neu.threshold = config.threshold;
            neu.leak = config.leak;
            neu
        }).collect();

        let baseline_weights = neurons.iter().map(|neu| neu.weights.clone()).collect();

        Self {
            neurons,
            config,
            total_routes: 0,
            channel_fatigue: vec![0.0; n],
            baseline_weights,
        }
    }

    /// Route raw channel signals through the SNN (non-modulated).
    ///
    /// `signals` must have length equal to `config.channel_count`.
    pub fn route<S: AsRef<[f32]>>(&mut self, signals: S) -> Result<RoutingDecision, crate::error::MeshError> {
        self.route_modulated(signals, &NeuromodState::balanced())
    }

    /// Route with neuromodulatory modulation.
    ///
    /// Seeks the least-resistance pathway by dynamically adjusting thresholds
    /// and applying use-it-or-lose-it plasticity:
    /// - Cortisol raises effective thresholds (resistance)
    /// - Dopamine lowers thresholds and strengthens active channels (conductance)
    /// - Serotonin increases leak (reduces persistence)
    /// - Inactive channels decay toward baseline weights
    /// - Active channels potentiate (dopamine-gated)
    pub fn route_modulated<S: AsRef<[f32]>>(
        &mut self,
        signals: S,
        mods: &NeuromodState,
    ) -> Result<RoutingDecision, crate::error::MeshError> {
        let signals = signals.as_ref();
        let n = self.config.channel_count;
        if signals.len() != n {
            return Err(crate::error::MeshError::NeuronCountMismatch {
                expected: n,
                got: signals.len(),
                context: "route_modulated signals".into(),
            });
        }

        let timesteps = self.config.routing_timesteps;
        let min_rate = self.config.min_fire_rate;

        // Compute effective thresholds per channel.
        let mut effective_thresholds = vec![0.0f32; n];
        let mut effective_leaks = vec![0.0f32; n];
        for i in 0..n {
            // Cortisol amplifies fatigue → higher threshold.
            let fatigue_factor = 1.0 + mods.cortisol * self.channel_fatigue[i];
            // Dopamine reduces threshold → lower resistance.
            let dopamine_factor = 1.0 - mods.dopamine * 0.5;
            effective_thresholds[i] = (self.config.threshold * fatigue_factor * dopamine_factor)
                .clamp(0.05, 2.0);
            // Serotonin increases leak → faster decay.
            effective_leaks[i] = (self.config.leak * (1.0 + mods.serotonin)).clamp(0.0, 1.0);
        }

        // Reset membrane potentials for a fresh routing decision.
        for neu in &mut self.neurons {
            neu.v = 0.0;
        }

        let mut spike_counts = vec![0u32; n];

        // Integrate over routing_timesteps.
        for _ in 0..timesteps {
            for (i, neu) in self.neurons.iter_mut().enumerate() {
                let stimulus: f32 = signals.iter()
                    .zip(neu.weights.iter())
                    .map(|(sig, w)| sig * w)
                    .sum();

                // Apply serotonin-modulated leak.
                neu.leak = effective_leaks[i];
                neu.integrate(stimulus);
                neu.threshold = effective_thresholds[i];

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

        // Apply use-it-or-lose-it plasticity.
        self.apply_plasticity(&active_channels, mods);

        self.total_routes += 1;
        Ok(RoutingDecision {
            active_channels,
            firing_rates,
            input_signals: signals.to_vec(),
        })
    }

    /// Apply use-it-or-lose-it plasticity.
    ///
    /// - Active channels: strengthen (dopamine-gated), accumulate fatigue
    /// - Inactive channels: decay toward baseline weights, recover fatigue
    fn apply_plasticity(&mut self, active_channels: &[usize], mods: &NeuromodState) {
        let n = self.config.channel_count;
        let decay = self.config.plasticity_decay;
        let potentiate = self.config.plasticity_potentiate;
        let fatigue_acc = self.config.fatigue_accumulation;
        let fatigue_rec = self.config.fatigue_recovery;

        let active_set: std::collections::HashSet<usize> = active_channels.iter().copied().collect();

        for i in 0..n {
            if active_set.contains(&i) {
                // Active channel: strengthen (dopamine-gated), accumulate fatigue.
                let strengthen = potentiate * (1.0 + mods.dopamine);
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    // Move toward amplified baseline.
                    let target = baseline * (1.0 + strengthen);
                    self.neurons[i].weights[j] = current + (target - current) * 0.1;
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] + fatigue_acc).min(1.0);
            } else {
                // Inactive channel: decay toward baseline, recover fatigue.
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    self.neurons[i].weights[j] = current + (baseline - current) * decay;
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] - fatigue_rec).max(0.0);
            }
        }
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

    /// Access per-channel fatigue levels.
    pub fn fatigue(&self) -> &[f32] {
        &self.channel_fatigue
    }
}
