// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic multi-channel SNN router with neuromodulatory adaptation.
//!
//! A domain-agnostic SNN router that integrates signal pulses across a bank
//! of neuromodulatory neurons to produce a sparse routing mask.
//!
//! The router is generic over channel count and supports adaptive
//! neuromodulatory routing — channels strengthen with use (dopamine-gated)
//! and weaken when idle (use-it-or-lose-it plasticity).

use crate::neuromod::NeuromodNeuron;
use serde::{Deserialize, Serialize};

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
    #[serde(default = "default_plasticity_decay")]
    pub plasticity_decay: f32,
    /// Weight potentiation rate for active channels (dopamine-gated).
    #[serde(default = "default_plasticity_potentiate")]
    pub plasticity_potentiate: f32,
    /// Smoothing factor for active-channel weight potentiation
    /// (0.0 = no change, 1.0 = snap to amplified target). Tunable so callers
    /// can trade off adaptation speed vs. numerical stability.
    #[serde(default = "default_plasticity_speed")]
    pub plasticity_speed: f32,
    /// Fatigue accumulation rate per activation.
    #[serde(default = "default_fatigue_accumulation")]
    pub fatigue_accumulation: f32,
    /// Fatigue recovery rate per tick.
    #[serde(default = "default_fatigue_recovery")]
    pub fatigue_recovery: f32,
}

fn default_plasticity_decay() -> f32 {
    0.02
}
fn default_plasticity_potentiate() -> f32 {
    0.05
}
fn default_plasticity_speed() -> f32 {
    0.1
}
fn default_fatigue_accumulation() -> f32 {
    0.15
}
fn default_fatigue_recovery() -> f32 {
    0.05
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
            plasticity_speed: 0.1,
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
    /// Raises firing thresholds (always — even on fresh / zero-fatigue channels)
    /// and additionally amplifies the effect of accumulated fatigue.
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
    #[serde(default)]
    pub channel_fatigue: Vec<f32>,
    /// Baseline weights for plasticity decay reference.
    #[serde(default)]
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
        assert!(
            config.routing_timesteps > 0,
            "routing_timesteps must be > 0"
        );
        let n = config.channel_count;
        let neurons: Vec<NeuromodNeuron> = (0..n)
            .map(|i| {
                let mut neu = NeuromodNeuron::new();
                // Strong self-affinity; weak cross-channel inhibition.
                neu.weights = vec![config.cross_weight; n];
                neu.weights[i] = config.self_weight;
                neu.threshold = config.threshold;
                neu.leak = config.leak;
                neu
            })
            .collect();

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
    ///
    /// Backward-compatible thin wrapper around [`route_modulated`]. The error
    /// context reported on a signal-length mismatch is `"route signals"`,
    /// matching the original pre-neuromodulation API — callers using this
    /// public method see the same error message they did before, even though
    /// the implementation now delegates to `route_modulated` internally.
    pub fn route<S: AsRef<[f32]>>(
        &mut self,
        signals: S,
    ) -> Result<RoutingDecision, crate::error::MeshError> {
        self.route_modulated_with_context(signals, &NeuromodState::balanced(), "route signals")
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
        self.route_modulated_with_context(signals, mods, "route_modulated signals")
    }

    /// Internal routing implementation. The `error_context` argument is the
    /// string used in the `NeuronCountMismatch` error so each public entry
    /// point can report the method the caller actually invoked.
    fn route_modulated_with_context<S: AsRef<[f32]>>(
        &mut self,
        signals: S,
        mods: &NeuromodState,
        error_context: &str,
    ) -> Result<RoutingDecision, crate::error::MeshError> {
        let signals = signals.as_ref();
        let n = self.config.channel_count;
        if signals.len() != n {
            return Err(crate::error::MeshError::NeuronCountMismatch {
                expected: n,
                got: signals.len(),
                context: error_context.into(),
            });
        }

        // Self-heal: keep neuromod state vectors aligned with the current
        // channel count (e.g. after deserializing an older router).
        self.ensure_neuromod_state_synced();
        let (effective_thresholds, effective_leaks) = self.compute_effective_params(mods);

        // Reset membrane potentials for a fresh routing decision.
        for neu in &mut self.neurons {
            neu.v = 0.0;
        }

        let timesteps = self.config.routing_timesteps;
        let min_rate = self.config.min_fire_rate;
        let spike_counts =
            self.integrate_signals(signals, &effective_thresholds, &effective_leaks, timesteps);

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

    /// Run the per-timestep integration loop and return spike counts per channel.
    ///
    /// For each timestep, every neuron computes its stimulus from the signal
    /// vector, applies the serotonin-modulated leak, integrates the stimulus,
    /// and checks against the dopamine/cortisol-modulated threshold. Neuron
    /// membrane potentials (`v`) are updated in place; per-channel spike counts
    /// are incremented on each fire.
    fn integrate_signals(
        &mut self,
        signals: &[f32],
        effective_thresholds: &[f32],
        effective_leaks: &[f32],
        timesteps: usize,
    ) -> Vec<u32> {
        let n = self.config.channel_count;
        let mut spike_counts = vec![0u32; n];
        for _ in 0..timesteps {
            // Iterate only up to n (channel_count) so we never index beyond
            // spike_counts, effective_thresholds, or effective_leaks. If
            // neurons.len() > n (malformed state), the extra neurons are
            // skipped. If neurons.len() < n, those channels produce no spikes.
            for (i, neu) in self.neurons.iter_mut().enumerate().take(n) {
                debug_assert_eq!(
                    neu.weights.len(),
                    signals.len(),
                    "Neuron weights length mismatch"
                );
                let stimulus: f32 = signals
                    .iter()
                    .zip(neu.weights.iter())
                    .map(|(sig, w)| sig * w)
                    .sum();
                neu.leak = effective_leaks[i];
                neu.integrate(stimulus);
                neu.threshold = effective_thresholds[i];
                if neu.check_fire().is_some() {
                    spike_counts[i] += 1;
                }
            }
        }
        spike_counts
    }

    /// Lazily (re)initialize `channel_fatigue` and `baseline_weights` so that
    /// their lengths match the current channel count. Called at the top of
    /// `route_modulated` so that deserializing older router states (where
    /// these fields default to empty) cannot trigger out-of-bounds indexing.
    ///
    /// Validates BOTH the outer length AND the inner row length of
    /// `baseline_weights`. A deserialized value like `[[], [], []]` would
    /// pass an outer-length-only check and then panic in `apply_plasticity`
    /// at `self.baseline_weights[i][j]`. If any row is the wrong size we
    /// rebuild the whole 2D table from the current neuron weights.
    fn ensure_neuromod_state_synced(&mut self) {
        let n = self.config.channel_count;
        if self.channel_fatigue.len() != n {
            self.channel_fatigue.resize(n, 0.0);
        }
        let baseline_ok = self.baseline_weights.len() == n
            && self.baseline_weights.iter().all(|row| row.len() == n);
        if !baseline_ok {
            self.baseline_weights = self.neurons.iter().map(|neu| neu.weights.clone()).collect();
        }
    }

    /// Per-channel effective thresholds and leaks under neuromodulation.
    ///
    /// - Cortisol: baseline stress component (always raises threshold, even on
    ///   fresh channels) + fatigue amplification (further raises it on
    ///   fatigued channels).
    /// - Dopamine: lowers threshold (conductance).
    /// - Serotonin: raises leak (faster decay).
    fn compute_effective_params(&self, mods: &NeuromodState) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.channel_count;
        let mut thresholds = vec![0.0f32; n];
        let mut leaks = vec![0.0f32; n];
        for i in 0..n {
            let baseline_stress = 1.0 + mods.cortisol * 0.5;
            let fatigue_amplification = 1.0 + mods.cortisol * self.channel_fatigue[i];
            let fatigue_factor = baseline_stress * fatigue_amplification;
            let dopamine_factor = 1.0 - mods.dopamine * 0.5;
            thresholds[i] =
                (self.config.threshold * fatigue_factor * dopamine_factor).clamp(0.05, 2.0);
            leaks[i] = (self.config.leak * (1.0 + mods.serotonin)).clamp(0.0, 1.0);
        }
        (thresholds, leaks)
    }

    /// Apply use-it-or-lose-it plasticity.
    ///
    /// - Active channels: strengthen (dopamine-gated), accumulate fatigue
    /// - Inactive channels: decay toward baseline weights, recover fatigue
    ///
    /// Weights are clamped to a unified range that covers both the
    /// self-affinity range ([0.1, 2.0]) and the cross-channel range
    /// ([-1.0, 1.5]) used by `apply_feedback`. This prevents the
    /// use-it-or-lose-it decay from drifting into a regime where the
    /// downstream `apply_feedback` clamp would suddenly snap a weight.
    fn apply_plasticity(&mut self, active_channels: &[usize], mods: &NeuromodState) {
        let n = self.config.channel_count;
        let decay = self.config.plasticity_decay;
        let potentiate = self.config.plasticity_potentiate;
        let plasticity_speed = self.config.plasticity_speed;
        let fatigue_acc = self.config.fatigue_accumulation;
        let fatigue_rec = self.config.fatigue_recovery;

        for i in 0..n {
            if active_channels.contains(&i) {
                // Active channel: strengthen (dopamine-gated), accumulate fatigue.
                let strengthen = potentiate * (1.0 + mods.dopamine);
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    // Move toward amplified baseline.
                    let target = baseline * (1.0 + strengthen);
                    self.neurons[i].weights[j] =
                        (current + (target - current) * plasticity_speed).clamp(-1.5, 2.0);
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] + fatigue_acc).min(1.0);
            } else {
                // Inactive channel: decay toward baseline, recover fatigue.
                for j in 0..n {
                    let baseline = self.baseline_weights[i][j];
                    let current = self.neurons[i].weights[j];
                    self.neurons[i].weights[j] =
                        (current + (baseline - current) * decay).clamp(-1.5, 2.0);
                }
                self.channel_fatigue[i] = (self.channel_fatigue[i] - fatigue_rec).max(0.0);
            }
        }
    }

    /// Apply feedback to adjust synaptic weights for a specific channel.
    ///
    /// Self-heals `channel_fatigue` and `baseline_weights` before any indexing,
    /// so calling `apply_feedback` on a freshly deserialized router (where
    /// the lazy repair in `route_modulated` has not yet run) is safe. See
    /// `ensure_neuromod_state_synced` for the exact shape check.
    pub fn apply_feedback(&mut self, channel_idx: usize, reward: f32) {
        self.ensure_neuromod_state_synced();
        let n = self.config.channel_count;
        // Guard both the direct channel_idx access AND the loop that indexes
        // neurons[j] for j in 0..n. A malformed deserialized state could have
        // config.channel_count > neurons.len(), which would panic in the loop.
        if channel_idx >= n || n > self.neurons.len() {
            return;
        }

        let delta = reward * 0.01;

        self.neurons[channel_idx].weights[channel_idx] =
            (self.neurons[channel_idx].weights[channel_idx] + delta).clamp(0.1, 2.0);

        if reward > 0.0 {
            for j in 0..n {
                if j != channel_idx {
                    self.neurons[j].weights[channel_idx] =
                        (self.neurons[j].weights[channel_idx] - delta * 0.3).clamp(-1.0, 1.5);
                }
            }
        }

        // Keep plasticity baseline in sync with feedback-driven learning.
        self.sync_baseline_after_feedback(channel_idx, reward);
    }

    /// Sync `baseline_weights` for the rows affected by a feedback call so that
    /// the new feedback-adjusted weights become the reference point for future
    /// use-it-or-lose-it decay. Skipped if `baseline_weights` hasn't been
    /// initialized yet (e.g. before the first `route_modulated` call).
    fn sync_baseline_after_feedback(&mut self, channel_idx: usize, reward: f32) {
        let n = self.config.channel_count;
        // Belt-and-suspenders: ensure_neuromod_state_synced() in apply_feedback
        // should have already rebuilt baseline_weights if rows were malformed,
        // but guard anyway to avoid a panic if the call ordering invariant
        // is ever violated.
        debug_assert!(
            self.baseline_weights.len() == n
                && self.baseline_weights.iter().all(|row| row.len() == n),
            "baseline_weights shape mismatch — ensure_neuromod_state_synced should have rebuilt"
        );
        if self.baseline_weights.len() != n
            || self.baseline_weights.iter().any(|row| row.len() != n)
        {
            return;
        }
        self.baseline_weights[channel_idx][channel_idx] =
            self.neurons[channel_idx].weights[channel_idx];
        if reward > 0.0 {
            for j in 0..n {
                if j != channel_idx {
                    self.baseline_weights[j][channel_idx] = self.neurons[j].weights[channel_idx];
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
