// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic multi-channel SNN router with neuromodulatory adaptation.
//!
//! A domain-agnostic SNN router that integrates signal pulses across a bank
//! of neuromodulatory neurons to produce a sparse routing mask.
//!
//! The router is generic over channel count and supports adaptive
//! neuromodulatory routing — channels strengthen with use (dopamine-gated)
//! and weaken when idle (use-it-or-lose-it plasticity).
//!
//! This module is **optional and self-contained**: it neither uses nor is
//! used by [`SynapticMesh`](crate::mesh::SynapticMesh). Reach for it when you
//! need to pick a few active channels out of many inputs; ignore it entirely
//! if you only need wiring, topology, and delays.

use serde::de::{Deserializer, Error as DeError};
use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};

/// Integration timesteps per routing decision (more → more stable).
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / `ROUTING_TIMESTEPS`) to activate a channel.
const MIN_FIRE_RATE: f32 = 0.1875;

/// Largest channel count [`ChannelRouter`] will allocate.
///
/// The router stores a dense `N × N` weight matrix plus a baseline copy, so
/// this cap is checked in [`RouterConfig::validate`] *before* any of those
/// vectors are allocated.
pub const MAX_ROUTER_CHANNELS: usize = 1024;

/// Neuromodulatory Integrative Fixed-threshold (NIF) neuron.
///
/// This is a **router-internal integration primitive** for [`ChannelRouter`],
/// not a general-purpose neuron model — canonical neuron models (LIF,
/// Izhikevich, Hodgkin-Huxley, GIF, FitzHugh-Nagumo, Lapicque) live in the
/// separate `neuromod` crate, which `synaptic-wiring` intentionally does not
/// depend on. See the crate-level docs for the full boundary rationale.
///
/// $V_{t+1} = V_t + (G \cdot I_{syn}) - \lambda(V_t - V_{rest})$
/// where $G$ is the modulation gain and $\lambda$ is the leak rate.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct NeuromodNeuron {
    /// Current membrane potential.
    pub v: f32,
    /// Resting membrane potential.
    pub v_rest: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Passive leak rate per timestep.
    pub leak: f32,
    /// Firing threshold.
    pub threshold: f32,

    /// Neuromodulatory gain (scales incoming stimulus).
    pub gain: f32,

    /// Synaptic weights — one per input channel.
    pub weights: Vec<f32>,
    /// Whether the neuron fired in the last timestep.
    pub last_spike: bool,
}

impl Default for NeuromodNeuron {
    fn default() -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            leak: 0.12,
            threshold: 0.25,
            gain: 1.0,
            weights: Vec::new(),
            last_spike: false,
        }
    }
}

impl NeuromodNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance neuron dynamics by one timestep.
    ///
    /// The `stimulus` is scaled by the neuron's current `gain`.
    pub fn integrate(&mut self, stimulus: f32) {
        // Apply modulated integration
        self.v += stimulus * self.gain;
        // Apply leak towards resting potential
        self.v -= (self.v - self.v_rest) * self.leak;
    }

    /// Check if the neuron spikes. Resets V on fire.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.v >= self.threshold {
            let peak = self.v;
            self.v = self.v_reset;
            self.last_spike = true;
            return Some(peak);
        }
        self.last_spike = false;
        None
    }

    /// Update the modulation gain.
    pub fn set_gain(&mut self, new_gain: f32) {
        self.gain = new_gain;
    }
}

/// Configuration for a generic channel router.
///
/// Direct struct literals can hold out-of-range values; they are rejected by
/// [`RouterConfig::validate`], [`ChannelRouter::try_with_config`], and by
/// deserialization of both this type and [`ChannelRouter`].
///
/// # Valid ranges
///
/// | Field | Constraint |
/// |-------|------------|
/// | `channel_count` | `1..=`[`MAX_ROUTER_CHANNELS`] |
/// | `routing_timesteps` | `> 0` |
/// | `self_weight`, `cross_weight`, `threshold` | finite; **signed weights are allowed** |
/// | `leak`, `min_fire_rate`, `plasticity_decay`, `plasticity_speed`, `fatigue_accumulation`, `fatigue_recovery` | finite, in `0.0..=1.0` |
/// | `plasticity_potentiate` | finite, `>= 0` (scale factor, not a probability) |
///
/// `cross_weight` is typically negative (lateral inhibition). That sign is
/// allowed and intended; it is not required. Positive self-weights are the
/// usual self-affinity pattern, but a signed self-weight is also accepted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RouterConfig {
    /// Number of input/output channels (`1..=`[`MAX_ROUTER_CHANNELS`]).
    pub channel_count: usize,
    /// Self-affinity weight (diagonal of the dense channel matrix).
    ///
    /// Must be finite. Signed values are allowed.
    pub self_weight: f32,
    /// Cross-channel weight (off-diagonal).
    ///
    /// Must be finite. Signed values are allowed; **negative values are the
    /// intended lateral-inhibition pattern** and are not rejected.
    pub cross_weight: f32,
    /// Firing threshold for neuromodulatory neurons.
    ///
    /// Must be finite. Typically positive; zero and negative values are
    /// accepted here (modulated routing later clamps the *effective*
    /// threshold).
    pub threshold: f32,
    /// Passive leak rate per timestep.
    ///
    /// Fraction of `(V − V_rest)` removed each tick. Valid range: `0.0..=1.0`.
    /// `0.0` = no leak; `1.0` = membrane snaps to rest in one tick.
    pub leak: f32,
    /// Integration timesteps per routing decision.
    ///
    /// Must be `> 0`. More timesteps → more stable firing-rate estimates.
    pub routing_timesteps: usize,
    /// Minimum firing rate to activate a channel.
    ///
    /// Fraction of `routing_timesteps` that must spike. Valid range: `0.0..=1.0`.
    pub min_fire_rate: f32,
    /// Weight decay rate for inactive channels (use-it-or-lose-it).
    ///
    /// Mix toward baseline: `current + (baseline − current) * decay`.
    /// Valid range: `0.0..=1.0`.
    pub plasticity_decay: f32,
    /// Weight potentiation rate for active channels (dopamine-gated).
    ///
    /// Non-negative finite scale factor, not a probability. The amplified
    /// target is `baseline * (1 + potentiate * (1 + dopamine))`.
    pub plasticity_potentiate: f32,
    /// Smoothing factor for active-channel weight potentiation
    /// (`0.0` = no change, `1.0` = snap to amplified target). Tunable so
    /// callers can trade off adaptation speed vs. numerical stability.
    /// Valid range: `0.0..=1.0`.
    pub plasticity_speed: f32,
    /// Fatigue accumulation per activation.
    ///
    /// Added to per-channel fatigue (itself in `0.0..=1.0`). Valid range:
    /// `0.0..=1.0`.
    pub fatigue_accumulation: f32,
    /// Fatigue recovery per idle routing decision.
    ///
    /// Subtracted from per-channel fatigue. Valid range: `0.0..=1.0`.
    pub fatigue_recovery: f32,
}

#[derive(Deserialize)]
struct RawRouterConfig {
    channel_count: usize,
    self_weight: f32,
    cross_weight: f32,
    threshold: f32,
    leak: f32,
    routing_timesteps: usize,
    min_fire_rate: f32,
    #[serde(default = "default_plasticity_decay")]
    plasticity_decay: f32,
    #[serde(default = "default_plasticity_potentiate")]
    plasticity_potentiate: f32,
    #[serde(default = "default_plasticity_speed")]
    plasticity_speed: f32,
    #[serde(default = "default_fatigue_accumulation")]
    fatigue_accumulation: f32,
    #[serde(default = "default_fatigue_recovery")]
    fatigue_recovery: f32,
}

impl RawRouterConfig {
    fn into_config(self) -> Result<RouterConfig> {
        let config = RouterConfig {
            channel_count: self.channel_count,
            self_weight: self.self_weight,
            cross_weight: self.cross_weight,
            threshold: self.threshold,
            leak: self.leak,
            routing_timesteps: self.routing_timesteps,
            min_fire_rate: self.min_fire_rate,
            plasticity_decay: self.plasticity_decay,
            plasticity_potentiate: self.plasticity_potentiate,
            plasticity_speed: self.plasticity_speed,
            fatigue_accumulation: self.fatigue_accumulation,
            fatigue_recovery: self.fatigue_recovery,
        };
        config.validate()?;
        Ok(config)
    }
}

impl<'de> Deserialize<'de> for RouterConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawRouterConfig::deserialize(deserializer)?
            .into_config()
            .map_err(DeError::custom)
    }
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

impl RouterConfig {
    /// Check that every field is in its documented range.
    ///
    /// This is the single validation path used by
    /// [`ChannelRouter::try_with_config`], [`ChannelRouter::with_config`],
    /// and by `Deserialize` for both [`RouterConfig`] and [`ChannelRouter`].
    /// It does not allocate.
    pub fn validate(&self) -> Result<()> {
        if self.channel_count == 0 || self.channel_count > MAX_ROUTER_CHANNELS {
            return Err(MeshError::invalid_router_config(
                "channel_count",
                format!(
                    "must be in 1..={MAX_ROUTER_CHANNELS}, got {}",
                    self.channel_count
                ),
            ));
        }
        if self.routing_timesteps == 0 {
            return Err(MeshError::invalid_router_config(
                "routing_timesteps",
                "must be > 0",
            ));
        }
        require_finite("self_weight", self.self_weight)?;
        require_finite("cross_weight", self.cross_weight)?;
        require_finite("threshold", self.threshold)?;
        require_unit_interval("leak", self.leak)?;
        require_unit_interval("min_fire_rate", self.min_fire_rate)?;
        require_unit_interval("plasticity_decay", self.plasticity_decay)?;
        require_non_negative("plasticity_potentiate", self.plasticity_potentiate)?;
        require_unit_interval("plasticity_speed", self.plasticity_speed)?;
        require_unit_interval("fatigue_accumulation", self.fatigue_accumulation)?;
        require_unit_interval("fatigue_recovery", self.fatigue_recovery)?;
        Ok(())
    }
}

fn require_finite(field: &'static str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite, got {value}"),
        ))
    }
}

fn require_unit_interval(field: &'static str, value: f32) -> Result<()> {
    require_finite(field, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite and in 0.0..=1.0, got {value}"),
        ))
    }
}

fn require_non_negative(field: &'static str, value: f32) -> Result<()> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(MeshError::invalid_router_config(
            field,
            format!("must be finite and >= 0, got {value}"),
        ))
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
/// Supports adaptive neuromodulatory routing via [`ChannelRouter::route_modulated`]:
/// - Channels strengthen with use (dopamine-gated potentiation)
/// - Channels weaken when idle (use-it-or-lose-it decay)
/// - Fatigue accumulates with activation, cortisol amplifies it
/// - The router naturally seeks the least-resistance pathway
///
/// Deserialization re-runs [`RouterConfig::validate`] and requires neuron,
/// weight, fatigue, and baseline-weight vector shapes to match
/// `channel_count`. Missing `config` / `channel_fatigue` / `baseline_weights`
/// (legacy snapshots) are filled from the neuron bank rather than rejected.
#[derive(Clone, Debug, Serialize)]
pub struct ChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    config: RouterConfig,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
    /// Per-channel fatigue (0.0 = fresh, 1.0 = fully exhausted).
    pub channel_fatigue: Vec<f32>,
    /// Baseline weights for plasticity decay reference.
    baseline_weights: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct RawChannelRouter {
    neurons: Vec<NeuromodNeuron>,
    #[serde(default)]
    config: Option<RouterConfig>,
    #[serde(default)]
    total_routes: u64,
    #[serde(default)]
    channel_fatigue: Option<Vec<f32>>,
    #[serde(default)]
    baseline_weights: Option<Vec<Vec<f32>>>,
}

impl RawChannelRouter {
    fn into_router(self) -> Result<ChannelRouter> {
        let n_neurons = self.neurons.len();
        let config = match self.config {
            Some(config) => config,
            None => RouterConfig {
                channel_count: n_neurons,
                ..RouterConfig::default()
            },
        };
        // Present configs were already validated by `RouterConfig`'s
        // `Deserialize`. Re-run the same path so a `Raw` built in tests, or a
        // future constructor that skips serde, cannot bypass it.
        config.validate()?;

        let n = config.channel_count;
        let channel_fatigue = match self.channel_fatigue {
            Some(fatigue) => fatigue,
            None => vec![0.0; n],
        };
        let baseline_weights = match self.baseline_weights {
            Some(weights) => weights,
            None => self.neurons.iter().map(|neu| neu.weights.clone()).collect(),
        };
        validate_router_vectors(n, &self.neurons, &channel_fatigue, &baseline_weights)?;
        Ok(ChannelRouter {
            neurons: self.neurons,
            config,
            total_routes: self.total_routes,
            channel_fatigue,
            baseline_weights,
        })
    }
}

/// Require neuron / fatigue / baseline tables to be an `n × n` (or length-`n`)
/// layout with finite entries. Called after config validation so a huge
/// `channel_count` is rejected before this walks caller-provided vectors.
fn validate_router_vectors(
    n: usize,
    neurons: &[NeuromodNeuron],
    channel_fatigue: &[f32],
    baseline_weights: &[Vec<f32>],
) -> Result<()> {
    if neurons.len() != n {
        return Err(MeshError::invalid_router_config(
            "neurons",
            format!("length {} does not match channel_count {n}", neurons.len()),
        ));
    }
    for (i, neu) in neurons.iter().enumerate() {
        if neu.weights.len() != n {
            return Err(MeshError::invalid_router_config(
                "weights",
                format!(
                    "neuron {i} weights length {} does not match channel_count {n}",
                    neu.weights.len()
                ),
            ));
        }
        if neu.weights.iter().any(|w| !w.is_finite()) {
            return Err(MeshError::invalid_router_config(
                "weights",
                format!("neuron {i} weights contain a non-finite value"),
            ));
        }
    }
    if channel_fatigue.len() != n {
        return Err(MeshError::invalid_router_config(
            "channel_fatigue",
            format!(
                "length {} does not match channel_count {n}",
                channel_fatigue.len()
            ),
        ));
    }
    for (i, &fatigue) in channel_fatigue.iter().enumerate() {
        if !fatigue.is_finite() || !(0.0..=1.0).contains(&fatigue) {
            return Err(MeshError::invalid_router_config(
                "channel_fatigue",
                format!("index {i} must be finite and in 0.0..=1.0, got {fatigue}"),
            ));
        }
    }
    if baseline_weights.len() != n {
        return Err(MeshError::invalid_router_config(
            "baseline_weights",
            format!(
                "length {} does not match channel_count {n}",
                baseline_weights.len()
            ),
        ));
    }
    for (i, row) in baseline_weights.iter().enumerate() {
        if row.len() != n {
            return Err(MeshError::invalid_router_config(
                "baseline_weights",
                format!(
                    "row {i} length {} does not match channel_count {n}",
                    row.len()
                ),
            ));
        }
        if row.iter().any(|w| !w.is_finite()) {
            return Err(MeshError::invalid_router_config(
                "baseline_weights",
                format!("row {i} contains a non-finite value"),
            ));
        }
    }
    Ok(())
}

impl<'de> Deserialize<'de> for ChannelRouter {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawChannelRouter::deserialize(deserializer)?
            .into_router()
            .map_err(DeError::custom)
    }
}

impl Default for ChannelRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelRouter {
    /// Create a new router with default configuration (3 channels).
    pub fn new() -> Self {
        Self::try_with_config(RouterConfig::default()).expect("default RouterConfig is valid")
    }

    /// Create a new router with a custom configuration.
    ///
    /// Invalid configs panic. Prefer [`ChannelRouter::try_with_config`] when
    /// the caller needs a recoverable error.
    ///
    /// # Panics
    ///
    /// Panics if [`RouterConfig::validate`] fails (zero `routing_timesteps`,
    /// out-of-range channel count, non-finite parameters, or rates outside
    /// their documented interval).
    pub fn with_config(config: RouterConfig) -> Self {
        Self::try_with_config(config).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible counterpart of [`ChannelRouter::with_config`].
    ///
    /// Validates `config` before allocating the dense weight tables, so an
    /// oversized `channel_count` cannot exhaust memory on the way to an error.
    pub fn try_with_config(config: RouterConfig) -> Result<Self> {
        config.validate()?;
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

        Ok(Self {
            neurons,
            config,
            total_routes: 0,
            channel_fatigue: vec![0.0; n],
            baseline_weights,
        })
    }

    /// Route raw channel signals through the SNN (non-modulated).
    ///
    /// `signals` must have length equal to `config.channel_count`.
    ///
    /// Backward-compatible thin wrapper around [`ChannelRouter::route_modulated`]. The error
    /// context reported on a signal-length mismatch is `"route signals"`,
    /// matching the original pre-neuromodulation API — callers using this
    /// public method see the same error message they did before, even though
    /// the implementation now delegates to `route_modulated` internally.
    pub fn route<S: AsRef<[f32]>>(&mut self, signals: S) -> Result<RoutingDecision> {
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
    ) -> Result<RoutingDecision> {
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
    ) -> Result<RoutingDecision> {
        let signals = signals.as_ref();
        let n = self.config.channel_count;
        if signals.len() != n {
            return Err(MeshError::NeuronCountMismatch {
                expected: n,
                got: signals.len(),
                context: error_context.into(),
            });
        }

        // Self-heal: keep neuromod state vectors aligned with the current
        // channel count if a caller mutated the public `channel_fatigue`
        // field (serde restore already rejects inconsistent shapes).
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

    /// Lazily (re)initialize `channel_fatigue`, `baseline_weights`, and
    /// individual neuron weight vectors so that their lengths match the
    /// current channel count. Called at the top of `route_modulated` so
    /// that mutating the public `channel_fatigue` field (or any other
    /// post-construction length drift) cannot trigger out-of-bounds
    /// indexing. Deserialization already rejects inconsistent shapes.
    ///
    /// Repairs three layers of state:
    /// 1. `channel_fatigue` — resized to `n` (zero-filled).
    /// 2. Each neuron's `weights` vector — truncated or zero-padded to `n`
    ///    so that `integrate_signals`, `apply_plasticity`, and
    ///    `apply_feedback` can safely index `neurons[i].weights[j]`.
    /// 3. `baseline_weights` — rebuilt from the (now-repaired) neuron
    ///    weights if any row is the wrong size.
    fn ensure_neuromod_state_synced(&mut self) {
        let n = self.config.channel_count;

        // 1. Repair channel_fatigue length.
        if self.channel_fatigue.len() != n {
            self.channel_fatigue.resize(n, 0.0);
        }

        // 2. Repair individual neuron weight vectors.
        //    A deserialized neuron may have weights.len() != n (e.g.
        //    serialized with an older channel_count). Truncate or
        //    zero-pad each to exactly n.
        //
        //    Zero-padding gives new channels weight 0.0, which differs
        //    from the constructor's self_weight/cross_weight pattern.
        //    This degrades routing on new channels until weights are
        //    explicitly set — acceptable as a self-heal path (the
        //    alternative is a panic). Callers who need proper weights
        //    after a channel_count change should reconstruct via
        //    with_config rather than relying on this repair.
        let mut weights_repaired = false;
        for neu in &mut self.neurons {
            if neu.weights.len() != n {
                weights_repaired = true;
                neu.weights.resize(n, 0.0);
            }
        }

        // 3. Repair baseline_weights (rebuild from neuron weights if
        //    any row is the wrong size, or if weights were repaired
        //    in step 2).
        let baseline_ok = !weights_repaired
            && self.baseline_weights.len() == n
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
        // Guard against malformed deserialized state: check both outer lengths
        // AND inner row lengths of neurons and baseline_weights, matching the
        // belt-and-suspenders pattern in sync_baseline_after_feedback.
        debug_assert!(
            self.neurons.len() >= n
                && self.neurons.iter().all(|neu| neu.weights.len() >= n)
                && self.baseline_weights.len() >= n
                && self.baseline_weights.iter().all(|row| row.len() >= n),
            "apply_plasticity invariant violation — ensure_neuromod_state_synced should have rebuilt"
        );
        if n > self.neurons.len()
            || self.neurons.iter().any(|neu| neu.weights.len() < n)
            || self.baseline_weights.len() < n
            || self.baseline_weights.iter().any(|row| row.len() < n)
        {
            return;
        }
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
        // Guard all indexing: channel_idx bounds, neuron vector length,
        // and individual neuron weight vector lengths. A malformed
        // deserialized state could have short weight rows even when
        // neurons.len() >= n.
        debug_assert!(
            channel_idx < n
                && self.neurons.len() >= n
                && self.neurons.iter().all(|neu| neu.weights.len() >= n),
            "apply_feedback invariant violation — ensure_neuromod_state_synced should have rebuilt"
        );
        if channel_idx >= n
            || n > self.neurons.len()
            || self.neurons.iter().any(|neu| neu.weights.len() < n)
        {
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

#[cfg(test)]
mod validate_tests {
    use super::*;
    use serde_json::json;

    fn router_field(err: &MeshError) -> &'static str {
        match err {
            MeshError::InvalidRouterConfig { field, .. } => field,
            other => panic!("expected InvalidRouterConfig, got {other:?}"),
        }
    }

    fn serde_field_from_config_json(value: serde_json::Value) -> String {
        let err = serde_json::from_value::<RouterConfig>(value).expect_err("config should fail");
        err.to_string()
    }

    fn serde_field_from_router_json(value: serde_json::Value) -> String {
        let err = serde_json::from_value::<ChannelRouter>(value).expect_err("router should fail");
        err.to_string()
    }

    struct InvalidCase {
        name: &'static str,
        field: &'static str,
        mutate: fn(&mut RouterConfig),
        json_representable: bool,
    }

    fn invalid_cases() -> Vec<InvalidCase> {
        fn set_channel_count_zero(c: &mut RouterConfig) {
            c.channel_count = 0;
        }
        fn set_channel_count_over_max(c: &mut RouterConfig) {
            c.channel_count = MAX_ROUTER_CHANNELS + 1;
        }
        fn set_channel_count_usize_max(c: &mut RouterConfig) {
            c.channel_count = usize::MAX;
        }
        fn set_timesteps_zero(c: &mut RouterConfig) {
            c.routing_timesteps = 0;
        }
        fn set_self_nan(c: &mut RouterConfig) {
            c.self_weight = f32::NAN;
        }
        fn set_self_inf(c: &mut RouterConfig) {
            c.self_weight = f32::INFINITY;
        }
        fn set_cross_neg_inf(c: &mut RouterConfig) {
            c.cross_weight = f32::NEG_INFINITY;
        }
        fn set_threshold_nan(c: &mut RouterConfig) {
            c.threshold = f32::NAN;
        }
        fn set_leak_below(c: &mut RouterConfig) {
            c.leak = -0.01;
        }
        fn set_leak_above(c: &mut RouterConfig) {
            c.leak = 1.01;
        }
        fn set_leak_nan(c: &mut RouterConfig) {
            c.leak = f32::NAN;
        }
        fn set_leak_inf(c: &mut RouterConfig) {
            c.leak = f32::INFINITY;
        }
        fn set_min_fire_below(c: &mut RouterConfig) {
            c.min_fire_rate = -0.01;
        }
        fn set_min_fire_above(c: &mut RouterConfig) {
            c.min_fire_rate = 1.01;
        }
        fn set_min_fire_nan(c: &mut RouterConfig) {
            c.min_fire_rate = f32::NAN;
        }
        fn set_decay_below(c: &mut RouterConfig) {
            c.plasticity_decay = -0.01;
        }
        fn set_decay_above(c: &mut RouterConfig) {
            c.plasticity_decay = 1.01;
        }
        fn set_decay_nan(c: &mut RouterConfig) {
            c.plasticity_decay = f32::NAN;
        }
        fn set_potentiate_neg(c: &mut RouterConfig) {
            c.plasticity_potentiate = -0.01;
        }
        fn set_potentiate_nan(c: &mut RouterConfig) {
            c.plasticity_potentiate = f32::NAN;
        }
        fn set_potentiate_inf(c: &mut RouterConfig) {
            c.plasticity_potentiate = f32::INFINITY;
        }
        fn set_speed_below(c: &mut RouterConfig) {
            c.plasticity_speed = -0.01;
        }
        fn set_speed_above(c: &mut RouterConfig) {
            c.plasticity_speed = 1.01;
        }
        fn set_speed_nan(c: &mut RouterConfig) {
            c.plasticity_speed = f32::NAN;
        }
        fn set_fatigue_acc_below(c: &mut RouterConfig) {
            c.fatigue_accumulation = -0.01;
        }
        fn set_fatigue_acc_above(c: &mut RouterConfig) {
            c.fatigue_accumulation = 1.01;
        }
        fn set_fatigue_acc_nan(c: &mut RouterConfig) {
            c.fatigue_accumulation = f32::NAN;
        }
        fn set_fatigue_rec_below(c: &mut RouterConfig) {
            c.fatigue_recovery = -0.01;
        }
        fn set_fatigue_rec_above(c: &mut RouterConfig) {
            c.fatigue_recovery = 1.01;
        }
        fn set_fatigue_rec_inf(c: &mut RouterConfig) {
            c.fatigue_recovery = f32::INFINITY;
        }
        vec![
            InvalidCase {
                name: "channel_count_zero",
                field: "channel_count",
                mutate: set_channel_count_zero,
                json_representable: true,
            },
            InvalidCase {
                name: "channel_count_over_max",
                field: "channel_count",
                mutate: set_channel_count_over_max,
                json_representable: true,
            },
            InvalidCase {
                name: "channel_count_usize_max",
                field: "channel_count",
                mutate: set_channel_count_usize_max,
                json_representable: true,
            },
            InvalidCase {
                name: "routing_timesteps_zero",
                field: "routing_timesteps",
                mutate: set_timesteps_zero,
                json_representable: true,
            },
            InvalidCase {
                name: "self_weight_nan",
                field: "self_weight",
                mutate: set_self_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "self_weight_inf",
                field: "self_weight",
                mutate: set_self_inf,
                json_representable: false,
            },
            InvalidCase {
                name: "cross_weight_neg_inf",
                field: "cross_weight",
                mutate: set_cross_neg_inf,
                json_representable: false,
            },
            InvalidCase {
                name: "threshold_nan",
                field: "threshold",
                mutate: set_threshold_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "leak_below",
                field: "leak",
                mutate: set_leak_below,
                json_representable: true,
            },
            InvalidCase {
                name: "leak_above",
                field: "leak",
                mutate: set_leak_above,
                json_representable: true,
            },
            InvalidCase {
                name: "leak_nan",
                field: "leak",
                mutate: set_leak_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "leak_inf",
                field: "leak",
                mutate: set_leak_inf,
                json_representable: false,
            },
            InvalidCase {
                name: "min_fire_rate_below",
                field: "min_fire_rate",
                mutate: set_min_fire_below,
                json_representable: true,
            },
            InvalidCase {
                name: "min_fire_rate_above",
                field: "min_fire_rate",
                mutate: set_min_fire_above,
                json_representable: true,
            },
            InvalidCase {
                name: "min_fire_rate_nan",
                field: "min_fire_rate",
                mutate: set_min_fire_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "plasticity_decay_below",
                field: "plasticity_decay",
                mutate: set_decay_below,
                json_representable: true,
            },
            InvalidCase {
                name: "plasticity_decay_above",
                field: "plasticity_decay",
                mutate: set_decay_above,
                json_representable: true,
            },
            InvalidCase {
                name: "plasticity_decay_nan",
                field: "plasticity_decay",
                mutate: set_decay_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "plasticity_potentiate_negative",
                field: "plasticity_potentiate",
                mutate: set_potentiate_neg,
                json_representable: true,
            },
            InvalidCase {
                name: "plasticity_potentiate_nan",
                field: "plasticity_potentiate",
                mutate: set_potentiate_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "plasticity_potentiate_inf",
                field: "plasticity_potentiate",
                mutate: set_potentiate_inf,
                json_representable: false,
            },
            InvalidCase {
                name: "plasticity_speed_below",
                field: "plasticity_speed",
                mutate: set_speed_below,
                json_representable: true,
            },
            InvalidCase {
                name: "plasticity_speed_above",
                field: "plasticity_speed",
                mutate: set_speed_above,
                json_representable: true,
            },
            InvalidCase {
                name: "plasticity_speed_nan",
                field: "plasticity_speed",
                mutate: set_speed_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "fatigue_accumulation_below",
                field: "fatigue_accumulation",
                mutate: set_fatigue_acc_below,
                json_representable: true,
            },
            InvalidCase {
                name: "fatigue_accumulation_above",
                field: "fatigue_accumulation",
                mutate: set_fatigue_acc_above,
                json_representable: true,
            },
            InvalidCase {
                name: "fatigue_accumulation_nan",
                field: "fatigue_accumulation",
                mutate: set_fatigue_acc_nan,
                json_representable: false,
            },
            InvalidCase {
                name: "fatigue_recovery_below",
                field: "fatigue_recovery",
                mutate: set_fatigue_rec_below,
                json_representable: true,
            },
            InvalidCase {
                name: "fatigue_recovery_above",
                field: "fatigue_recovery",
                mutate: set_fatigue_rec_above,
                json_representable: true,
            },
            InvalidCase {
                name: "fatigue_recovery_inf",
                field: "fatigue_recovery",
                mutate: set_fatigue_rec_inf,
                json_representable: false,
            },
        ]
    }

    #[test]
    fn default_config_is_valid() {
        RouterConfig::default().validate().unwrap();
    }

    #[test]
    fn table_rejects_every_invalid_config_field() {
        for case in invalid_cases() {
            let mut config = RouterConfig::default();
            (case.mutate)(&mut config);
            let err = match config.validate() {
                Err(err) => err,
                Ok(()) => panic!("{}: validate should fail", case.name),
            };
            assert_eq!(
                router_field(&err),
                case.field,
                "{}: unexpected field in {err}",
                case.name
            );

            let ctor_err = ChannelRouter::try_with_config(config.clone()).expect_err(case.name);
            assert_eq!(
                router_field(&ctor_err),
                case.field,
                "{}: constructor field mismatch",
                case.name
            );
            assert_eq!(
                ctor_err, err,
                "{}: validate and try_with_config must return the same error",
                case.name
            );

            if !case.json_representable {
                continue;
            }
            let mut config_json = serde_json::to_value(RouterConfig::default()).unwrap();
            let mut mutated = RouterConfig::default();
            (case.mutate)(&mut mutated);
            // Copy the mutated field through JSON so constructor and serde see
            // the same representable value (usize::MAX, 0, out-of-range f32).
            let mutated_json = serde_json::to_value(&mutated).unwrap();
            config_json[case.field] = mutated_json[case.field].clone();
            let config_msg = serde_field_from_config_json(config_json.clone());
            assert!(
                config_msg.contains(case.field),
                "{}: RouterConfig serde missing field name: {config_msg}",
                case.name
            );
            assert!(
                config_msg.contains(&err.to_string()),
                "{}: RouterConfig serde should carry the same MeshError Display, got {config_msg}, expected {}",
                case.name,
                err
            );

            let mut router_json = serde_json::to_value(ChannelRouter::new()).unwrap();
            router_json["config"][case.field] = mutated_json[case.field].clone();
            let router_msg = serde_field_from_router_json(router_json);
            assert!(
                router_msg.contains(case.field),
                "{}: ChannelRouter serde missing field name: {router_msg}",
                case.name
            );
            assert!(
                router_msg.contains(&err.to_string()),
                "{}: ChannelRouter serde should carry the same MeshError Display, got {router_msg}, expected {}",
                case.name,
                err
            );
        }
    }

    #[test]
    fn valid_boundaries_round_trip() {
        let default = RouterConfig::default();
        let cases = [
            (
                "min_channels",
                RouterConfig {
                    channel_count: 1,
                    ..default.clone()
                },
            ),
            (
                "max_channels_validate_only",
                RouterConfig {
                    channel_count: MAX_ROUTER_CHANNELS,
                    ..default.clone()
                },
            ),
            (
                "leak_0",
                RouterConfig {
                    leak: 0.0,
                    ..default.clone()
                },
            ),
            (
                "leak_1",
                RouterConfig {
                    leak: 1.0,
                    ..default.clone()
                },
            ),
            (
                "min_fire_0",
                RouterConfig {
                    min_fire_rate: 0.0,
                    ..default.clone()
                },
            ),
            (
                "min_fire_1",
                RouterConfig {
                    min_fire_rate: 1.0,
                    ..default.clone()
                },
            ),
            (
                "decay_0",
                RouterConfig {
                    plasticity_decay: 0.0,
                    ..default.clone()
                },
            ),
            (
                "decay_1",
                RouterConfig {
                    plasticity_decay: 1.0,
                    ..default.clone()
                },
            ),
            (
                "potentiate_0",
                RouterConfig {
                    plasticity_potentiate: 0.0,
                    ..default.clone()
                },
            ),
            (
                "speed_0",
                RouterConfig {
                    plasticity_speed: 0.0,
                    ..default.clone()
                },
            ),
            (
                "speed_1",
                RouterConfig {
                    plasticity_speed: 1.0,
                    ..default.clone()
                },
            ),
            (
                "fatigue_acc_0",
                RouterConfig {
                    fatigue_accumulation: 0.0,
                    ..default.clone()
                },
            ),
            (
                "fatigue_acc_1",
                RouterConfig {
                    fatigue_accumulation: 1.0,
                    ..default.clone()
                },
            ),
            (
                "fatigue_rec_0",
                RouterConfig {
                    fatigue_recovery: 0.0,
                    ..default.clone()
                },
            ),
            (
                "fatigue_rec_1",
                RouterConfig {
                    fatigue_recovery: 1.0,
                    ..default.clone()
                },
            ),
            (
                "signed_cross_inhibition",
                RouterConfig {
                    cross_weight: -1.0,
                    ..default.clone()
                },
            ),
            (
                "positive_cross_weight",
                RouterConfig {
                    cross_weight: 0.25,
                    ..default.clone()
                },
            ),
            (
                "signed_self_weight",
                RouterConfig {
                    self_weight: -0.3,
                    ..default.clone()
                },
            ),
            (
                "zero_threshold",
                RouterConfig {
                    threshold: 0.0,
                    ..default.clone()
                },
            ),
            (
                "one_timestep",
                RouterConfig {
                    routing_timesteps: 1,
                    ..default
                },
            ),
        ];

        for (name, config) in cases {
            config
                .validate()
                .unwrap_or_else(|err| panic!("{name}: valid boundary rejected: {err}"));
            let json = serde_json::to_value(&config).unwrap();
            let restored: RouterConfig = serde_json::from_value(json)
                .unwrap_or_else(|err| panic!("{name}: valid config failed to deserialize: {err}"));
            assert_eq!(restored, config, "{name}: config round-trip");

            // Skip constructing the 1024-channel router in this table; the
            // dedicated max-channel test covers allocation after validation.
            if config.channel_count == MAX_ROUTER_CHANNELS {
                continue;
            }
            let router = ChannelRouter::try_with_config(config.clone())
                .unwrap_or_else(|err| panic!("{name}: try_with_config: {err}"));
            let router_json = serde_json::to_value(&router).unwrap();
            let restored_router: ChannelRouter = serde_json::from_value(router_json)
                .unwrap_or_else(|err| panic!("{name}: router deserialize: {err}"));
            assert_eq!(restored_router.config(), router.config(), "{name}");
            assert_eq!(
                restored_router.weight_matrix(),
                router.weight_matrix(),
                "{name}"
            );
            assert_eq!(restored_router.fatigue(), router.fatigue(), "{name}");
            assert_eq!(restored_router.total_routes, router.total_routes, "{name}");
        }
    }

    #[test]
    fn custom_and_default_routers_round_trip() {
        let default_router = ChannelRouter::new();
        let json = serde_json::to_value(&default_router).unwrap();
        let restored: ChannelRouter = serde_json::from_value(json).unwrap();
        assert_eq!(restored.config(), default_router.config());
        assert_eq!(restored.weight_matrix(), default_router.weight_matrix());

        let custom = RouterConfig {
            channel_count: 4,
            self_weight: 1.1,
            cross_weight: -0.2,
            threshold: 0.3,
            leak: 0.05,
            routing_timesteps: 8,
            min_fire_rate: 0.25,
            plasticity_decay: 0.03,
            plasticity_potentiate: 0.2,
            plasticity_speed: 0.4,
            fatigue_accumulation: 0.2,
            fatigue_recovery: 0.1,
        };
        let router = ChannelRouter::try_with_config(custom.clone()).unwrap();
        let json = serde_json::to_value(&router).unwrap();
        let restored: ChannelRouter = serde_json::from_value(json).unwrap();
        assert_eq!(restored.config(), &custom);
        assert_eq!(restored.weight_matrix(), router.weight_matrix());
    }

    #[test]
    fn try_with_config_rejects_oversized_channel_count_without_allocating() {
        // If validation ran after `vec![0.0; n]`, usize::MAX would OOM/hang
        // this test rather than returning.
        let config = RouterConfig {
            channel_count: usize::MAX,
            ..RouterConfig::default()
        };
        let err = ChannelRouter::try_with_config(config).unwrap_err();
        assert_eq!(router_field(&err), "channel_count");
    }

    #[test]
    fn max_channel_count_is_accepted_by_validate_and_constructor() {
        let config = RouterConfig {
            channel_count: MAX_ROUTER_CHANNELS,
            ..RouterConfig::default()
        };
        config.validate().unwrap();
        let router = ChannelRouter::try_with_config(config).unwrap();
        assert_eq!(router.config().channel_count, MAX_ROUTER_CHANNELS);
        assert_eq!(router.weight_matrix().len(), MAX_ROUTER_CHANNELS);
        assert_eq!(router.weight_matrix()[0].len(), MAX_ROUTER_CHANNELS);
        assert_eq!(router.fatigue().len(), MAX_ROUTER_CHANNELS);
    }

    #[test]
    fn malformed_shapes_are_rejected_with_matching_fields() {
        let router = ChannelRouter::new();
        let base = serde_json::to_value(&router).unwrap();

        let mut empty_inner = base.clone();
        empty_inner["baseline_weights"] = json!([[], [], []]);
        let msg = serde_field_from_router_json(empty_inner);
        assert!(msg.contains("baseline_weights"), "empty inner rows: {msg}");

        let mut short_fatigue = base.clone();
        short_fatigue["channel_fatigue"] = json!([0.0, 0.0]);
        let msg = serde_field_from_router_json(short_fatigue);
        assert!(msg.contains("channel_fatigue"), "short fatigue: {msg}");

        let mut extra_neurons = base.clone();
        extra_neurons["neurons"] = json!([]);
        let msg = serde_field_from_router_json(extra_neurons);
        assert!(msg.contains("neurons"), "empty neurons: {msg}");

        let mut missing_weights = base.clone();
        missing_weights["neurons"][0]
            .as_object_mut()
            .unwrap()
            .remove("weights");
        let msg = serde_field_from_router_json(missing_weights);
        assert!(msg.contains("weights"), "missing neuron weights: {msg}");

        let mut ragged_baseline = base.clone();
        ragged_baseline["baseline_weights"] = json!([[0.9, -0.15, -0.15], [0.0], [0.0, 0.0, 0.0]]);
        let msg = serde_field_from_router_json(ragged_baseline);
        assert!(msg.contains("baseline_weights"), "ragged baseline: {msg}");

        let mut out_of_range_fatigue = base;
        out_of_range_fatigue["channel_fatigue"] = json!([0.0, 1.5, 0.0]);
        let msg = serde_field_from_router_json(out_of_range_fatigue);
        assert!(
            msg.contains("channel_fatigue"),
            "fatigue out of range: {msg}"
        );
    }

    #[test]
    fn legacy_snapshot_without_optional_fields_deserializes() {
        let router = ChannelRouter::new();
        let mut json = serde_json::to_value(&router).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("config");
        obj.remove("channel_fatigue");
        obj.remove("baseline_weights");

        let restored: ChannelRouter =
            serde_json::from_value(json).expect("legacy snapshot must deserialize");
        assert_eq!(restored.config().channel_count, 3);
        assert_eq!(restored.weight_matrix().len(), 3);
        assert_eq!(restored.fatigue(), &[0.0, 0.0, 0.0]);
        let _ = restored
            .clone()
            .route_modulated([0.5, 0.0, 0.0], &NeuromodState::balanced())
            .unwrap();
    }

    #[test]
    fn signed_zero_rates_are_accepted() {
        let config = RouterConfig {
            leak: -0.0,
            min_fire_rate: -0.0,
            plasticity_decay: -0.0,
            plasticity_potentiate: -0.0,
            plasticity_speed: -0.0,
            fatigue_accumulation: -0.0,
            fatigue_recovery: -0.0,
            ..RouterConfig::default()
        };
        config.validate().unwrap();
        ChannelRouter::try_with_config(config).unwrap();
    }
}
