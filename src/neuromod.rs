//! Neuromodulatory Integrative Fixed-threshold (NIF) neuron model.
//!
//! Provides a neuron model where the effective synaptic input is modulated
//! by a dynamic gain parameter, allowing for global or local modulation
//! of signal integration sensitivity.

use serde::{Deserialize, Serialize};

/// A neuron model with neuromodulatory gain control.
///
/// $V_{t+1} = V_t + (G \cdot I_{syn}) - \lambda(V_t - V_{rest})$
/// where $G$ is the modulation gain and $\lambda$ is the leak rate.
#[derive(Clone, Serialize, Deserialize, Debug)]
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
