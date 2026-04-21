//! Generalized Integrate-and-Fire (GIF) neuron.
//!
//! Provides a more biologically plausible model than the standard LIF,
//! incorporating an adaptive threshold that increases with mỗi spike and
//! decays back to a baseline, simulating exhaustion/adaptation.

use serde::{Deserialize, Serialize};

/// Generalized Integrate-and-Fire neuron.
///
/// Unlike basic LIF, the GIF model features a dynamic threshold $\Theta(t)$.
/// Every time the neuron fires, the threshold increases by a step ($\Delta\Theta$),
/// making it harder to fire again immediately (adaptation).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct GifNeuron {
    /// Current membrane potential.
    pub v: f32,
    /// Resting membrane potential (baseline).
    pub v_rest: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Passive decay rate (leak) per timestep.
    pub decay_rate: f32,

    /// Instantaneous firing threshold.
    pub theta: f32,
    /// Base firing threshold (minimum).
    pub theta_base: f32,
    /// Threshold decay rate (return to baseline).
    pub theta_decay: f32,
    /// Threshold increment per spike (adaptation strength).
    pub theta_step: f32,

    /// Synaptic weights — one per input channel.
    pub weights: Vec<f32>,

    /// Whether this neuron fired on the last timestep.
    pub last_spike: bool,

    /// Normalised adaptation state [0.0, 1.0] for telemetry.
    pub adaptation_level: f32,
}

impl Default for GifNeuron {
    fn default() -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            decay_rate: 0.15,
            theta: 0.1,
            theta_base: 0.1,
            theta_decay: 0.1,
            theta_step: 0.05,
            weights: Vec::new(),
            last_spike: false,
            adaptation_level: 0.0,
        }
    }
}

impl GifNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance neuron dynamics by one timestep.
    pub fn integrate(&mut self, stimulus: f32) {
        // 1. Update membrane potential (LIF logic)
        self.v += stimulus;
        self.v -= (self.v - self.v_rest) * self.decay_rate;

        // 2. Decay threshold towards baseline
        self.theta -= (self.theta - self.theta_base) * self.theta_decay;

        // 3. Update normalised adaptation level for telemetry
        // (Assuming theta_step scale is roughly where 1.0 = exhausted)
        self.adaptation_level = (self.theta - self.theta_base) / (self.theta_base * 5.0);
        self.adaptation_level = self.adaptation_level.clamp(0.0, 1.0);
    }

    /// Check if the neuron spikes. Resets V and increases Theta on fire.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.v >= self.theta {
            let peak = self.v;
            self.v = self.v_reset;
            // Adaptive threshold bump
            self.theta += self.theta_step;
            return Some(peak);
        }
        None
    }
}
