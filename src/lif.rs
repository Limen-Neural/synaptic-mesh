//! Leaky Integrate-and-Fire (LIF) neuron.
//!
//! # Circuit analogy (RC circuit)
//!
//! ```text
//!  V_membrane  ──┤C├──┬── (leak) ──GND
//!                     │
//!               stimulus (current in)
//! ```
//!
//! - `membrane_potential` — voltage across the capacitor
//! - `decay_rate` — conductance of the leak resistor
//! - `threshold` — breakdown voltage (Diode / Spark Gap)
//! - `weights` — resistor values on each input trace (synaptic strength)
//!
//! ## Biological reference
//!
//! Integrate-and-fire model: Lapicque, L. (1907). *Recherches quantitatives sur
//! l'excitation électrique des nerfs traitée comme une polarisation.* Journal de
//! Physiologie et de Pathologie Générale, 9, 620–635.
//!
//! Leaky variant: Stein, R. B. (1967). *Some models of neuronal variability.*
//! Biophysical Journal, 7(1), 37–68.

use serde::{Deserialize, Serialize};

/// Leaky Integrate-and-Fire neuron.
///
/// Stateful: each call to [`integrate`] advances the membrane dynamics by one
/// timestep; [`check_fire`] returns `Some(peak)` and resets to zero on threshold
/// crossing.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LifNeuron {
    /// Current membrane potential (charge state).
    pub membrane_potential: f32,
    /// Passive leak rate per timestep (fraction of membrane potential lost).
    pub decay_rate: f32,
    /// Firing threshold — spike issued when `membrane_potential >= threshold`.
    pub threshold: f32,
    /// Resting threshold (stored so dynamic modulation can decay back to baseline).
    #[serde(default)]
    pub base_threshold: f32,
    /// Whether this neuron fired on the last timestep.
    pub last_spike: bool,
    /// Synaptic weights — one per input channel (learned via STDP).
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Global step index of the most recent spike (for STDP Δt calculation).
    #[serde(default = "default_last_spike_time")]
    pub last_spike_time: i64,
}

fn default_last_spike_time() -> i64 { -1 }

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15,
            threshold: 0.02,
            base_threshold: 0.02,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }
}

impl LifNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance membrane dynamics by one timestep.
    ///
    /// 1. Integration: `V ← V + stimulus`
    /// 2. Leak: `V ← V − V · decay_rate`
    pub fn integrate(&mut self, stimulus: f32) {
        self.membrane_potential += stimulus;
        self.membrane_potential -= self.membrane_potential * self.decay_rate;
    }

    /// Check threshold crossing. Returns `Some(peak_before_reset)` on spike,
    /// `None` otherwise. Hard-resets membrane to 0 on fire (refractory period).
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.membrane_potential >= self.threshold {
            let peak = self.membrane_potential;
            self.membrane_potential = 0.0;
            return Some(peak);
        }
        None
    }
}
