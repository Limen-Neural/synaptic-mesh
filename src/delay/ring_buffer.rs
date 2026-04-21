//! Ring-buffer delay queue for tick-aligned spike delivery.
//!
//! The [`SpikeDelayBuffer`] implements a fixed-size circular buffer where
//! spikes are injected at the current tick plus a per-synapse delay, and
//! delivered (drained) at each tick advance.
//!
//! # Design
//!
//! ```text
//! tick 0:  inject spike at delay=3  →  buffer[3] += weight
//! tick 1:  ...
//! tick 2:  ...
//! tick 3:  drain buffer[3]  →  deliver accumulated current to target neuron
//! ```
//!
//! The ring buffer has `max_delay + 1` slots, each slot is a vector of
//! length `neuron_count` accumulating incoming synaptic current.

use serde::{Deserialize, Serialize};

/// Ring-buffer delay queue for spike delivery.
///
/// At each simulation tick:
/// 1. Call [`inject`] for each spiking synapse to schedule future delivery.
/// 2. Call [`drain_current_tick`] to collect all currents that have arrived.
/// 3. Call [`advance`] to move the tick forward.
///
/// The buffer is zero-cost when `max_delay == 0` (all spikes delivered
/// in the same tick they are injected, same as the existing AHL router behaviour).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpikeDelayBuffer {
    /// Ring buffer: `slots[slot_index][neuron_id]` → accumulated current.
    slots: Vec<Vec<f32>>,
    /// Number of neurons (width of each slot).
    neuron_count: usize,
    /// Maximum delay in ticks (depth of the ring buffer minus 1).
    max_delay: usize,
    /// Current simulation tick.
    current_tick: u64,
}

impl SpikeDelayBuffer {
    /// Create a new delay buffer.
    ///
    /// # Arguments
    ///
    /// * `neuron_count` — number of target neurons (slot width)
    /// * `max_delay` — maximum axonal delay in ticks
    pub fn new(neuron_count: usize, max_delay: usize) -> Self {
        let depth = max_delay + 1;
        Self {
            slots: vec![vec![0.0; neuron_count]; depth],
            neuron_count,
            max_delay,
            current_tick: 0,
        }
    }

    /// Inject a spike from a source neuron through a synapse.
    ///
    /// The synaptic current `weight` will be delivered to `target` neuron
    /// after `delay` ticks from the current tick.
    ///
    /// # Panics
    ///
    /// Panics if `delay > max_delay` or `target >= neuron_count`.
    #[inline]
    pub fn inject(&mut self, target: usize, weight: f32, delay: usize) {
        debug_assert!(delay <= self.max_delay, "delay {delay} > max_delay {}", self.max_delay);
        debug_assert!(target < self.neuron_count, "target {target} >= neuron_count {}", self.neuron_count);

        let slot_idx = (self.current_tick as usize + delay) % self.slots.len();
        self.slots[slot_idx][target] += weight;
    }

    /// Drain the current tick's accumulated synaptic currents.
    ///
    /// Returns a slice of length `neuron_count` with the total synaptic
    /// current arriving at each neuron in this tick. The slot is zeroed
    /// after draining.
    pub fn drain_current_tick(&mut self) -> Vec<f32> {
        let slot_idx = self.current_tick as usize % self.slots.len();
        let currents = self.slots[slot_idx].clone();
        self.slots[slot_idx].fill(0.0);
        currents
    }

    /// Advance to the next tick.
    pub fn advance(&mut self) {
        self.current_tick += 1;
    }

    /// Current simulation tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Maximum delay supported by this buffer.
    pub fn max_delay(&self) -> usize {
        self.max_delay
    }

    /// Number of neurons (slot width).
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Reset all slots and the tick counter.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.fill(0.0);
        }
        self.current_tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_delay_delivers_same_tick() {
        let mut buf = SpikeDelayBuffer::new(4, 0);
        buf.inject(2, 0.75, 0);
        let currents = buf.drain_current_tick();
        assert!((currents[2] - 0.75).abs() < 1e-6);
        assert_eq!(currents[0], 0.0);
    }

    #[test]
    fn delayed_delivery() {
        let mut buf = SpikeDelayBuffer::new(4, 5);

        // Inject at tick 0 with delay 3 → should arrive at tick 3
        buf.inject(1, 0.5, 3);

        // Tick 0: nothing delivered to neuron 1
        let c0 = buf.drain_current_tick();
        assert_eq!(c0[1], 0.0);
        buf.advance();

        // Tick 1: nothing
        let c1 = buf.drain_current_tick();
        assert_eq!(c1[1], 0.0);
        buf.advance();

        // Tick 2: nothing
        let c2 = buf.drain_current_tick();
        assert_eq!(c2[1], 0.0);
        buf.advance();

        // Tick 3: delivered!
        let c3 = buf.drain_current_tick();
        assert!((c3[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn multiple_spikes_accumulate() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 0.3, 2);
        buf.inject(0, 0.7, 2);

        buf.advance();
        buf.advance();

        let currents = buf.drain_current_tick();
        assert!((currents[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ring_buffer_wraps_correctly() {
        let mut buf = SpikeDelayBuffer::new(2, 3);

        // Run for more ticks than the ring buffer depth
        for tick in 0..10 {
            buf.inject(0, 1.0, 2);
            let currents = buf.drain_current_tick();
            if tick >= 2 {
                // After tick 2, we should receive the spike injected 2 ticks ago
                assert!(
                    (currents[0] - 1.0).abs() < 1e-6,
                    "tick {tick}: expected 1.0, got {}",
                    currents[0]
                );
            }
            buf.advance();
        }
    }

    #[test]
    fn reset_clears_everything() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 0.5, 3);
        buf.advance();
        buf.advance();
        buf.reset();

        assert_eq!(buf.current_tick(), 0);
        let currents = buf.drain_current_tick();
        assert!(currents.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn different_delays_different_arrival() {
        let mut buf = SpikeDelayBuffer::new(4, 5);
        buf.inject(0, 1.0, 1); // arrives tick 1
        buf.inject(1, 2.0, 3); // arrives tick 3

        buf.advance(); // tick 1
        let c1 = buf.drain_current_tick();
        assert!((c1[0] - 1.0).abs() < 1e-6);
        assert_eq!(c1[1], 0.0);

        buf.advance(); // tick 2
        let c2 = buf.drain_current_tick();
        assert_eq!(c2[0], 0.0);
        assert_eq!(c2[1], 0.0);

        buf.advance(); // tick 3
        let c3 = buf.drain_current_tick();
        assert_eq!(c3[0], 0.0);
        assert!((c3[1] - 2.0).abs() < 1e-6);
    }
}
