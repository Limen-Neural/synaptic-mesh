// SPDX-License-Identifier: MIT OR Apache-2.0

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

use crate::error::{MeshError, Result};

/// Ring-buffer delay queue for spike delivery.
///
/// At each simulation tick:
/// 1. Call [`SpikeDelayBuffer::inject`] for each spiking synapse to schedule future delivery.
/// 2. Call [`SpikeDelayBuffer::drain_current_tick`] to collect all currents that have arrived.
/// 3. Call [`SpikeDelayBuffer::advance`] to move the tick forward.
///
/// With `max_delay == 0` the buffer holds a single slot and every spike is
/// delivered in the same tick it is injected, behaving as if there were no
/// delay layer — it still allocates that one slot, one `f32` per neuron.
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
    /// The ring has `max_delay + 1` slots so a spike injected with
    /// `delay == max_delay` lands on a future tick rather than wrapping
    /// onto the current slot.
    ///
    /// # Arguments
    ///
    /// * `neuron_count` — number of target neurons (slot width)
    /// * `max_delay` — maximum axonal delay in ticks
    ///
    /// # Panics
    ///
    /// Panics if `max_delay + 1` overflows `usize`. Prefer
    /// [`SpikeDelayBuffer::try_new`] when the caller needs a recoverable
    /// error.
    pub fn new(neuron_count: usize, max_delay: usize) -> Self {
        Self::try_new(neuron_count, max_delay).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible constructor that rejects a `max_delay` whose ring depth
    /// (`max_delay + 1`) would overflow `usize`.
    pub fn try_new(neuron_count: usize, max_delay: usize) -> Result<Self> {
        let depth = max_delay.checked_add(1).ok_or_else(|| {
            MeshError::DelayError(format!(
                "max_delay {max_delay} + 1 overflows usize; ring depth cannot be represented"
            ))
        })?;
        Ok(Self {
            slots: vec![vec![0.0; neuron_count]; depth],
            neuron_count,
            max_delay,
            current_tick: 0,
        })
    }

    /// Inject a spike from a source neuron through a synapse.
    ///
    /// The synaptic current `weight` will be delivered to `target` neuron
    /// after `delay` ticks from the current tick.
    ///
    /// Bounds are checked in **both** debug and release builds before any
    /// slot is modified. An oversized delay is rejected rather than
    /// wrapping onto an earlier tick via modulo arithmetic.
    ///
    /// # Panics
    ///
    /// Panics if `delay > max_delay` or `target >= neuron_count`. Prefer
    /// [`SpikeDelayBuffer::try_inject`] when the caller needs a recoverable
    /// error.
    #[inline]
    pub fn inject(&mut self, target: usize, weight: f32, delay: usize) {
        self.try_inject(target, weight, delay)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible inject that leaves the buffer unchanged when `delay` or
    /// `target` is out of range.
    #[inline]
    pub fn try_inject(&mut self, target: usize, weight: f32, delay: usize) -> Result<()> {
        if delay > self.max_delay {
            return Err(MeshError::DelayError(format!(
                "delay {delay} exceeds max_delay {}",
                self.max_delay
            )));
        }
        if target >= self.neuron_count {
            return Err(MeshError::IndexOutOfBounds {
                index: target,
                max: self.neuron_count.saturating_sub(1),
            });
        }
        // Constructors keep depth == max_delay + 1, but a deserialized
        // ring can still be shorter. Refuse delay >= depth so we never
        // wrap onto an earlier tick.
        let depth = self.slots.len();
        if depth == 0 || delay >= depth {
            return Err(MeshError::DelayError(format!(
                "delay {delay} exceeds buffer depth {}",
                depth.saturating_sub(1)
            )));
        }
        let slot_idx = (self.current_tick as usize + delay) % depth;
        self.slots[slot_idx][target] += weight;
        Ok(())
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
    fn inject_delay_zero_is_accepted() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(1, 1.0, 0);
        let currents = buffer.drain_current_tick();
        assert!((currents[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inject_delay_equal_to_capacity_is_accepted() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(0, 1.0, 1);
        let c0 = buffer.drain_current_tick();
        assert_eq!(c0[0], 0.0);
        buffer.advance();
        let c1 = buffer.drain_current_tick();
        assert!((c1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "delay 2 exceeds max_delay 1")]
    fn inject_excessive_delay_is_rejected() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(1, 1.0, 2);
    }

    #[test]
    fn try_inject_excessive_delay_leaves_buffer_unchanged() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        assert!(buffer.try_inject(1, 1.0, 2).is_err());
        let currents = buffer.drain_current_tick();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn inject_out_of_range_target_is_rejected() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        buffer.inject(2, 1.0, 0);
    }

    #[test]
    fn try_inject_out_of_range_target_leaves_buffer_unchanged() {
        let mut buffer = SpikeDelayBuffer::new(2, 1);
        assert!(buffer.try_inject(2, 1.0, 0).is_err());
        let currents = buffer.drain_current_tick();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn try_new_rejects_max_delay_plus_one_overflow() {
        let err = SpikeDelayBuffer::try_new(1, usize::MAX).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("overflows usize") && msg.contains("ring depth cannot be represented"),
            "expected overflow error, got {msg}"
        );
    }

    #[test]
    fn try_inject_rejects_delay_that_exceeds_deserialized_depth() {
        // Derived Deserialize still accepts slots.len() < max_delay + 1.
        // A delay within max_delay must not wrap through modulo onto an
        // earlier tick.
        let json = r#"{"slots":[[0.0]],"neuron_count":1,"max_delay":1,"current_tick":0}"#;
        let mut buf: SpikeDelayBuffer = serde_json::from_str(json).unwrap();
        assert!(buf.try_inject(0, 1.0, 1).is_err());
        let currents = buf.drain_current_tick();
        assert_eq!(currents[0], 0.0);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn new_rejects_max_delay_plus_one_overflow() {
        let _ = SpikeDelayBuffer::new(1, usize::MAX);
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
