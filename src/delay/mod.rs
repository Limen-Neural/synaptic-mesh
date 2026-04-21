//! Delay module — temporal delay infrastructure for spike propagation.
//!
//! Provides ring-buffer-based delay queues that deliver spikes at the
//! correct future tick according to per-synapse axonal propagation delays.

mod ring_buffer;

pub use ring_buffer::SpikeDelayBuffer;
