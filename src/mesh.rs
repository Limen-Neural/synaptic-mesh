//! [`SynapticMesh`] — top-level orchestrator owning topology + delays.
//!
//! The `SynapticMesh` is the primary public-facing struct of this crate.
//! It owns a [`SynapticGraph`] (the wiring diagram) and a [`SpikeDelayBuffer`]
//! (the temporal delay infrastructure), and provides a single `propagate()`
//! method that converts source spikes into delayed synaptic currents.
//!
//! # Usage
//!
//! ```rust
//! use synapse_router::mesh::SynapticMesh;
//! use synapse_router::topology::generators::generate_random;
//!
//! let graph = generate_random(64, 0.1, 5, 0.2).unwrap();
//! let mut mesh = SynapticMesh::new(graph);
//!
//! // Each tick: provide binary spike vector, receive synaptic currents
//! let spikes = vec![false; 64];
//! let currents = mesh.propagate(&spikes);
//! ```

use serde::{Deserialize, Serialize};

use crate::delay::SpikeDelayBuffer;
use crate::error::{MeshError, Result};
use crate::topology::SynapticGraph;

/// Top-level synaptic wiring orchestrator.
///
/// Owns the network topology (graph with weights, delays, polarities) and
/// the temporal delay buffer. Converts source spikes into time-delayed
/// synaptic currents delivered to target neurons.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticMesh {
    /// The wiring diagram.
    graph: SynapticGraph,
    /// Ring-buffer delay queue for spike delivery.
    delay_buffer: SpikeDelayBuffer,
    /// Current simulation tick.
    tick: u64,
}

impl SynapticMesh {
    /// Create a new mesh from a pre-built graph.
    ///
    /// The delay buffer is sized to the graph's maximum delay.
    pub fn new(graph: SynapticGraph) -> Self {
        let max_delay = graph.max_delay() as usize;
        let n = graph.neuron_count();
        Self {
            delay_buffer: SpikeDelayBuffer::new(n, max_delay),
            graph,
            tick: 0,
        }
    }

    /// Create a mesh with a custom maximum delay (overriding graph's max).
    ///
    /// Useful when you want headroom for future dynamic delay changes.
    pub fn with_max_delay(graph: SynapticGraph, max_delay: usize) -> Self {
        let n = graph.neuron_count();
        Self {
            delay_buffer: SpikeDelayBuffer::new(n, max_delay),
            graph,
            tick: 0,
        }
    }

    /// Propagate spikes through the mesh for one tick.
    ///
    /// # Arguments
    ///
    /// * `source_spikes` — boolean spike vector: `true` if neuron `i` fired
    ///   this tick. Length must equal `neuron_count()`.
    ///
    /// # Returns
    ///
    /// Synaptic current vector of length `neuron_count()` — the total
    /// incoming current at each neuron for this tick (including delayed
    /// spikes from previous ticks).
    pub fn propagate(&mut self, source_spikes: &[bool]) -> Result<Vec<f32>> {
        let n = self.graph.neuron_count();
        if source_spikes.len() != n {
            return Err(MeshError::NeuronCountMismatch {
                expected: n,
                got: source_spikes.len(),
                context: "propagate source_spikes".into(),
            });
        }

        // Inject spikes from all firing neurons into the delay buffer
        for (src, &fired) in source_spikes.iter().enumerate() {
            if !fired {
                continue;
            }
            for (target, weight, delay, _polarity) in self.graph.outgoing(src) {
                self.delay_buffer.inject(target as usize, weight, delay as usize);
            }
        }

        // Drain currents that have arrived at this tick
        let currents = self.delay_buffer.drain_current_tick();

        // Advance the buffer
        self.delay_buffer.advance();
        self.tick += 1;

        Ok(currents)
    }

    /// Propagate with floating-point spike strengths instead of binary.
    ///
    /// # Arguments
    ///
    /// * `source_activations` — activation level for each neuron.
    ///   Non-zero values are treated as spikes; the activation value
    ///   scales the synaptic weight.
    pub fn propagate_graded(&mut self, source_activations: &[f32]) -> Result<Vec<f32>> {
        let n = self.graph.neuron_count();
        if source_activations.len() != n {
            return Err(MeshError::NeuronCountMismatch {
                expected: n,
                got: source_activations.len(),
                context: "propagate_graded source_activations".into(),
            });
        }

        for (src, &activation) in source_activations.iter().enumerate() {
            if activation.abs() < 1e-9 {
                continue;
            }
            for (target, weight, delay, _polarity) in self.graph.outgoing(src) {
                self.delay_buffer.inject(
                    target as usize,
                    weight * activation,
                    delay as usize,
                );
            }
        }

        let currents = self.delay_buffer.drain_current_tick();
        self.delay_buffer.advance();
        self.tick += 1;

        Ok(currents)
    }

    /// Current simulation tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Number of neurons in the mesh.
    pub fn neuron_count(&self) -> usize {
        self.graph.neuron_count()
    }

    /// Total number of synapses.
    pub fn synapse_count(&self) -> usize {
        self.graph.synapse_count()
    }

    /// Sparsity of the connectivity matrix.
    pub fn sparsity(&self) -> f32 {
        self.graph.sparsity()
    }

    /// Maximum delay in ticks.
    pub fn max_delay(&self) -> usize {
        self.delay_buffer.max_delay()
    }

    /// Mean out-degree of the network.
    pub fn mean_degree(&self) -> f32 {
        self.graph.mean_degree()
    }

    /// Immutable access to the underlying graph.
    pub fn graph(&self) -> &SynapticGraph {
        &self.graph
    }

    /// Reset delay buffer and tick counter.
    pub fn reset(&mut self) {
        self.delay_buffer.reset();
        self.tick = 0;
    }

    /// Export the CSR arrays + delays for GPU upload.
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u16>) {
        self.graph.to_gpu_arrays()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{generate_layered, generate_random, generate_small_world};

    #[test]
    fn propagate_length_mismatch_rejected() {
        let graph = generate_random(10, 0.5, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        let bad_spikes = vec![false; 5]; // wrong length
        assert!(mesh.propagate(&bad_spikes).is_err());
    }

    #[test]
    fn no_spikes_no_current() {
        let graph = generate_random(10, 0.5, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        let spikes = vec![false; 10];
        let currents = mesh.propagate(&spikes).unwrap();
        assert!(currents.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn instant_delivery_with_zero_delay() {
        // Build a 2-neuron graph with 0-delay connection: 0 → 1
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.8,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        let mut spikes = vec![false; 2];
        spikes[0] = true;
        let currents = mesh.propagate(&spikes).unwrap();
        assert!((currents[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn delayed_delivery_through_mesh() {
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 1.0,
            delay: 2,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Tick 0: neuron 0 fires
        let mut spikes = vec![false; 2];
        spikes[0] = true;
        let c0 = mesh.propagate(&spikes).unwrap();
        assert_eq!(c0[1], 0.0); // not arrived yet

        // Tick 1: no fires
        spikes[0] = false;
        let c1 = mesh.propagate(&spikes).unwrap();
        assert_eq!(c1[1], 0.0); // not arrived yet

        // Tick 2: spike arrives
        let c2 = mesh.propagate(&spikes).unwrap();
        assert!((c2[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn graded_propagation() {
        use crate::types::{Polarity, SynapseDescriptor};
        let desc = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let graph = SynapticGraph::from_descriptors(2, &desc).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        let activations = vec![0.6, 0.0];
        let currents = mesh.propagate_graded(&activations).unwrap();
        // 0.5 * 0.6 = 0.3
        assert!((currents[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tick_counter_increments() {
        let graph = generate_random(8, 0.3, 2, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        assert_eq!(mesh.tick(), 0);
        mesh.propagate(&vec![false; 8]).unwrap();
        mesh.propagate(&vec![false; 8]).unwrap();
        assert_eq!(mesh.tick(), 2);
    }

    #[test]
    fn reset_clears_state() {
        let graph = generate_random(8, 0.3, 2, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        mesh.propagate(&vec![true; 8]).unwrap();
        mesh.propagate(&vec![true; 8]).unwrap();
        mesh.reset();
        assert_eq!(mesh.tick(), 0);
    }

    #[test]
    fn small_world_mesh_propagation() {
        let graph = generate_small_world(32, 4, 0.2, 5, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);

        // Fire a single neuron and run for several ticks
        let mut any_current = false;
        for tick in 0..10 {
            let mut spikes = vec![false; 32];
            if tick == 0 {
                spikes[0] = true;
            }
            let currents = mesh.propagate(&spikes).unwrap();
            if currents.iter().any(|&c| c.abs() > 1e-6) {
                any_current = true;
            }
        }
        assert!(any_current, "expected some delayed current delivery");
    }

    #[test]
    fn layered_mesh_feed_forward() {
        let graph = generate_layered(&[4, 8, 2], 1.0, 3, 0.2).unwrap();
        let mut mesh = SynapticMesh::new(graph);
        assert_eq!(mesh.neuron_count(), 14);

        // Fire input layer
        let mut spikes = vec![false; 14];
        for i in 0..4 {
            spikes[i] = true;
        }
        // Run enough ticks for delays to propagate
        let mut last_layer_activated = false;
        for _ in 0..10 {
            let currents = mesh.propagate(&spikes).unwrap();
            spikes = vec![false; 14];
            // Check if last-layer neurons (12, 13) received current
            if currents[12].abs() > 1e-6 || currents[13].abs() > 1e-6 {
                last_layer_activated = true;
            }
        }
        assert!(
            last_layer_activated,
            "feed-forward should eventually reach the output layer"
        );
    }
}
