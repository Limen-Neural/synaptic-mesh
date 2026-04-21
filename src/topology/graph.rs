//! [`SynapticGraph`] — adjacency structure with delay and polarity metadata.
//!
//! Extends the CSR sparse representation from `sparse.rs` with per-synapse
//! axonal delay and polarity information. This is the core "wiring diagram"
//! that the `SynapticMesh` orchestrator uses for spike propagation.

use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};
use crate::types::{DelayTicks, NeuronId, Polarity, SynapseDescriptor};

/// Adjacency structure for a spiking neural network with delay and polarity metadata.
///
/// Stores connections in Compressed Sparse Row (CSR) format, augmented with
/// per-synapse delay and polarity vectors aligned to the same index space.
///
/// # Layout
///
/// ```text
/// row_ptr:    [0,  3,  7, 10, ...]     — start index per source neuron
/// targets:    [5, 12,  0,  1,  3, ...]  — target neuron IDs
/// weights:    [0.9, -0.15, 0.3, ...]    — effective signed weights
/// delays:     [2, 5, 1, ...]            — axonal delay in ticks
/// polarities: [Exc, Inh, Exc, ...]      — Dale's law polarity
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapticGraph {
    /// Number of neurons in the graph.
    neuron_count: usize,
    /// Row pointers: `row_ptr[i]` gives the start index in the edge arrays
    /// for neuron `i`. Length is `neuron_count + 1`.
    row_ptr: Vec<usize>,
    /// Target neuron IDs for each synapse.
    targets: Vec<NeuronId>,
    /// Synaptic weights (signed: polarity × |weight|).
    weights: Vec<f32>,
    /// Axonal propagation delay per synapse.
    delays: Vec<DelayTicks>,
    /// Polarity per synapse.
    polarities: Vec<Polarity>,
}

impl SynapticGraph {
    /// Create an empty graph with `n` neurons and no connections.
    pub fn new(n: usize) -> Self {
        Self {
            neuron_count: n,
            row_ptr: vec![0; n + 1],
            targets: Vec::new(),
            weights: Vec::new(),
            delays: Vec::new(),
            polarities: Vec::new(),
        }
    }

    /// Build from a list of synapse descriptors.
    ///
    /// Descriptors need not be sorted; they will be grouped by source neuron.
    pub fn from_descriptors(neuron_count: usize, descriptors: &[SynapseDescriptor]) -> Result<Self> {
        // Validate indices
        for desc in descriptors {
            if desc.source as usize >= neuron_count {
                return Err(MeshError::IndexOutOfBounds {
                    index: desc.source as usize,
                    max: neuron_count - 1,
                });
            }
            if desc.target as usize >= neuron_count {
                return Err(MeshError::IndexOutOfBounds {
                    index: desc.target as usize,
                    max: neuron_count - 1,
                });
            }
        }

        // Count edges per source neuron
        let mut counts = vec![0usize; neuron_count];
        for desc in descriptors {
            counts[desc.source as usize] += 1;
        }

        // Build row_ptr
        let mut row_ptr = Vec::with_capacity(neuron_count + 1);
        row_ptr.push(0);
        for &c in &counts {
            row_ptr.push(row_ptr.last().unwrap() + c);
        }

        let nnz = descriptors.len();
        let mut targets = vec![0u32; nnz];
        let mut weights = vec![0.0f32; nnz];
        let mut delays = vec![0u16; nnz];
        let mut polarities = vec![Polarity::Excitatory; nnz];

        // Place each descriptor at the right position
        let mut cursor = counts.clone();
        // cursor[i] = how many edges for neuron i have been placed so far
        cursor.fill(0);

        for desc in descriptors {
            let src = desc.source as usize;
            let pos = row_ptr[src] + cursor[src];
            targets[pos] = desc.target;
            weights[pos] = desc.effective_weight();
            delays[pos] = desc.delay;
            polarities[pos] = desc.polarity;
            cursor[src] += 1;
        }

        Ok(Self {
            neuron_count,
            row_ptr,
            targets,
            weights,
            delays,
            polarities,
        })
    }

    /// Number of neurons in the graph.
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Total number of synapses (non-zero entries).
    pub fn synapse_count(&self) -> usize {
        self.targets.len()
    }

    /// Sparsity: fraction of the N×N matrix that is zero.
    pub fn sparsity(&self) -> f32 {
        let total = self.neuron_count * self.neuron_count;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.synapse_count() as f32 / total as f32)
    }

    /// Iterate over all outgoing synapses from `source`.
    /// Returns `(target, weight, delay, polarity)` tuples.
    pub fn outgoing(
        &self,
        source: usize,
    ) -> impl Iterator<Item = (NeuronId, f32, DelayTicks, Polarity)> + '_ {
        let start = self.row_ptr[source];
        let end = self.row_ptr[source + 1];
        (start..end).map(move |i| (self.targets[i], self.weights[i], self.delays[i], self.polarities[i]))
    }

    /// Get the out-degree of a neuron.
    pub fn out_degree(&self, neuron: usize) -> usize {
        self.row_ptr[neuron + 1] - self.row_ptr[neuron]
    }

    /// Maximum delay present in the graph.
    pub fn max_delay(&self) -> DelayTicks {
        self.delays.iter().copied().max().unwrap_or(0)
    }

    /// Mean out-degree across all neurons.
    pub fn mean_degree(&self) -> f32 {
        if self.neuron_count == 0 {
            return 0.0;
        }
        self.synapse_count() as f32 / self.neuron_count as f32
    }

    /// Raw access to CSR arrays for GPU upload.
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u16>) {
        let row_ptr: Vec<u32> = self.row_ptr.iter().map(|&x| x as u32).collect();
        let targets: Vec<u32> = self.targets.iter().map(|&x| x).collect();
        (row_ptr, targets, self.weights.clone(), self.delays.clone())
    }

    /// Row pointer slice (for external iteration).
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Target neuron slice.
    pub fn targets(&self) -> &[NeuronId] {
        &self.targets
    }

    /// Weight slice.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Delay slice.
    pub fn delays_slice(&self) -> &[DelayTicks] {
        &self.delays
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_no_synapses() {
        let g = SynapticGraph::new(100);
        assert_eq!(g.neuron_count(), 100);
        assert_eq!(g.synapse_count(), 0);
        assert!((g.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_descriptors_roundtrip() {
        let descs = vec![
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.9,
                delay: 3,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 0,
                target: 2,
                weight: 0.15,
                delay: 1,
                polarity: Polarity::Inhibitory,
            },
            SynapseDescriptor {
                source: 1,
                target: 0,
                weight: 0.5,
                delay: 2,
                polarity: Polarity::Excitatory,
            },
        ];

        let graph = SynapticGraph::from_descriptors(3, &descs).unwrap();
        assert_eq!(graph.neuron_count(), 3);
        assert_eq!(graph.synapse_count(), 3);

        // Neuron 0 has 2 outgoing
        assert_eq!(graph.out_degree(0), 2);
        // Neuron 1 has 1 outgoing
        assert_eq!(graph.out_degree(1), 1);
        // Neuron 2 has 0 outgoing
        assert_eq!(graph.out_degree(2), 0);

        // Check inhibitory weight is negative
        let edges: Vec<_> = graph.outgoing(0).collect();
        let inh_edge = edges.iter().find(|(_, _, _, p)| *p == Polarity::Inhibitory).unwrap();
        assert!(inh_edge.1 < 0.0);
    }

    #[test]
    fn max_delay_reports_correctly() {
        let descs = vec![
            SynapseDescriptor {
                source: 0,
                target: 1,
                weight: 0.5,
                delay: 7,
                polarity: Polarity::Excitatory,
            },
            SynapseDescriptor {
                source: 1,
                target: 0,
                weight: 0.5,
                delay: 3,
                polarity: Polarity::Excitatory,
            },
        ];
        let graph = SynapticGraph::from_descriptors(2, &descs).unwrap();
        assert_eq!(graph.max_delay(), 7);
    }

    #[test]
    fn out_of_bounds_source_rejected() {
        let descs = vec![SynapseDescriptor {
            source: 5,
            target: 0,
            weight: 0.5,
            delay: 1,
            polarity: Polarity::Excitatory,
        }];
        assert!(SynapticGraph::from_descriptors(3, &descs).is_err());
    }
}
