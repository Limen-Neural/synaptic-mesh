//! Compressed Sparse Row (CSR) synaptic map for GPU-optimized execution.
//!
//! Replaces dense $N \times N$ weight matrices with adjacency lists stored in CSR format,
//! reducing VRAM pressure and enabling warp-optimized shared memory pulls.
//!
//! # Layout
//!
//! ```text
//! row_ptr:    [0, 3, 7, 10, ...]        — start index in col_indices/values per neuron
//! col_indices:[0, 5, 12, 1, 3, 8, 15, ...] — target neuron indices
//! values:     [0.9, -0.15, 0.3, ...]     — synaptic weights
//! ```
//!
//! For a 2048-neuron network with ~5% connectivity, this reduces storage from
//! ~16 MB (dense f32) to ~800 KB (sparse), a 20× reduction.

use serde::{Deserialize, Serialize};

/// A single sparse synaptic connection.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Synapse {
    /// Target neuron index (column in the weight matrix).
    pub target: u16,
    /// Synaptic weight strength.
    pub weight: f32,
}

/// Compressed Sparse Row representation of the synaptic weight matrix.
///
/// Generic over `N` (number of neurons) to support both small channel routers
/// and full 2048-neuron routing fabrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseSynapticMap<const N: usize> {
    /// Row pointers: `row_ptr[i]` gives the start index in `col_indices`/`values`
    /// for neuron `i`. Length is `N + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices: target neuron for each non-zero entry.
    pub col_indices: Vec<u16>,
    /// Non-zero weight values.
    pub values: Vec<f32>,
}

impl<const N: usize> Default for SparseSynapticMap<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> SparseSynapticMap<N> {
    /// Create an empty sparse synaptic map with `N` neurons.
    pub fn new() -> Self {
        Self {
            row_ptr: vec![0; N + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Build from a dense weight matrix (for migration from dense representations).
    pub fn from_dense(matrix: &[[f32; N]; N], sparsity_threshold: f32) -> Self {
        let mut row_ptr = Vec::with_capacity(N + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for row in matrix.iter() {
            for (col, &w) in row.iter().enumerate() {
                if w.abs() > sparsity_threshold {
                    col_indices.push(col as u16);
                    values.push(w);
                }
            }
            row_ptr.push(col_indices.len());
        }

        Self {
            row_ptr,
            col_indices,
            values,
        }
    }

    /// Build from explicit adjacency lists.
    pub fn from_adjacency(adjacency: &[Vec<Synapse>]) -> Self {
        assert!(
            adjacency.len() == N,
            "adjacency length must match neuron count"
        );

        let mut row_ptr = Vec::with_capacity(N + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for synapses in adjacency {
            for syn in synapses {
                col_indices.push(syn.target);
                values.push(syn.weight);
            }
            row_ptr.push(col_indices.len());
        }

        Self {
            row_ptr,
            col_indices,
            values,
        }
    }

    /// Get all synapses for a given neuron (row).
    pub fn get_row(&self, row: usize) -> impl Iterator<Item = (u16, f32)> + '_ {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        (start..end).map(move |i| (self.col_indices[i], self.values[i]))
    }

    /// Get the weight from neuron `row` to neuron `col`. Returns 0.0 if not connected.
    pub fn get_weight(&self, row: usize, col: usize) -> f32 {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for i in start..end {
            if self.col_indices[i] as usize == col {
                return self.values[i];
            }
        }
        0.0
    }

    /// Update a single synaptic weight. Creates the connection if it doesn't exist
    /// (when weight exceeds threshold) or removes it (when weight drops below).
    pub fn set_weight(&mut self, row: usize, col: usize, weight: f32, sparsity_threshold: f32) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        // Search for existing connection
        for i in start..end {
            if self.col_indices[i] as usize == col {
                if weight.abs() > sparsity_threshold {
                    self.values[i] = weight;
                } else {
                    // Remove: shift everything after
                    self.col_indices.remove(i);
                    self.values.remove(i);
                    for r in (row + 1)..=N {
                        self.row_ptr[r] -= 1;
                    }
                }
                return;
            }
        }

        // Add new connection if significant
        if weight.abs() > sparsity_threshold {
            self.col_indices.insert(end, col as u16);
            self.values.insert(end, weight);
            for r in (row + 1)..=N {
                self.row_ptr[r] += 1;
            }
        }
    }

    /// Number of non-zero synapses.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio: fraction of zero entries in the full $N \times N$ matrix.
    pub fn sparsity(&self) -> f32 {
        let total = N * N;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / total as f32)
    }

    /// Convert back to dense matrix (for debugging / interoperability).
    pub fn to_dense(&self) -> [[f32; N]; N] {
        let mut matrix = [[0.0; N]; N];
        for (row, row_data) in matrix.iter_mut().enumerate().take(N) {
            for (col, w) in self.get_row(row) {
                row_data[col as usize] = w;
            }
        }
        matrix
    }

    /// Export as GPU-ready flat arrays for kernel launch.
    /// Returns (row_ptr, col_indices, values) as owned vectors.
    pub fn to_gpu_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let row_ptr: Vec<u32> = self.row_ptr.iter().map(|&x| x as u32).collect();
        let col_indices: Vec<u32> = self.col_indices.iter().map(|&x| x as u32).collect();
        (row_ptr, col_indices, self.values.clone())
    }
}

/// Builder for constructing a sparse synaptic map with a fluent API.
pub struct SparseSynapticMapBuilder<const N: usize> {
    adjacency: Vec<Vec<Synapse>>,
    default_weight: f32,
    sparsity_threshold: f32,
}

impl<const N: usize> SparseSynapticMapBuilder<N> {
    pub fn new() -> Self {
        Self {
            adjacency: vec![Vec::new(); N],
            default_weight: 0.0,
            sparsity_threshold: 0.01,
        }
    }

    /// Set the default weight for self-connections.
    pub fn with_self_weight(mut self, weight: f32) -> Self {
        self.default_weight = weight;
        self
    }

    /// Set the sparsity threshold below which connections are pruned.
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Add a synaptic connection.
    pub fn connect(mut self, from: usize, to: usize, weight: f32) -> Self {
        if weight.abs() > self.sparsity_threshold {
            self.adjacency[from].push(Synapse {
                target: to as u16,
                weight,
            });
        }
        self
    }

    /// Add self-connections for all neurons with the default weight.
    pub fn with_self_connections(self) -> Self {
        let w = self.default_weight;
        let mut result = self;
        for i in 0..N {
            result = result.connect(i, i, w);
        }
        result
    }

    /// Add lateral inhibition between all pairs with the given weight.
    pub fn with_lateral_inhibition(self, weight: f32) -> Self {
        let mut result = self;
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    result = result.connect(i, j, weight);
                }
            }
        }
        result
    }

    /// Build the CSR map.
    pub fn build(self) -> SparseSynapticMap<N> {
        SparseSynapticMap::from_adjacency(&self.adjacency)
    }
}

impl<const N: usize> Default for SparseSynapticMapBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-neuron state snapshot for adaptation-aware routing.
///
/// Captures the state needed for dynamic routing: per-neuron
/// adaptation levels, spike counts, and error estimates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronStateSnapshot {
    /// Per-neuron adaptation state (0.0 = fresh, 1.0 = fully adapted/exhausted).
    pub adaptation: Vec<f32>,
    /// Per-neuron spike count from the last routing window.
    pub spike_counts: Vec<u32>,
    /// Estimated error per neuron (e.g. quantization error from external calibration).
    pub error: Vec<f32>,
    /// Global routing step index.
    pub step: u64,
}

/// Backward-compatible alias for [`NeuronStateSnapshot`].
///
/// Deprecated: use [`NeuronStateSnapshot`] instead.
pub type TelemetrySnapshot = NeuronStateSnapshot;

impl NeuronStateSnapshot {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            adaptation: vec![0.0; num_neurons],
            spike_counts: vec![0; num_neurons],
            error: vec![0.0; num_neurons],
            step: 0,
        }
    }

    /// Get a routing penalty for a neuron based on its adaptation state.
    /// Higher adaptation → higher penalty → less likely to be selected.
    pub fn adaptation_penalty(&self, neuron: usize, alpha: f32) -> f32 {
        alpha * self.adaptation.get(neuron).copied().unwrap_or(0.0)
    }

    /// Get a routing bonus for a neuron based on low error.
    /// Lower error → higher bonus → preferred for routing.
    pub fn error_bonus(&self, neuron: usize, beta: f32) -> f32 {
        let err = self.error.get(neuron).copied().unwrap_or(0.0);
        beta * (1.0 - err.min(1.0))
    }
}

/// A routing policy equation for scoring neurons.
///
/// The policy computes a routing score for each neuron:
/// `score = α·spikes - β·adaptation - γ·error + δ·base_weight`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingPolicy {
    /// Weight for spike count contribution.
    pub alpha: f32,
    /// Weight for adaptation penalty.
    pub beta: f32,
    /// Weight for error penalty.
    pub gamma: f32,
    /// Weight for base synaptic strength.
    pub delta: f32,
    /// Minimum score threshold to activate a neuron.
    pub threshold: f32,
    /// Human-readable description of the policy origin.
    pub description: String,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.3,
            delta: 0.8,
            threshold: 0.1,
            description: "default policy".to_string(),
        }
    }
}

impl RoutingPolicy {
    /// Compute the routing score for a single neuron.
    pub fn score(&self, neuron: usize, snapshot: &NeuronStateSnapshot, base_weight: f32) -> f32 {
        let spikes = snapshot.spike_counts.get(neuron).copied().unwrap_or(0) as f32;
        let adapt = snapshot.adaptation.get(neuron).copied().unwrap_or(0.0);
        let err = snapshot.error.get(neuron).copied().unwrap_or(0.0);

        self.alpha * spikes - self.beta * adapt - self.gamma * err + self.delta * base_weight
    }

    /// Check if a neuron should be activated based on its score.
    pub fn should_activate(&self, score: f32) -> bool {
        score >= self.threshold
    }
}
