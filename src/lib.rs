//! # synaptic-mesh
//!
//! Manages the wiring, topology, and temporal delays between neurons in the
//! Spikenaut SNN ecosystem.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`topology`] | Network graph construction — CSR adjacency with delay & polarity metadata, plus deterministic generators (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, layered) |
//! | [`delay`] | Temporal delay infrastructure — ring-buffer spike queues for tick-aligned delivery with configurable axonal propagation delays |
//! | [`mesh`] | [`SynapticMesh`] orchestrator — the top-level struct owning topology + delays, provides `propagate()` for spike → current conversion |
//! | [`sparse`] | Compressed Sparse Row (CSR) synaptic maps for GPU-optimized weight matrices |
//! | [`router`] | AHL (Anti-Hallucination Layer) router — generic SNN router using neuromodulatory neurons |
//! | [`neuromod`] | Neuromodulatory (NIF) neuron model with gain control |
//!
//! ## Quick start
//!
//! ```rust
//! use synaptic_mesh::topology::generate_small_world;
//! use synaptic_mesh::mesh::SynapticMesh;
//!
//! // Build a 256-neuron small-world network with delays up to 5 ticks
//! let graph = generate_small_world(256, 6, 0.2, 5, 0.2).unwrap();
//! let mut mesh = SynapticMesh::new(graph);
//!
//! // Each tick: provide spike vector → receive delayed synaptic currents
//! let mut spikes = vec![false; 256];
//! spikes[0] = true; // neuron 0 fires
//!
//! let currents = mesh.propagate(&spikes).unwrap();
//! // currents[i] = total incoming synaptic current at neuron i this tick
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  Source spike vector  [bool; N]                   │
//! └────────────────┬─────────────────────────────────┘
//!                  │
//!        ┌─────────▼─────────┐
//!        │   SynapticGraph   │  CSR adjacency + delays + polarities
//!        │   (topology)      │  Generators: random, small-world,
//!        │                   │  scale-free, layered
//!        └─────────┬─────────┘
//!                  │  per-synapse: (target, weight, delay)
//!        ┌─────────▼─────────┐
//!        │  SpikeDelayBuffer │  Ring-buffer delay queue
//!        │   (delay)         │  inject() → advance() → drain()
//!        └─────────┬─────────┘
//!                  │  tick-aligned delivery
//!        ┌─────────▼─────────┐
//!        │  Synaptic current │  Vec<f32> of length N
//!        │  per target neuron│
//!        └───────────────────┘
//! ```
//!
//! ## References
//!
//! - Lapicque, L. (1907). *Recherches quantitatives sur l'excitation électrique des
//!   nerfs traitée comme une polarisation.* Journal de Physiologie et de Pathologie
//!   Générale, 9, 620–635.
//! - Stein, R. B. (1967). *Some models of neuronal variability.* Biophysical
//!   Journal, 7(1), 37–68.
//!
//! **Dale's law:**
//! - Dale, H. H. (1935). *Pharmacology and Nerve-endings.* Proceedings of the
//!   Royal Society of Medicine, 28(3), 319–332.
//!
//! **Network topology:**
//! - Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of 'small-world'
//!   networks.* Nature, 393, 440–442.
//! - Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random networks.*
//!   Science, 286, 509–512.
//!
//! **STDP / Hebbian plasticity:**
//! - Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
//! - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
//!   hippocampal neurons: dependence on spike timing, synaptic strength, and
//!   postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464–10472.
//!
//! **Winner-take-all lateral inhibition:**
//! - Maass, W. (2000). On the computational power of winner-take-all.
//!   *Neural Computation*, 12(11), 2519–2535.

// ── New modules: wiring, topology, delays ─────────────────────────────────────
pub mod delay;
pub mod error;
pub mod mesh;
pub mod topology;
pub mod types;
pub mod neuromod;

// ── Existing modules: router + sparse maps ────────────────────────────────────
pub mod router;
pub mod sparse;

// ── Public re-exports ─────────────────────────────────────────────────────────

// New mesh infrastructure
pub use delay::SpikeDelayBuffer;
pub use error::{MeshError, Result};
pub use mesh::SynapticMesh;
pub use topology::SynapticGraph;
pub use types::{
    ConnectionModel, DelayModel, DelayTicks, NeuronId, Polarity, SynapseDescriptor, TopologyConfig,
};

pub use neuromod::NeuromodNeuron;
// Existing router + sparse exports (backward compatible)
pub use router::{AhlRouter, RoutingDecision, AHL_NUM_CHANNELS};
pub use sparse::{
    RoutingPolicy, SparseSynapticMap, SparseSynapticMapBuilder, Synapse, TelemetrySnapshot,
};

#[cfg(test)]
mod tests;
