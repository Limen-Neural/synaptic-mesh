<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">synaptic-mesh</h1>
<p align="center">SNN wiring, topology generation, and temporal delay infrastructure</p>

<p align="center">
 <img src="https://img.shields.io/crates/v/synapse-router" alt="crates.io"></a>
  <a href="https://docs.rs/synapse-router"><img src="https://docs.rs/synapse-router/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

`synaptic-mesh` manages the wiring, topology, and temporal delays between neurons in the Spikenaut SNN ecosystem. It provides high-performance, deterministic graph generators (Small-World, Scale-Free, Layered) and a temporal delay infrastructure that simulates realistic axonal propagation.

## Core Capabilities

- **Topology Generators** — Deterministic generation of Erdős–Rényi (random), Watts–Strogatz (small-world), Barabási–Albert (scale-free), and Layered feed-forward topologies.
- **Temporal Propation** — Per-synapse axonal delays stored alongside weights. Spikes are delivered at the correct future tick via a high-performance ring-buffer queue.
- **Biologically Inspired Wiring** — Support for Dale's Law (fixed neuron polarity) and position-based distance-dependent connectivity.
- **Sparse Synaptic Map (CSR)** — Compressed Sparse Row format for memory-efficient weight storage (20× reduction for sparse networks).
- **AHL Domain Router** — A specialized consumer of synaptic wiring used for sparse Anti-Hallucination Layer classification in LLM pipelines.

## Installation

```toml
synapse-router = "0.2"
```

## Quick Start: Building a Mesh

```rust
use synapse_router::topology::generators::generate_small_world;
use synapse_router::mesh::SynapticMesh;

// 1. Build a 1024-neuron small-world network with delays up to 10 ticks
// (N=1024, k=6 neighbors, beta=0.1 rewiring, max_delay=10, inh_fraction=0.2)
let graph = generate_small_world(1024, 6, 0.1, 10, 0.2).unwrap();

// 2. Wrap in a Mesh orchestrator that manages temporal state
let mut mesh = SynapticMesh::new(graph);

// 3. Each tick: provide current spikes -> receive time-delayed synaptic currents
let mut spikes = vec![false; 1024];
spikes[0] = true; // neuron 0 fires

let currents = mesh.propagate(&spikes).unwrap();
// currents[i] = total incoming synaptic current at neuron i this tick,
// potentially including delayed spikes from previous ticks.
```

## Topology Generation

`synaptic-mesh` provides several deterministic models for growing network graphs. All generators use golden-ratio fractional hashing for reproducibility across runs without external RNG dependencies.

| Model | Generator | Best For |
|-------|-----------|----------|
| **Small-World** | `generate_small_world` | Local clustering with short path lengths (mimics cortical connectivity). |
| **Scale-Free** | `generate_scale_free` | Networks with "hubs" following a power-law degree distribution. |
| **Random** | `generate_random` | Erdős–Rényi random graphs for baseline comparisons. |
| **Layered** | `generate_layered` | Classical feed-forward structures (Input -> Hidden -> Output). |

## Temporal Delays & Spike Propagation

In biological networks, spikes do not arrive instantly. `synaptic-mesh` implements a temporal logic layer using a **Ring-Buffer Delay Queue**:

1.  Each synapse in the `SynapticGraph` stores a `DelayTicks` value.
2.  When a neuron fires, its spike is projected through its outgoing synapses.
3.  The `SpikeDelayBuffer` schedules delivery at `current_tick + delay`.
4.  At each tick, `propagate()` drains the current slot and returns the accumulated currents.

This enables complex temporal dynamics like polychronization and coincidence detection.

## Anti-Hallucination Layer (AHL) Router

The crate includes `AhlRouter`, a specialized application that uses a small SNN for text-domain classification.

```rust
use synapse_router::AhlRouter;

let mut router = AhlRouter::new();
let decision = router.route("solve the differential equation dy/dx = sin(x)");

// decision.active_domains → [Mathematics]
// decision.firing_rates   → [0.0, 0.87, 0.0]
```

## SAAQ Adaptation & Ballast-Lab Integration

- **SAAQ Telemetry** — Adaptation-aware routing that steers traffic away from exhausted neurons.
- **Ballast-Lab Loop** — Export CSV telemetry for Julia symbolic regression to discover optimal routing policy equations:
  ```rust
  let csv = router.telemetry_csv(&telemetry, &firing_rates);
  // Feed into SR.jl to discover optimal α·spikes - β·adaptation coefficients
  ```

## Architecture

```text
┌──────────────────────────────────────────────────┐
│  Source spike vector  [bool; N]                   │
└────────────────┬─────────────────────────────────┘
                 │
       ┌─────────▼─────────┐
       │   SynapticGraph   │  CSR adjacency + delays + polarities
       │   (topology)      │  Generators: random, small-world, etc.
       └─────────┬─────────┘
                 │  per-synapse: (target, weight, delay)
       ┌─────────▼─────────┐
       │  SpikeDelayBuffer │  Ring-buffer delay queue
       │   (delay)         │  inject() → advance() → drain()
       └─────────┬─────────┘
                 │  tick-aligned delivery
       ┌─────────▼─────────┐
       │  Synaptic current │  Vec<f32> of length N
       │  per target neuron│
       └───────────────────┘
```

## References

**Network Topology & Dynamics:**
- Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of 'small-world' networks.* Nature.
- Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random networks.* Science.
- Dale, H. H. (1935). *Pharmacology and Nerve-endings.*

**SNN Logic:**
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications... *Journal of Neuroscience*.
- Maass, W. (2000). On the computational power of winner-take-all. *Neural Computation*.

## License

GPL-3.0-or-later
