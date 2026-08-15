<p align="center">
  <img src="docs/logo.png" width="220" alt="synaptic-mesh">
</p>

<h1 align="center">synaptic-mesh</h1>
<p align="center">SNN wiring, topology generation, and temporal delay infrastructure</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-informational" alt="0.1.0 (not yet on crates.io)">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="MIT OR Apache-2.0">
</p>

---

`synaptic-mesh` manages the wiring, topology, and temporal delays between neurons in spiking neural networks. It provides high-performance, deterministic graph generators (Small-World, Scale-Free, Layered) and a temporal delay infrastructure that simulates realistic axonal propagation.

## Core Capabilities

- **Topology Generators** — Deterministic generation of Erdős–Rényi (random), Watts–Strogatz (small-world), Barabási–Albert (scale-free), and Layered feed-forward topologies.
- **Temporal Propagation** — Per-synapse axonal delays stored alongside weights. Spikes are delivered at the correct future tick via a high-performance ring-buffer queue.
- **Biologically Inspired Wiring** — Support for Dale's Law (fixed neuron polarity) and position-based distance-dependent connectivity.
- **Sparse Synaptic Map (CSR)** — Compressed Sparse Row format for memory-efficient weight storage (20× reduction for sparse networks).
- **Generic Channel Router** — A configurable multi-channel SNN router using neuromodulatory neurons for sparse signal classification.

## Installation

Not yet on crates.io. Until a crates.io release is available, depend on git:

```toml
synaptic-mesh = { git = "https://github.com/Limen-Neural/synaptic-mesh" }
```

After the `v0.1.0` tag exists you can pin it with `tag = "v0.1.0"`.

## Quick Start: Building a Mesh

```rust
use synaptic_mesh::topology::generators::generate_small_world;
use synaptic_mesh::mesh::SynapticMesh;

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

## Generic Channel Router

The crate includes `ChannelRouter`, a configurable multi-channel SNN router that integrates signal pulses across neuromodulatory neurons to produce sparse activation masks.

```rust
use synaptic_mesh::{ChannelRouter, RouterConfig};

// Default 3-channel router
let mut router = ChannelRouter::new();
let decision = router.route(&[0.8, 0.2, 0.1]);

// decision.active_channels → [0]
// decision.firing_rates   → [0.75, 0.12, 0.0]
```

### Configurable Channel Count

```rust
use synaptic_mesh::{ChannelRouter, RouterConfig};

let config = RouterConfig {
    channel_count: 8,
    ..RouterConfig::default()
};
let mut router = ChannelRouter::with_config(config);
let decision = router.route(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
// decision.active_channels → [3]
```

## Adaptation-Aware Routing

- **Neuron State Snapshots** — Per-neuron adaptation and error tracking for dynamic routing decisions.
- **Routing Policies** — Configurable scoring equations that balance spike activity, adaptation penalties, and error bonuses.

## Crate Boundary

`synaptic-mesh` does **not** depend on the separate [`neuromod`](https://github.com/Limen-Neural/neuromod) crate. The two crates are kept independent so each can evolve without coupling:

| Crate | Owns |
|-------|------|
| **neuromod** | Canonical neuron models — LIF, Izhikevich, Hodgkin-Huxley, GIF, FitzHugh-Nagumo, Lapicque |
| **synaptic-mesh** | Topology, wiring, delay infrastructure, CSR sparse maps, `ChannelRouter`, and its router-internal `NeuromodNeuron` (NIF) integration primitive |

`NeuromodNeuron` lives in [`router`](src/router.rs) — it's a router-internal integration primitive, not a general-purpose neuron model.

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

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option. Contributions intentionally submitted for inclusion in this
crate by you, as defined in the Apache-2.0 license, shall be dual-licensed as
above, without any additional terms or conditions.
