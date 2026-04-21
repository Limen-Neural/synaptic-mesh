# Agent Notes

This file contains workflow and orientation notes for AI agents working on this repository.

## Repo Map

- `src/lib.rs`: crate root, re-exports from all modules
- `src/types.rs`: core type aliases (NeuronId, DelayTicks, Polarity, SynapseDescriptor) and config structs
- `src/error.rs`: `MeshError` unified error enum
- `src/topology/`: network graph construction and wiring rules
  - `graph.rs`: `SynapticGraph` — CSR adjacency with delay + polarity metadata
  - `generators.rs`: deterministic topology generators (Erdős–Rényi, Watts–Strogatz, Barabási–Albert, layered)
  - `wiring_rules.rs`: Dale's law polarity assignment, distance-based delay assignment
- `src/delay/`: temporal delay infrastructure
  - `ring_buffer.rs`: `SpikeDelayBuffer` — ring-buffer delay queue for tick-aligned spike delivery
- `src/mesh.rs`: `SynapticMesh` — top-level orchestrator owning graph + delays, provides `propagate()` 
- `src/router.rs`: `AhlRouter` — AHL (Anti-Hallucination Layer) domain router (consumer of wiring)
- `src/sparse.rs`: `SparseSynapticMap` CSR format, `TelemetrySnapshot`, `RoutingPolicy`
- `src/tests.rs`: test suite for router + sparse modules

## Entry Order

- Start at `src/lib.rs` for the exported crate surface.
- Read `src/types.rs` for core type definitions.
- Read `src/topology/graph.rs` before `generators.rs` or `wiring_rules.rs`.
- Read `src/delay/ring_buffer.rs` for the temporal delay model.
- Read `src/mesh.rs` for the orchestrator that ties everything together.

## Design Principles

- **Determinism**: all generators use index-based pseudo-random hashing (golden-ratio fractional), no external RNG. Same inputs always produce the same topology.
- **Dale's Law**: neuron polarity is per-neuron (all outgoing synapses share polarity). First `inh_fraction × N` neurons are inhibitory.
- **Temporal Delays**: per-synapse axonal delays stored in CSR alongside weights. Ring-buffer delivers spikes at correct future tick.
- **Backward Compatibility**: existing `AhlRouter`, `SparseSynapticMap`, `TelemetrySnapshot`, `RoutingPolicy` all preserved.

## Workflow Policy

- All `synaptic-mesh` work stays in `/home/raulmc/synaptic-mesh`.
- Sister crate: `corinth-canal` at `/home/raulmc/corinth-canal` (inspiration for architecture patterns).
- Run `cargo check`, `cargo test` before closing substantial Rust changes.

## Repository Context

- **Repo**: `Limen-Neural/synapse-router`
- **Main branch**: `main`
- **Language**: Rust (edition 2024)
- **Key concepts**: SNN wiring, topology generation, axonal delays, CSR sparse maps, Dale's law
