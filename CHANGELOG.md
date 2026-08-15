# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-14

First versioned history. Nothing before this tag was published or tagged.

### Added

- Neuromodulatory adaptation in `ChannelRouter` — `NeuromodState` (cortisol,
  dopamine, serotonin), `route_modulated()`, `apply_plasticity()`, per-channel
  fatigue, use-it-or-lose-it plasticity.
- Generic `ChannelRouter` with configurable channel count.
- CSR sparse synaptic maps, topology generators (small-world, scale-free,
  random, layered), temporal delay ring-buffer, Dale's law wiring.
- **CI**: GitHub Actions workflow (`.github/workflows/ci.yml`) running
  `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, and
  `cargo test` on every push/PR to `main`.
- `inhibitory_fraction` range validation to all topology generators
  (`generate_random`, `generate_small_world`, `generate_scale_free`,
  `generate_layered`) — rejects values outside `[0, 1]`.

### Changed

- **License**: Dual `MIT OR Apache-2.0`. SPDX identifier added to all source
  files.
- `apply_dale_polarity` (`topology::wiring_rules`) returns
  `Result<Vec<Polarity>, MeshError>` instead of `Vec<Polarity>`. Out-of-range
  `inhibitory_fraction` values (< 0 or > 1) are rejected with
  `MeshError::InvalidConfig`.

### Removed

- **`AhlRouter`** — use `ChannelRouter` with `RouterConfig::default()`.
- **`AHL_NUM_CHANNELS`** — default channel count is `RouterConfig::default().channel_count` (3).
- **`TelemetrySnapshot`** — use `NeuronStateSnapshot`.
- **`NeuronStateSnapshot::quant_bonus`** — use `NeuronStateSnapshot::error_bonus`.
- **`NeuronStateSnapshot::quant_error`** — use the `error` field.
  The serde `alias = "quant_error"` on `error` is kept so older snapshots
  still deserialize.

### Fixed

- **Test**: `mesh::tests::layered_mesh_feed_forward` — the test now models
  multi-hop propagation by thresholding received currents back into spikes
  (matching the documented one-hop `propagate()` semantics).
- **Underflow**: `SynapticGraph::validate_descriptor_indices` used
  `neuron_count - 1` which underflows on `neuron_count = 0`. Now uses
  `saturating_sub(1)`.
- **Consistency**: `apply_feedback` and `sync_baseline_after_feedback` now
  consistently use `self.config.channel_count` (matching all other methods)
  instead of `self.neurons.len()`. `apply_feedback` retains a double-guard
  for Codacy HIGH RISK out-of-bounds protection.
- **Defensive**: `sync_baseline_after_feedback` inner-row length check
  converted to `debug_assert!` documenting the call-ordering invariant
  (`ensure_neuromod_state_synced` rebuilds `baseline_weights` beforehand).
- Zero-size layer validation in `generate_layered` now rejects `&[0, 5]`
  via `layer_sizes.contains(&0)` with a clearer error message.

### Internal

- Resolved clippy warnings (`RangeInclusive::contains`, iterator idioms,
  derivable impls, redundant borrows).
- Fixed `cargo fmt` formatting (module ordering, debug_assert wrapping).

[Unreleased]: https://github.com/Limen-Neural/synaptic-mesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Limen-Neural/synaptic-mesh/releases/tag/v0.1.0
