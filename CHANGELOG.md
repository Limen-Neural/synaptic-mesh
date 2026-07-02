# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **License**: Switched from `GPL-3.0-or-later` to dual `MIT OR Apache-2.0`.
  This aligns `synaptic-mesh` with the rest of the Limen-Neural SNN stack and
  permits broader downstream use. Old `LICENSE` (GPL-3.0) removed; replaced by
  `LICENSE-MIT` and `LICENSE-APACHE-2.0`. SPDX identifier
  `MIT OR Apache-2.0` added to all source files.

### Added

- **CI**: GitHub Actions workflow (`.github/workflows/ci.yml`) running
  `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, and
  `cargo test` on every push/PR to `main`.

### Fixed

- **Test**: `mesh::tests::layered_mesh_feed_forward` — the test now models
  multi-hop propagation by thresholding received currents back into spikes
  (matching the documented one-hop `propagate()` semantics).

### Internal

- Resolved ~40 clippy warnings (`RangeInclusive::contains`, iterator idioms,
  derivable impls, redundant borrows).
- Fixed `cargo fmt` formatting (module ordering, debug_assert wrapping).

## [0.2.0] - 2026-05-28

### Features Added

- Neuromodulatory adaptation in `ChannelRouter` — `NeuromodState` (cortisol,
  dopamine, serotonin), `route_modulated()`, `apply_plasticity()`, per-channel
  fatigue, use-it-or-lose-it plasticity (PR #8).
- Generic `ChannelRouter` replacing domain-specific `AhlRouter` (PR #7).
- CSR sparse synaptic maps, topology generators (small-world, scale-free,
  random, layered), temporal delay ring-buffer, Dale's law wiring.
