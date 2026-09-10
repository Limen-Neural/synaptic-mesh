# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Packaging**: `documentation = "https://docs.rs/synaptic-mesh"` and an
  explicit `readme = "README.md"` in `Cargo.toml` (issue #34).
- **Tests**: `tests/propagate_contract.rs` — deterministic consumer contract
  for `SynapticMesh::propagate` over a fixed four-neuron graph, asserting the
  destination, sign, magnitude, and delivery tick of every spike, plus
  co-arrival summing, one-hop semantics, and replay after `reset()`
  (issue #52).
- **Docs**: `SynapticMesh::propagate` rustdoc now states the delivery
  contract explicitly and carries a runnable minimal example (issue #52).
- **CI**: `package` job running `cargo package --locked`, asserting the
  packaged file list against an allowlist of consumer-relevant files, and
  building the unpacked `.crate` outside the repository so an over-eager
  `exclude` or a repo-only build dependency fails CI instead of crates.io
  (issue #36).

### Changed

- **MSRV**: Rust pin raised from **1.97.1** to **1.98.1** in `Cargo.toml`
  `rust-version`, `rust-toolchain.toml`, and CI (both the toolchain install
  and the `cargo-deny` action) ahead of the first crates.io publish
  (issue #51).
- **CI**: the MSRV pin-agreement check now compares *every* `toolchain:` /
  `rust-version:` pin in `.github/workflows/ci.yml` against `Cargo.toml`,
  instead of only the first toolchain install, so a partially bumped or
  newly added job fails the check (issue #51).
- **Version**: bumped to **0.3.0**, the first version published to crates.io
  (issue #34).
- **Packaging**: `exclude` now also drops `/.github/`, `/qodana.yaml`, and
  `/REVIEW.md` from the published `.crate` (issue #34), plus the remaining
  dev-tooling files `/.codacy.yml`, `/.devin/`, `/.gitignore`, `/AGENTS.md`,
  `/deny.toml`, and `/rust-toolchain.toml` (issue #49). The packaged crate is
  now source, Cargo manifests, licenses, README, and CHANGELOG only.
- **Docs**: README leads with the crates.io install path
  (`synaptic-mesh = "0.3"`), marks the git dependency as the bleeding-edge
  alternative, and describes 0.3.0 as experimental pre-1.0. Repo-relative
  logo and `REVIEW.md` links are absolute so they resolve on crates.io and
  docs.rs, where those files are not packaged (issue #34).

## [0.2.0] - 2026-09-06

Bridges v0.1.0 (API solidify) and v0.3.0 (first crates.io publish). Hardens
the build/test/bench setup ahead of packaging for publish.

### Added

- **Cargo profiles**: explicit `[profile.dev]`, `[profile.test]`,
  `[profile.release]`, and `[profile.bench]` in `Cargo.toml`, each tuned and
  documented with a rationale comment (issue #40).
- **CI**: workflow now builds under `release` and compiles under `bench`
  in addition to the existing `dev` build and `test` run, so a broken or
  reverted profile fails CI instead of only surfacing locally (issue #41).
- **CI**: `cargo-deny` job checking advisories, license allowlist (dual
  MIT/Apache-2.0), bans, and sources ahead of the crates.io publish
  (issue #42).
- **Docs**: "Build profiles" section in `REVIEW.md` listing each profile,
  the command that uses it, and why it's configured that way, cross-linked
  from `Cargo.toml` and `README.md` (issue #43).

[0.2.0]: https://github.com/Limen-Neural/synaptic-mesh/compare/v0.1.0...v0.2.0

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
- MSRV pin **1.97.1** in `Cargo.toml` `rust-version`, `rust-toolchain.toml`,
  and CI (`dtolnay/rust-toolchain` + pin-agreement check).
- `PartialEq` on `SynapticGraph` and `SynapseDescriptor`, plus a JSON
  serde round-trip test.
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

[Unreleased]: https://github.com/Limen-Neural/synaptic-mesh/compare/v0.2.0...HEAD
[0.1.0]: https://github.com/Limen-Neural/synaptic-mesh/releases/tag/v0.1.0
