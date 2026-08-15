# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, or public APIs.

## MSRV pin rule

`Cargo.toml` `rust-version`, `rust-toolchain.toml` `channel`, and the
`toolchain:` string in `.github/workflows/ci.yml` must stay **identical**
(currently **1.97.1**). CI fails if they drift (issue #35 / LIM-1042).

To bump MSRV:

1. Set the new version in Cargo.toml, rust-toolchain.toml, and ci.yml.
2. Run the mandatory commands below on that toolchain
   (`rustup run <ver> cargo test --locked`, etc.).
3. Confirm GitHub Actions is green.

Do not bump only one pin.

## When to run

- Before every push that changes core mesh, topology, or neuromodulation code.
- After resolving merges with `main`.
- Before requesting a review.

## Mandatory commands

### Format + core locked test matrix

```bash
# Success is silent: exit 0 and no stdout means formatting is clean.
cargo fmt --check

cargo test --locked
cargo clippy --all-features -- -D warnings
```

## Regression guards

After any "security" or dependency PR, confirm core product APIs still exist:

```bash
# Check for key structs and modules
rg -n 'pub struct SynapticMesh' src/mesh.rs
rg -n 'pub struct NeuromodNeuron' src/router.rs
rg -n 'pub struct SynapticGraph' src/topology/graph.rs
rg -n 'pub mod topology' src/lib.rs
rg -n 'pub use router::\{[^}]*NeuromodNeuron' src/lib.rs   # confirms the router re-export (not a removed `neuromod` module)
```

## Diff hygiene

```bash
git fetch origin main
git diff --stat origin/main...HEAD
# Expect only intentional files
```

## Origin hygiene (never push local tooling)

These paths must stay untracked and ignored (aligned with `.gitignore`):

- `.worktrees/`
- `.idea/`
- `target/`

```bash
git ls-files .worktrees .idea target   # must print nothing
git check-ignore -v .worktrees .idea target
```

## Do not merge if

- `src/mesh.rs` or `src/router.rs` are unexpectedly altered or removed.
- `git diff origin/main` shows unexpected public-API removals.

## Pass criteria

- All mandatory commands exit 0
- `cargo fmt --check` is silent (no output) with exit 0
- Clippy reports zero warnings under `-D warnings`
- Regression guards pass
- Diff hygiene: only intentional files for the PR
- Local tooling dirs are not in the commit
