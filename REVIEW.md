# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, or public APIs.

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
rg -n 'pub struct NeuromodNeuron' src/neuromod.rs
rg -n 'pub struct SynapticGraph' src/topology/graph.rs
rg -n 'pub mod topology' src/lib.rs
rg -n 'pub mod neuromod' src/lib.rs
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

- `src/mesh.rs` or `src/neuromod.rs` are unexpectedly altered or removed.
- `git diff origin/main` shows unexpected public-API removals.

## Pass criteria

- All mandatory commands exit 0
- `cargo fmt --check` is silent (no output) with exit 0
- Clippy reports zero warnings under `-D warnings`
- Regression guards pass
- Diff hygiene: only intentional files for the PR
- Local tooling dirs are not in the commit
