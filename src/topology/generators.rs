//! Deterministic topology generators.
//!
//! All generators use index-based pseudo-random hashing (no external RNG)
//! following the corinth-canal pattern — deterministic from neuron index
//! alone, reproducible across runs.
//!
//! # References
//!
//! - Erdős, P. & Rényi, A. (1959). *On random graphs.*
//! - Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of
//!   'small-world' networks.* Nature, 393, 440–442.
//! - Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in
//!   random networks.* Science, 286, 509–512.

use crate::error::{MeshError, Result};
use crate::topology::graph::SynapticGraph;
use crate::types::{Polarity, SynapseDescriptor};

// ── Deterministic hash helpers ────────────────────────────────────────────────

/// Simple deterministic hash from two indices — produces a value in [0, 1).
/// Uses a combination of golden-ratio fractional hashing and bit mixing.
fn hash_pair(a: usize, b: usize) -> f32 {
    const GOLDEN: f64 = 1.618_033_988_749_895;
    const SILVER: f64 = 2.414_213_562_373_095;
    let mixed = (a as f64 * GOLDEN + b as f64 * SILVER) % 1.0;
    mixed.abs() as f32
}

/// Deterministic weight from neuron pair.
fn hash_weight(src: usize, tgt: usize, base: f32, spread: f32) -> f32 {
    base + hash_pair(src * 97 + 13, tgt * 53 + 7) * spread
}

/// Deterministic delay from neuron pair.
fn hash_delay(src: usize, tgt: usize, max_delay: u16) -> u16 {
    if max_delay == 0 {
        return 0;
    }
    let h = hash_pair(src * 71 + 3, tgt * 37 + 11);
    (h * max_delay as f32).round().min(max_delay as f32).max(1.0) as u16
}

// ── Generators ────────────────────────────────────────────────────────────────

/// Erdős–Rényi random graph.
///
/// Each directed edge `(i, j)` with `i ≠ j` exists independently with
/// probability `p`. Self-connections are not created.
///
/// Deterministic: the same `(n, p)` always produces the same graph.
pub fn generate_random(
    n: usize,
    p: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if p < 0.0 || p > 1.0 {
        return Err(MeshError::InvalidConfig(format!(
            "connection probability p={p} must be in [0, 1]"
        )));
    }
    if n == 0 {
        return Err(MeshError::InvalidConfig("neuron count must be ≥ 1".into()));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();

    for src in 0..n {
        let polarity = if src < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        for tgt in 0..n {
            if src == tgt {
                continue;
            }
            if hash_pair(src, tgt) < p {
                descriptors.push(SynapseDescriptor {
                    source: src as u32,
                    target: tgt as u32,
                    weight: hash_weight(src, tgt, 0.3, 0.6),
                    delay: hash_delay(src, tgt, max_delay),
                    polarity,
                });
            }
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Watts–Strogatz small-world graph.
///
/// Start with a ring lattice where each neuron is connected to its `k` nearest
/// neighbours (k/2 on each side). Then each edge is rewired with probability
/// `beta` to a uniformly random target.
///
/// # References
///
/// Watts, D. J. & Strogatz, S. H. (1998). *Collective dynamics of
/// 'small-world' networks.* Nature, 393, 440–442.
pub fn generate_small_world(
    n: usize,
    k: usize,
    beta: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if n < 3 {
        return Err(MeshError::InvalidConfig(
            "small-world requires n ≥ 3".into(),
        ));
    }
    if k == 0 || k >= n {
        return Err(MeshError::InvalidConfig(format!(
            "k={k} must be in [1, n-1)"
        )));
    }
    if beta < 0.0 || beta > 1.0 {
        return Err(MeshError::InvalidConfig(format!(
            "beta={beta} must be in [0, 1]"
        )));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let half_k = k / 2;
    let mut descriptors = Vec::new();

    for src in 0..n {
        let polarity = if src < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        for offset in 1..=half_k {
            let mut tgt = (src + offset) % n;

            // Rewire with probability beta
            if hash_pair(src * 131 + offset, tgt * 79) < beta {
                // Pick a deterministic "random" target
                let new_tgt = (hash_pair(src * 173 + offset * 41, n * 29)
                    * (n - 1) as f32) as usize;
                let new_tgt = if new_tgt >= src {
                    (new_tgt + 1) % n
                } else {
                    new_tgt
                };
                tgt = new_tgt;
            }

            if tgt != src {
                descriptors.push(SynapseDescriptor {
                    source: src as u32,
                    target: tgt as u32,
                    weight: hash_weight(src, tgt, 0.4, 0.5),
                    delay: hash_delay(src, tgt, max_delay),
                    polarity,
                });
            }
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Barabási–Albert scale-free graph.
///
/// Start with `m0` fully connected neurons. Each new neuron attaches to `m`
/// existing neurons with probability proportional to their current degree
/// (preferential attachment).
///
/// # References
///
/// Barabási, A.-L. & Albert, R. (1999). *Emergence of scaling in random
/// networks.* Science, 286, 509–512.
pub fn generate_scale_free(
    n: usize,
    m0: usize,
    m: usize,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if m0 < 2 || m0 > n {
        return Err(MeshError::InvalidConfig(format!(
            "m0={m0} must be in [2, n]"
        )));
    }
    if m == 0 || m > m0 {
        return Err(MeshError::InvalidConfig(format!(
            "m={m} must be in [1, m0]"
        )));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();
    let mut degree = vec![0usize; n];

    // Seed: fully connect the first m0 nodes
    for i in 0..m0 {
        let polarity = if i < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };
        for j in 0..m0 {
            if i != j {
                descriptors.push(SynapseDescriptor {
                    source: i as u32,
                    target: j as u32,
                    weight: hash_weight(i, j, 0.3, 0.6),
                    delay: hash_delay(i, j, max_delay),
                    polarity,
                });
                degree[i] += 1;
            }
        }
    }

    // Growth phase: add nodes m0..n
    for new_node in m0..n {
        let polarity = if new_node < inhibitory_cutoff {
            Polarity::Inhibitory
        } else {
            Polarity::Excitatory
        };

        let total_degree: usize = degree[..new_node].iter().sum();
        let mut attached = 0;
        let mut target_cursor = 0;

        // Deterministic preferential attachment
        while attached < m && target_cursor < new_node {
            let prob = if total_degree > 0 {
                degree[target_cursor] as f32 / total_degree as f32
            } else {
                1.0 / new_node as f32
            };

            let roll = hash_pair(new_node * 113 + attached * 59, target_cursor * 83);
            if roll < prob * (m as f32) {
                descriptors.push(SynapseDescriptor {
                    source: new_node as u32,
                    target: target_cursor as u32,
                    weight: hash_weight(new_node, target_cursor, 0.3, 0.6),
                    delay: hash_delay(new_node, target_cursor, max_delay),
                    polarity,
                });
                // Bidirectional attachment
                let reverse_polarity = if target_cursor < inhibitory_cutoff {
                    Polarity::Inhibitory
                } else {
                    Polarity::Excitatory
                };
                descriptors.push(SynapseDescriptor {
                    source: target_cursor as u32,
                    target: new_node as u32,
                    weight: hash_weight(target_cursor, new_node, 0.3, 0.6),
                    delay: hash_delay(target_cursor, new_node, max_delay),
                    polarity: reverse_polarity,
                });
                degree[new_node] += 1;
                degree[target_cursor] += 1;
                attached += 1;
            }
            target_cursor += 1;
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

/// Feed-forward layered network.
///
/// Neurons are arranged in layers; each neuron in layer `l` connects to each
/// neuron in layer `l+1` with probability `inter_layer_p`.
pub fn generate_layered(
    layer_sizes: &[usize],
    inter_layer_p: f32,
    max_delay: u16,
    inhibitory_fraction: f32,
) -> Result<SynapticGraph> {
    if layer_sizes.is_empty() {
        return Err(MeshError::InvalidConfig("at least one layer required".into()));
    }
    if inter_layer_p < 0.0 || inter_layer_p > 1.0 {
        return Err(MeshError::InvalidConfig(format!(
            "inter_layer_p={inter_layer_p} must be in [0, 1]"
        )));
    }

    let n: usize = layer_sizes.iter().sum();
    if n == 0 {
        return Err(MeshError::InvalidConfig("total neuron count must be ≥ 1".into()));
    }

    let inhibitory_cutoff = (n as f32 * inhibitory_fraction) as usize;
    let mut descriptors = Vec::new();

    // Compute layer offsets
    let mut offsets = Vec::with_capacity(layer_sizes.len());
    let mut offset = 0usize;
    for &size in layer_sizes {
        offsets.push(offset);
        offset += size;
    }

    for layer_idx in 0..layer_sizes.len().saturating_sub(1) {
        let src_start = offsets[layer_idx];
        let src_end = src_start + layer_sizes[layer_idx];
        let tgt_start = offsets[layer_idx + 1];
        let tgt_end = tgt_start + layer_sizes[layer_idx + 1];

        for src in src_start..src_end {
            let polarity = if src < inhibitory_cutoff {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            };

            for tgt in tgt_start..tgt_end {
                if hash_pair(src * 67 + layer_idx, tgt * 43) < inter_layer_p {
                    descriptors.push(SynapseDescriptor {
                        source: src as u32,
                        target: tgt as u32,
                        weight: hash_weight(src, tgt, 0.3, 0.6),
                        delay: hash_delay(src, tgt, max_delay),
                        polarity,
                    });
                }
            }
        }
    }

    SynapticGraph::from_descriptors(n, &descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_graph_deterministic() {
        let g1 = generate_random(64, 0.1, 5, 0.2).unwrap();
        let g2 = generate_random(64, 0.1, 5, 0.2).unwrap();
        assert_eq!(g1.synapse_count(), g2.synapse_count());
    }

    #[test]
    fn random_graph_respects_probability() {
        let g = generate_random(100, 0.0, 5, 0.2).unwrap();
        assert_eq!(g.synapse_count(), 0);

        let g_full = generate_random(10, 1.0, 5, 0.2).unwrap();
        // 10 neurons, no self-connections → 10 * 9 = 90 edges
        assert_eq!(g_full.synapse_count(), 90);
    }

    #[test]
    fn small_world_produces_connected_graph() {
        let g = generate_small_world(32, 4, 0.1, 5, 0.2).unwrap();
        assert!(g.synapse_count() > 0);
        assert_eq!(g.neuron_count(), 32);
    }

    #[test]
    fn small_world_deterministic() {
        let g1 = generate_small_world(32, 4, 0.3, 5, 0.2).unwrap();
        let g2 = generate_small_world(32, 4, 0.3, 5, 0.2).unwrap();
        assert_eq!(g1.synapse_count(), g2.synapse_count());
    }

    #[test]
    fn scale_free_hub_structure() {
        let g = generate_scale_free(50, 5, 3, 5, 0.2).unwrap();
        assert!(g.synapse_count() > 0);

        // The seed nodes should have higher degree than late arrivals
        let seed_degree: usize = (0..5).map(|i| g.out_degree(i)).sum();
        let late_degree: usize = (45..50).map(|i| g.out_degree(i)).sum();
        assert!(
            seed_degree >= late_degree,
            "seed degree {seed_degree} should be ≥ late degree {late_degree}"
        );
    }

    #[test]
    fn layered_feed_forward() {
        let g = generate_layered(&[8, 16, 4], 1.0, 5, 0.2).unwrap();
        assert_eq!(g.neuron_count(), 28);

        // Layer 0→1: 8×16 = 128 edges, Layer 1→2: 16×4 = 64 edges
        // With p=1.0, all inter-layer connections present
        assert_eq!(g.synapse_count(), 128 + 64);
    }

    #[test]
    fn layered_no_back_connections() {
        let g = generate_layered(&[4, 4, 4], 1.0, 5, 0.0).unwrap();
        // No neuron in layer 1 or 2 should connect back to layer 0
        for src in 4..12 {
            for (tgt, _, _, _) in g.outgoing(src) {
                assert!(tgt as usize >= 4, "back-connection from {src} to {tgt}");
            }
        }
    }

    #[test]
    fn dale_law_polarity_applied() {
        let g = generate_random(20, 0.5, 3, 0.3).unwrap();
        // First 6 neurons (30% of 20) should be inhibitory
        for src in 0..6 {
            for (_, weight, _, polarity) in g.outgoing(src) {
                assert_eq!(polarity, Polarity::Inhibitory);
                assert!(weight <= 0.0, "inhibitory neuron {src} has positive weight {weight}");
            }
        }
    }
}
