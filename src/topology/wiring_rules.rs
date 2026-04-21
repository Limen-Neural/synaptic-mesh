//! Wiring rules for synaptic graph construction.
//!
//! These functions operate on an existing [`SynapticGraph`] to enforce
//! biological constraints like Dale's law or to assign realistic
//! axonal propagation delays.

use crate::topology::graph::SynapticGraph;
use crate::types::{DelayTicks, Polarity, SynapseDescriptor};

/// Apply Dale's law to a graph: all outgoing synapses from each neuron
/// share a single polarity (excitatory or inhibitory).
///
/// The first `inh_count` neurons (by ID) are assigned inhibitory polarity;
/// the rest are excitatory.
///
/// # References
///
/// Dale, H. H. (1935). *Pharmacology and Nerve-endings.*
pub fn apply_dale_polarity(
    graph: &SynapticGraph,
    inhibitory_fraction: f32,
) -> Vec<Polarity> {
    let n = graph.neuron_count();
    let inh_count = (n as f32 * inhibitory_fraction.clamp(0.0, 1.0)) as usize;
    (0..n)
        .map(|i| {
            if i < inh_count {
                Polarity::Inhibitory
            } else {
                Polarity::Excitatory
            }
        })
        .collect()
}

/// Assign axonal propagation delays based on distance between neurons.
///
/// If the graph has position data, delay is proportional to Euclidean distance:
///   `delay = clamp(round(distance / speed), 1, max_delay)`
///
/// If no positions are available, delay is assigned deterministically from
/// source/target indices.
///
/// # Arguments
///
/// * `n` — neuron count
/// * `descriptors` — synapse descriptors to update
/// * `positions` — optional 3D positions per neuron
/// * `speed` — propagation speed (distance units per tick)
/// * `max_delay` — maximum delay in ticks
pub fn assign_delays(
    descriptors: &mut [SynapseDescriptor],
    positions: Option<&[[f32; 3]]>,
    speed: f32,
    max_delay: DelayTicks,
) {
    for desc in descriptors.iter_mut() {
        if let Some(pos) = positions {
            let src_pos = &pos[desc.source as usize];
            let tgt_pos = &pos[desc.target as usize];
            let dist = ((src_pos[0] - tgt_pos[0]).powi(2)
                + (src_pos[1] - tgt_pos[1]).powi(2)
                + (src_pos[2] - tgt_pos[2]).powi(2))
            .sqrt();
            let delay = (dist / speed.max(1e-6)).round() as u16;
            desc.delay = delay.clamp(1, max_delay);
        } else {
            // Deterministic delay from indices
            let mixed = (desc.source as usize * 71 + desc.target as usize * 37 + 13) % (max_delay as usize + 1);
            desc.delay = (mixed as u16).max(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::generators::generate_random;

    #[test]
    fn dale_polarity_splits_correctly() {
        let graph = generate_random(100, 0.1, 5, 0.2).unwrap();
        let polarities = apply_dale_polarity(&graph, 0.2);
        assert_eq!(polarities.len(), 100);
        assert_eq!(
            polarities.iter().filter(|&&p| p == Polarity::Inhibitory).count(),
            20
        );
        assert_eq!(
            polarities.iter().filter(|&&p| p == Polarity::Excitatory).count(),
            80
        );
    }

    #[test]
    fn distance_based_delays() {
        let mut descs = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let positions = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]; // distance = 5
        assign_delays(&mut descs, Some(&positions), 2.5, 10);
        // 5.0 / 2.5 = 2.0 → delay = 2
        assert_eq!(descs[0].delay, 2);
    }

    #[test]
    fn delay_clamped_to_max() {
        let mut descs = vec![SynapseDescriptor {
            source: 0,
            target: 1,
            weight: 0.5,
            delay: 0,
            polarity: Polarity::Excitatory,
        }];
        let positions = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]; // distance = 100
        assign_delays(&mut descs, Some(&positions), 1.0, 5);
        assert_eq!(descs[0].delay, 5);
    }
}
