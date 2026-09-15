// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint/resume equivalence: a valid restored [`SynapticMesh`] must
//! continue **tick-for-tick identically** to the uninterrupted live mesh.
//!
//! Malformed checkpoints are rejected by the load-path tests in `src/mesh.rs`
//! and `src/delay/ring_buffer.rs` (LIM-1106 / LIM-1151). This suite never
//! repairs invalid state; it only serializes meshes that the constructors
//! and `propagate` already consider valid.
//!
//! # What is compared
//!
//! After the prefix, and again after every suffix tick, live and restored
//! copies must match on:
//!
//! - returned synaptic currents
//! - `tick`
//! - queued delay-buffer deliveries (the checkpoint's `delay_buffer` slots)
//! - the rest of the checkpoint snapshot (graph + buffer metadata)
//!
//! Restore is exercised through JSON and one non-JSON serde format (`bincode`,
//! a dev-only dependency).
//!
//! # CI vs nightly
//!
//! Default `cargo test` runs [`resume_equivalence_seeded_ci`]: 512 generated
//! cases from deterministic seeds, plus named boundary and regression
//! fixtures. That is the bounded CI profile.
//!
//! A longer ignored/nightly profile lives in [`resume_equivalence_nightly`]:
//!
//! ```text
//! cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
//! CHECKPOINT_RESUME_CASES=50000 cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
//! ```
//!
//! The default nightly length is 10_000 seeds. Override with
//! `CHECKPOINT_RESUME_CASES`. Failing seeds belong in [`REGRESSION_SEEDS`];
//! if the on-failure shrinker prints a smaller scenario, persist that as a
//! named fixture beside the other boundary tests.

use synaptic_wiring::mesh::SynapticMesh;
use synaptic_wiring::topology::{SynapticGraph, generate_random, generate_small_world};
use synaptic_wiring::types::{Polarity, SynapseDescriptor};

/// Seeded cases executed in normal CI (`cargo test`).
const CI_CASES: u64 = 512;

/// Default seed count for [`resume_equivalence_nightly`].
const NIGHTLY_CASES_DEFAULT: u64 = 10_000;

/// Seeds that previously failed, or that pin a recipe at the start of the
/// generator stream. Add every newly discovered failure here.
const REGRESSION_SEEDS: &[u64] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // one of each [`Recipe`] at the base stream
];

/// Exact magnitudes used by the generator so snapshots stay easy to read.
const WEIGHTS: [f32; 8] = [0.25, 0.5, 0.75, 1.0, 0.125, 1.5, 0.375, 0.625];

/// Graded activations, including negatives (signed weights × signed input).
const ACTIVATIONS: [f32; 7] = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Recipe {
    EmptyGraph,
    SingleNeuron,
    DelayZero,
    DelayCapacity,
    MultiSpikeSameSlot,
    CheckpointBeforeDelivery,
    SignedWeights,
    EmptyTicks,
    GeneratedRandom,
    GeneratedSmallWorld,
}

impl Recipe {
    const ALL: [Recipe; 10] = [
        Self::EmptyGraph,
        Self::SingleNeuron,
        Self::DelayZero,
        Self::DelayCapacity,
        Self::MultiSpikeSameSlot,
        Self::CheckpointBeforeDelivery,
        Self::SignedWeights,
        Self::EmptyTicks,
        Self::GeneratedRandom,
        Self::GeneratedSmallWorld,
    ];

    fn from_seed(seed: u64) -> Self {
        Self::ALL[(seed as usize) % Self::ALL.len()]
    }
}

#[derive(Clone, Debug, PartialEq)]
enum TickEvent {
    Binary(Vec<bool>),
    Graded(Vec<f32>),
}

impl TickEvent {
    fn idle(n: usize) -> Self {
        Self::Binary(vec![false; n])
    }

    fn fire(n: usize, neurons: &[usize]) -> Self {
        let mut spikes = vec![false; n];
        for &i in neurons {
            spikes[i] = true;
        }
        Self::Binary(spikes)
    }
}

#[derive(Clone, Debug)]
struct Scenario {
    seed: u64,
    recipe: Recipe,
    neuron_count: usize,
    descriptors: Vec<SynapseDescriptor>,
    buffer_max_delay: usize,
    prefix: Vec<TickEvent>,
    suffix: Vec<TickEvent>,
}

impl Scenario {
    fn from_seed(seed: u64) -> Self {
        let recipe = Recipe::from_seed(seed);
        let mut rng = SplitMix64::new(seed ^ 0xC0FF_EE11_D15C_A5E5);
        match recipe {
            Recipe::EmptyGraph => empty_graph(seed),
            Recipe::SingleNeuron => single_neuron(seed, &mut rng),
            Recipe::DelayZero => delay_zero(seed, &mut rng),
            Recipe::DelayCapacity => delay_capacity(seed, &mut rng),
            Recipe::MultiSpikeSameSlot => multi_spike_same_slot(seed, &mut rng),
            Recipe::CheckpointBeforeDelivery => checkpoint_before_delivery(seed, &mut rng),
            Recipe::SignedWeights => signed_weights(seed, &mut rng),
            Recipe::EmptyTicks => empty_ticks(seed, &mut rng),
            Recipe::GeneratedRandom => generated_random(seed, &mut rng),
            Recipe::GeneratedSmallWorld => generated_small_world(seed, &mut rng),
        }
    }

    fn mesh(&self) -> SynapticMesh {
        let graph = SynapticGraph::from_descriptors(self.neuron_count, &self.descriptors)
            .expect("generator must emit valid descriptors");
        SynapticMesh::try_with_max_delay(graph, self.buffer_max_delay)
            .expect("buffer capacity must hold every synapse delay")
    }
}

// ── Boundary / recipe constructors ────────────────────────────────────────────

fn empty_graph(seed: u64) -> Scenario {
    Scenario {
        seed,
        recipe: Recipe::EmptyGraph,
        neuron_count: 0,
        descriptors: Vec::new(),
        buffer_max_delay: 0,
        prefix: vec![TickEvent::idle(0), TickEvent::idle(0)],
        suffix: vec![TickEvent::idle(0), TickEvent::Graded(Vec::new())],
    }
}

fn single_neuron(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let delay = rng.inclusive(4) as u16;
    let extra = rng.inclusive(3);
    let descriptors = if rng.bool() {
        vec![descriptor(0, 0, rng.weight(), delay, Polarity::Excitatory)]
    } else {
        Vec::new()
    };
    let n = 1;
    let buffer_max_delay = usize::from(delay) + extra;
    let prefix_len = rng.inclusive(4);
    let suffix_len = rng.inclusive(4).max(1);
    Scenario {
        seed,
        recipe: Recipe::SingleNeuron,
        neuron_count: n,
        descriptors,
        buffer_max_delay,
        prefix: random_events(rng, n, prefix_len),
        suffix: random_events(rng, n, suffix_len),
    }
}

fn delay_zero(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(6).max(2);
    let mut descriptors = Vec::new();
    for src in 0..n {
        let src_polarity = random_polarity(rng);
        let tgt = (src + 1) % n;
        descriptors.push(descriptor(src, tgt, rng.weight(), 0, src_polarity));
        if rng.bool() {
            let tgt2 = rng.bounded(n);
            descriptors.push(descriptor(src, tgt2, rng.weight(), 0, src_polarity));
        }
    }
    let prefix_len = rng.inclusive(5);
    let suffix_len = rng.inclusive(5).max(1);
    Scenario {
        seed,
        recipe: Recipe::DelayZero,
        neuron_count: n,
        descriptors,
        buffer_max_delay: rng.inclusive(3),
        prefix: random_events(rng, n, prefix_len),
        suffix: random_events(rng, n, suffix_len),
    }
}

fn delay_capacity(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(5).max(2);
    let delay = rng.inclusive(8).max(1) as u16;
    let extra = rng.inclusive(2);
    let descriptors = vec![
        descriptor(0, 1, rng.weight(), delay, random_polarity(rng)),
        descriptor(1, 0, rng.weight(), delay, random_polarity(rng)),
    ];
    let buffer_max_delay = usize::from(delay) + extra;
    let extra_suffix = rng.inclusive(3);
    // Fire into the max-delay slot, then checkpoint immediately so the
    // restored copy still has to wait the full capacity.
    Scenario {
        seed,
        recipe: Recipe::DelayCapacity,
        neuron_count: n,
        descriptors,
        buffer_max_delay,
        prefix: vec![TickEvent::fire(n, &[0, 1])],
        suffix: {
            let mut suffix = vec![TickEvent::idle(n); usize::from(delay) + 1];
            suffix.extend(random_events(rng, n, extra_suffix));
            suffix
        },
    }
}

fn multi_spike_same_slot(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = 3;
    let delay = rng.inclusive(5).max(1) as u16;
    let descriptors = vec![
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(1, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
    ];
    Scenario {
        seed,
        recipe: Recipe::MultiSpikeSameSlot,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay),
        prefix: vec![TickEvent::fire(n, &[0, 1])],
        suffix: {
            let mut suffix = vec![TickEvent::idle(n); usize::from(delay)];
            suffix.push(TickEvent::idle(n));
            suffix.push(TickEvent::fire(n, &[0, 1]));
            suffix.extend(vec![TickEvent::idle(n); usize::from(delay)]);
            suffix
        },
    }
}

fn checkpoint_before_delivery(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(4).max(2);
    let delay = rng.inclusive(6).max(1) as u16;
    let descriptors = vec![descriptor(0, 1, rng.weight(), delay, Polarity::Excitatory)];
    let mut prefix = vec![TickEvent::fire(n, &[0])];
    prefix.extend(vec![TickEvent::idle(n); usize::from(delay) - 1]);
    Scenario {
        seed,
        recipe: Recipe::CheckpointBeforeDelivery,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay) + rng.inclusive(2),
        prefix,
        suffix: {
            let extra = rng.inclusive(4);
            let mut suffix = vec![TickEvent::idle(n)];
            suffix.extend(random_events(rng, n, extra));
            suffix
        },
    }
}

fn signed_weights(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = 3;
    let delay = rng.inclusive(4) as u16;
    let descriptors = vec![
        descriptor(0, 2, rng.weight(), delay, Polarity::Excitatory),
        descriptor(1, 2, rng.weight(), delay.max(1), Polarity::Inhibitory),
    ];
    Scenario {
        seed,
        recipe: Recipe::SignedWeights,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay.max(1)) + rng.inclusive(2),
        prefix: vec![TickEvent::fire(n, &[0, 1]), TickEvent::idle(n)],
        suffix: vec![
            TickEvent::idle(n),
            TickEvent::Graded(vec![0.5, -0.5, 0.0]),
            TickEvent::idle(n),
            TickEvent::idle(n),
        ],
    }
}

fn empty_ticks(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(5).max(2);
    let delay = rng.inclusive(4) as u16;
    let descriptors = vec![descriptor(0, 1, rng.weight(), delay, random_polarity(rng))];
    let prefix_len = rng.inclusive(6);
    let suffix_len = rng.inclusive(6).max(2);
    Scenario {
        seed,
        recipe: Recipe::EmptyTicks,
        neuron_count: n,
        descriptors,
        buffer_max_delay: usize::from(delay) + rng.inclusive(2),
        prefix: (0..prefix_len).map(|_| TickEvent::idle(n)).collect(),
        suffix: (0..suffix_len).map(|_| TickEvent::idle(n)).collect(),
    }
}

fn generated_random(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = rng.inclusive(8).max(2);
    let max_delay = rng.inclusive(6) as u16;
    let p = 0.2 + rng.f32() * 0.6;
    let inh = rng.f32() * 0.4;
    let graph = generate_random(n, p, max_delay, inh).expect("bounded generate_random");
    from_graph(seed, Recipe::GeneratedRandom, graph, rng)
}

fn generated_small_world(seed: u64, rng: &mut SplitMix64) -> Scenario {
    let n = [4, 6, 8][rng.bounded(3)];
    let k = 2;
    let max_delay = rng.inclusive(5) as u16;
    let graph =
        generate_small_world(n, k, 0.2, max_delay, 0.25).expect("bounded generate_small_world");
    from_graph(seed, Recipe::GeneratedSmallWorld, graph, rng)
}

fn from_graph(seed: u64, recipe: Recipe, graph: SynapticGraph, rng: &mut SplitMix64) -> Scenario {
    let n = graph.neuron_count();
    let graph_max = usize::from(graph.max_delay());
    let extra = rng.inclusive(3);
    let descriptors = descriptors_from_graph(&graph);
    Scenario {
        seed,
        recipe,
        neuron_count: n,
        descriptors,
        buffer_max_delay: graph_max + extra,
        prefix: {
            let prefix_len = rng.inclusive(8);
            random_events(rng, n, prefix_len)
        },
        suffix: {
            let suffix_len = rng.inclusive(8).max(1);
            random_events(rng, n, suffix_len)
        },
    }
}

fn descriptors_from_graph(graph: &SynapticGraph) -> Vec<SynapseDescriptor> {
    let mut descriptors = Vec::new();
    for src in 0..graph.neuron_count() {
        for (target, weight, delay, polarity) in graph.outgoing(src) {
            descriptors.push(SynapseDescriptor {
                source: src as u32,
                target,
                weight: weight.abs(),
                delay,
                polarity,
            });
        }
    }
    descriptors
}

fn descriptor(
    source: usize,
    target: usize,
    weight: f32,
    delay: u16,
    polarity: Polarity,
) -> SynapseDescriptor {
    SynapseDescriptor {
        source: source as u32,
        target: target as u32,
        weight,
        delay,
        polarity,
    }
}

fn random_polarity(rng: &mut SplitMix64) -> Polarity {
    if rng.bool() {
        Polarity::Inhibitory
    } else {
        Polarity::Excitatory
    }
}

fn random_events(rng: &mut SplitMix64, n: usize, len: usize) -> Vec<TickEvent> {
    (0..len).map(|_| random_event(rng, n)).collect()
}

fn random_event(rng: &mut SplitMix64, n: usize) -> TickEvent {
    match rng.bounded(5) {
        0 | 1 => TickEvent::idle(n),
        2 | 3 => {
            let spikes = (0..n).map(|_| rng.bounded(4) == 0).collect();
            TickEvent::Binary(spikes)
        }
        _ => {
            let activations = (0..n)
                .map(|_| ACTIVATIONS[rng.bounded(ACTIVATIONS.len())])
                .collect();
            TickEvent::Graded(activations)
        }
    }
}

// ── Equivalence engine ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SerdeFormat {
    Json,
    Bincode,
}

impl SerdeFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Bincode => "bincode",
        }
    }

    fn restore(self, mesh: &SynapticMesh) -> SynapticMesh {
        match self {
            Self::Json => {
                let json = serde_json::to_string(mesh).expect("valid mesh must serialize to JSON");
                serde_json::from_str(&json).expect("valid JSON checkpoint must deserialize")
            }
            Self::Bincode => {
                let bytes = bincode::serialize(mesh).expect("valid mesh must serialize to bincode");
                bincode::deserialize(&bytes).expect("valid bincode checkpoint must deserialize")
            }
        }
    }
}

fn apply_event(mesh: &mut SynapticMesh, event: &TickEvent) -> Vec<f32> {
    match event {
        TickEvent::Binary(spikes) => mesh
            .propagate(spikes)
            .expect("generated spike vector matches neuron_count"),
        TickEvent::Graded(activations) => mesh
            .propagate_graded(activations)
            .expect("generated activations are finite and match neuron_count"),
    }
}

fn checkpoint_snapshot(mesh: &SynapticMesh) -> serde_json::Value {
    serde_json::to_value(mesh).expect("valid mesh must serialize")
}

fn queued_deliveries(snapshot: &serde_json::Value) -> &serde_json::Value {
    &snapshot["delay_buffer"]["slots"]
}

fn meshes_equivalent(live: &SynapticMesh, restored: &SynapticMesh) -> Result<(), String> {
    if live.tick() != restored.tick() {
        return Err(format!(
            "tick mismatch: live={} restored={}",
            live.tick(),
            restored.tick()
        ));
    }
    let live_snap = checkpoint_snapshot(live);
    let restored_snap = checkpoint_snapshot(restored);
    if live_snap != restored_snap {
        return Err(format!(
            "checkpoint snapshot mismatch\nlive queued={}\nrestored queued={}",
            queued_deliveries(&live_snap),
            queued_deliveries(&restored_snap)
        ));
    }
    Ok(())
}

fn check_resume(scenario: &Scenario) -> Result<(), String> {
    check_resume_format(scenario, SerdeFormat::Json)?;
    check_resume_format(scenario, SerdeFormat::Bincode)?;
    Ok(())
}

fn check_resume_format(scenario: &Scenario, format: SerdeFormat) -> Result<(), String> {
    let mut live = scenario.mesh();
    for event in &scenario.prefix {
        let _ = apply_event(&mut live, event);
    }

    let mut restored = format.restore(&live);
    meshes_equivalent(&live, &restored).map_err(|err| {
        format!(
            "{} immediately after restore (seed {}, {:?}): {err}",
            format.name(),
            scenario.seed,
            scenario.recipe
        )
    })?;

    for (i, event) in scenario.suffix.iter().enumerate() {
        let live_currents = apply_event(&mut live, event);
        let restored_currents = apply_event(&mut restored, event);
        if live_currents != restored_currents {
            return Err(format!(
                "{} suffix tick {i} currents mismatch (seed {}, {:?}): live={live_currents:?} restored={restored_currents:?}",
                format.name(),
                scenario.seed,
                scenario.recipe
            ));
        }
        meshes_equivalent(&live, &restored).map_err(|err| {
            format!(
                "{} after suffix tick {i} (seed {}, {:?}): {err}",
                format.name(),
                scenario.seed,
                scenario.recipe
            )
        })?;
    }
    Ok(())
}

fn check_seed(seed: u64) {
    let scenario = Scenario::from_seed(seed);
    if let Err(err) = check_resume(&scenario) {
        let minimized = shrink(&scenario);
        panic!(
            "{err}\n\noriginal scenario: {scenario:?}\n\nminimized scenario: {minimized:?}\n\n\
             Persist seed {seed} in REGRESSION_SEEDS. If the minimized case is smaller, \
             add it as a named fixture."
        );
    }
}

/// Greedy shrinker used only on failure so the panic carries a smaller
/// counterexample. It never runs on the passing CI path.
fn shrink(scenario: &Scenario) -> Scenario {
    let mut best = scenario.clone();
    let mut changed = true;
    while changed {
        changed = false;
        if best.prefix.len() > 1 {
            for i in 0..best.prefix.len() {
                let mut candidate = best.clone();
                candidate.prefix.remove(i);
                if check_resume(&candidate).is_err() {
                    best = candidate;
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }
        }
        if best.suffix.len() > 1 {
            for i in 0..best.suffix.len() {
                let mut candidate = best.clone();
                candidate.suffix.remove(i);
                if check_resume(&candidate).is_err() {
                    best = candidate;
                    changed = true;
                    break;
                }
            }
            if changed {
                continue;
            }
        }
        if best.descriptors.len() > 1 {
            for i in 0..best.descriptors.len() {
                let mut candidate = best.clone();
                candidate.descriptors.remove(i);
                let graph_max = candidate
                    .descriptors
                    .iter()
                    .map(|d| usize::from(d.delay))
                    .max()
                    .unwrap_or(0);
                candidate.buffer_max_delay = candidate.buffer_max_delay.max(graph_max);
                if check_resume(&candidate).is_err() {
                    best = candidate;
                    changed = true;
                    break;
                }
            }
        }
    }
    best
}

// ── SplitMix64 (deterministic, no extra RNG crate) ────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bounded(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }

    fn inclusive(&mut self, max: usize) -> usize {
        self.bounded(max.saturating_add(1))
    }

    fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    fn weight(&mut self) -> f32 {
        WEIGHTS[self.bounded(WEIGHTS.len())]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Seeded property suite for default CI: hundreds of generated graphs,
/// delays, signed weights, checkpoint locations, and suffix sequences.
#[test]
fn resume_equivalence_seeded_ci() {
    for seed in 0..CI_CASES {
        check_seed(seed);
    }
}

/// Replays persisted seeds so a once-failing case cannot silently disappear
/// from the generator stream.
#[test]
fn resume_equivalence_regression_seeds() {
    for &seed in REGRESSION_SEEDS {
        check_seed(seed);
    }
}

/// Narrow in-flight regression matching `checkpoint_resume_with_spikes_in_flight_matches_live`.
#[test]
fn regression_in_flight_delay2_json_and_bincode() {
    let scenario = Scenario {
        seed: u64::MAX,
        recipe: Recipe::CheckpointBeforeDelivery,
        neuron_count: 2,
        descriptors: vec![descriptor(0, 1, 1.0, 2, Polarity::Excitatory)],
        buffer_max_delay: 2,
        prefix: vec![TickEvent::fire(2, &[0])],
        suffix: vec![
            TickEvent::idle(2),
            TickEvent::fire(2, &[0]),
            TickEvent::idle(2),
            TickEvent::idle(2),
            TickEvent::fire(2, &[0]),
            TickEvent::idle(2),
            TickEvent::idle(2),
            TickEvent::idle(2),
        ],
    };
    check_resume(&scenario).unwrap();
}

/// Frozen JSON checkpoint captured immediately before the delay-2 delivery.
/// If serde field names or ring layout change, this fixture must be updated
/// deliberately rather than silently accepted.
#[test]
fn regression_frozen_json_in_flight_before_delivery() {
    let json = r#"{
        "graph": {
            "neuron_count": 2,
            "row_ptr": [0, 1, 1],
            "targets": [1],
            "weights": [1.0],
            "delays": [2],
            "polarities": ["Excitatory"]
        },
        "delay_buffer": {
            "slots": [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            "neuron_count": 2,
            "max_delay": 2,
            "current_tick": 1
        },
        "tick": 1
    }"#;
    let mut restored: SynapticMesh =
        serde_json::from_str(json).expect("fixture is a valid checkpoint");
    assert_eq!(restored.tick(), 1);

    let mut live = SynapticMesh::new(
        SynapticGraph::from_descriptors(2, &[descriptor(0, 1, 1.0, 2, Polarity::Excitatory)])
            .unwrap(),
    );
    assert_eq!(live.propagate(&[true, false]).unwrap()[1], 0.0);
    assert_eq!(checkpoint_snapshot(&live), checkpoint_snapshot(&restored));

    for spikes in [[false, false], [true, false], [false, false]] {
        let a = live.propagate(&spikes).unwrap();
        let b = restored.propagate(&spikes).unwrap();
        assert_eq!(a, b);
        assert_eq!(live.tick(), restored.tick());
        assert_eq!(checkpoint_snapshot(&live), checkpoint_snapshot(&restored));
    }
}

#[test]
fn boundary_empty_graph() {
    check_resume(&empty_graph(0)).unwrap();
}

#[test]
fn boundary_single_neuron_no_synapses() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::SingleNeuron,
        neuron_count: 1,
        descriptors: Vec::new(),
        buffer_max_delay: 0,
        prefix: vec![TickEvent::fire(1, &[0]), TickEvent::idle(1)],
        suffix: vec![TickEvent::Graded(vec![0.5]), TickEvent::idle(1)],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_single_neuron_self_loop_delay_zero() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::DelayZero,
        neuron_count: 1,
        descriptors: vec![descriptor(0, 0, 0.75, 0, Polarity::Excitatory)],
        buffer_max_delay: 0,
        prefix: vec![TickEvent::fire(1, &[0])],
        suffix: vec![TickEvent::idle(1), TickEvent::fire(1, &[0])],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_delay_zero_with_inhibitory_and_empty_ticks() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::DelayZero,
        neuron_count: 3,
        descriptors: vec![
            descriptor(0, 1, 0.5, 0, Polarity::Excitatory),
            descriptor(1, 2, 0.4, 0, Polarity::Inhibitory),
        ],
        buffer_max_delay: 0,
        prefix: vec![TickEvent::idle(3), TickEvent::fire(3, &[0, 1])],
        suffix: vec![TickEvent::idle(3), TickEvent::fire(3, &[1])],
    };
    check_resume(&scenario).unwrap();
}

#[test]
fn boundary_delay_capacity_exact_and_headroom() {
    for extra in [0usize, 3] {
        let delay = 5u16;
        let scenario = Scenario {
            seed: extra as u64,
            recipe: Recipe::DelayCapacity,
            neuron_count: 2,
            descriptors: vec![descriptor(0, 1, 1.0, delay, Polarity::Excitatory)],
            buffer_max_delay: usize::from(delay) + extra,
            prefix: vec![TickEvent::fire(2, &[0])],
            suffix: vec![TickEvent::idle(2); usize::from(delay) + 2],
        };
        check_resume(&scenario).unwrap();
    }
}

#[test]
fn boundary_multiple_spikes_same_slot_and_checkpoint_before_delivery() {
    let scenario = Scenario {
        seed: 0,
        recipe: Recipe::MultiSpikeSameSlot,
        neuron_count: 3,
        descriptors: vec![
            descriptor(0, 2, 0.3, 2, Polarity::Excitatory),
            descriptor(1, 2, 0.7, 2, Polarity::Inhibitory),
        ],
        buffer_max_delay: 2,
        prefix: vec![TickEvent::fire(3, &[0, 1]), TickEvent::idle(3)],
        suffix: vec![TickEvent::idle(3), TickEvent::idle(3)],
    };
    check_resume(&scenario).unwrap();

    let mut live = scenario.mesh();
    let _ = apply_event(&mut live, &scenario.prefix[0]);
    let _ = apply_event(&mut live, &scenario.prefix[1]);
    let mut restored = SerdeFormat::Json.restore(&live);
    let live_delivery = apply_event(&mut live, &TickEvent::idle(3));
    let restored_delivery = apply_event(&mut restored, &TickEvent::idle(3));
    assert_eq!(live_delivery, restored_delivery);
    assert!((live_delivery[2] - (0.3 - 0.7)).abs() < 1e-6);
}

#[test]
fn cross_format_restore_equivalence() {
    let seeds = [0u64, 1, 3, 4, 5, 8, 42, 255];
    for seed in seeds {
        let scenario = Scenario::from_seed(seed);
        let mut live = scenario.mesh();
        for event in &scenario.prefix {
            let _ = apply_event(&mut live, event);
        }
        let from_json = SerdeFormat::Json.restore(&live);
        let from_bincode = SerdeFormat::Bincode.restore(&live);
        meshes_equivalent(&live, &from_json).unwrap();
        meshes_equivalent(&live, &from_bincode).unwrap();
        meshes_equivalent(&from_json, &from_bincode).unwrap();

        // JSON → bincode → JSON must not drift for a valid checkpoint.
        let via_bincode = SerdeFormat::Bincode.restore(&from_json);
        let via_json = SerdeFormat::Json.restore(&from_bincode);
        meshes_equivalent(&live, &via_bincode).unwrap();
        meshes_equivalent(&live, &via_json).unwrap();
    }
}

/// Deeper seeded run, skipped by default CI.
///
/// ```text
/// cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
/// CHECKPOINT_RESUME_CASES=50000 cargo test --locked --test checkpoint_resume resume_equivalence_nightly -- --ignored
/// ```
#[test]
#[ignore = "nightly profile: CHECKPOINT_RESUME_CASES (default 10000) seeded resume-equivalence run"]
fn resume_equivalence_nightly() {
    let cases = std::env::var("CHECKPOINT_RESUME_CASES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(NIGHTLY_CASES_DEFAULT);
    let start = CI_CASES;
    for seed in start..start.saturating_add(cases) {
        check_seed(seed);
    }
}
