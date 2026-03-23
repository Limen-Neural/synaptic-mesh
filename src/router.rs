//! SNN-based Anti-Hallucination Layer Router.
//!
//! Uses a small Leaky Integrate-and-Fire (LIF) network to **sparsely** route
//! incoming LLM claims to the minimum set of verification modules needed for the
//! current conversational domain.
//!
//! # Why SNN routing?
//!
//! Loading all Julia verification packages concurrently (ChemEquations.jl,
//! Symbolics.jl, Satisfiability.jl, Clapeyron.jl, etc.) causes JIT compilation
//! storms, VRAM overflow on constrained GPUs, and CPU throttling. The SNN router
//! acts as a **neural gate**: it encodes textual domain signals into spike trains,
//! and only the neuron bank whose firing rate exceeds threshold triggers the
//! corresponding Julia solver pipeline.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────┐
//! │  LLM Output Text                                      │
//! │  "The balanced equation NaOH + HCl → NaCl + H₂O..."  │
//! └──────────────────┬────────────────────────────────────┘
//!                    │
//!        ┌───────────▼───────────┐
//!        │   Feature Extractor   │   keyword density → [0.0, 1.0] per domain
//!        └───────────┬───────────┘
//!                    │  3 input channels
//!        ┌───────────▼───────────┐
//!        │   LIF Neuron Bank     │   one neuron per verification domain
//!        │   N0: Chemistry       │   ← CH0 stimulus
//!        │   N1: Mathematics     │   ← CH1 stimulus
//!        │   N2: Digital Logic   │   ← CH2 stimulus
//!        └───────────┬───────────┘
//!                    │  sparse fire mask
//!        ┌───────────▼──────────────────┐
//!        │  AHL Pipeline activates ONLY │
//!        │  the domains that spiked     │
//!        └──────────────────────────────┘
//! ```
//!
//! STDP feedback loop: when a domain neuron fires and Julia verification
//! succeeds (positive reward), the routing weight for that domain's keyword
//! channel is potentiated (LTP). Failed verifications cause depression (LTD),
//! refining routing accuracy over time.
//!
//! # References
//!
//! Hebbian / STDP learning rule:
//! - Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
//! - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
//!   hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464–10472.
//!
//! Winner-take-all lateral inhibition:
//! - Maass, W. (2000). On the computational power of winner-take-all.
//!   *Neural Computation*, 12(11), 2519–2535.

use serde::{Deserialize, Serialize};

use crate::lif::LifNeuron;

// ── Keyword lists (lowercase) ─────────────────────────────────────────────

const CHEM_KEYWORDS: &[&str] = &[
    "mole", "molarity", "stoichiom", "reactant", "product", "yield",
    "enthalpy", "entropy", "gibbs", "thermodynamic", "exothermic", "endothermic",
    "equilibrium", "le chatelier", "acid", "base", "ph", "buffer",
    "oxidation", "reduction", "redox", "electrochemistry", "galvanic",
    "ideal gas", "van der waals", "clapeyron", "pressure", "volume",
    "boyle", "charles", "avogadro", "dalton", "partial pressure",
    "lewis structure", "vsepr", "hybridization", "molecular geometry",
    "balanced equation", "limiting", "excess", "theoretical yield",
    "calorimetry", "hess", "bond energy", "lattice energy",
    "molality", "colligative", "osmotic", "raoult",
    "naoh", "hcl", "h2o", "nacl", "h2so4", "co2",
];

const MATH_KEYWORDS: &[&str] = &[
    "derivative", "integral", "differentiat", "antiderivative", "limit",
    "calculus", "chain rule", "product rule", "quotient rule",
    "taylor", "maclaurin", "series", "convergence", "divergence",
    "polynomial", "quadratic", "factoring", "roots", "discriminant",
    "logarithm", "exponential", "trig", "sine", "cosine", "tangent",
    "algebra", "equation", "inequality", "simplif", "expand",
    "matrix", "determinant", "eigenvalue", "vector", "linear algebra",
    "proof", "theorem", "lemma", "geometry", "euclidean",
    "congruent", "similar", "parallel", "perpendicular", "angle",
    "conic", "parabola", "ellipse", "hyperbola",
    "sequence", "arithmetic", "geometric", "fibonacci",
    "permutation", "combination", "binomial",
    "symbolics", "canonical", "equivalence",
];

const LOGIC_KEYWORDS: &[&str] = &[
    "boolean", "truth table", "and gate", "or gate", "not gate",
    "nand", "nor", "xor", "xnor", "logic gate",
    "karnaugh", "k-map", "minterm", "maxterm", "sop", "pos",
    "quine-mccluskey", "minimiz", "simplif",
    "flip-flop", "latch", "register", "counter",
    "fsm", "finite state", "mealy", "moore", "state machine",
    "state diagram", "transition table", "reachab", "determinism",
    "combinational", "sequential", "decoder", "encoder", "mux",
    "multiplexer", "demux", "alu", "adder", "subtractor",
    "verilog", "vhdl", "fpga", "lut", "rtl",
    "satisfiab", "smt", "sat solver", "counter-example",
    "don't care", "hazard", "glitch", "timing",
];

// ── Verification domains ──────────────────────────────────────────────────

/// The three verification domains of the Anti-Hallucination Layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationDomain {
    /// Deep Chemistry Logic: stoichiometry, thermodynamics, real-gas EOS.
    Chemistry,
    /// Advanced Mathematical Logic: symbolic algebra, calculus, geometric proofs.
    Mathematics,
    /// Digital Logic & Determinism: Boolean simplification, FSM, reachability.
    DigitalLogic,
}

impl VerificationDomain {
    pub const ALL: [VerificationDomain; 3] = [
        Self::Chemistry,
        Self::Mathematics,
        Self::DigitalLogic,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Chemistry    => 0,
            Self::Mathematics  => 1,
            Self::DigitalLogic => 2,
        }
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Chemistry),
            1 => Some(Self::Mathematics),
            2 => Some(Self::DigitalLogic),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Chemistry    => "Deep Chemistry",
            Self::Mathematics  => "Advanced Math",
            Self::DigitalLogic => "Digital Logic",
        }
    }
}

// ── Domain signal extraction ──────────────────────────────────────────────

/// Domain signal strengths extracted from text, normalised to `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainSignals {
    pub chemistry:     f32,
    pub mathematics:   f32,
    pub digital_logic: f32,
}

impl DomainSignals {
    /// Extract domain keyword densities from LLM output text.
    ///
    /// Saturates at 8 hits → 1.0 (strong signal). Case-insensitive.
    pub fn from_text(text: &str) -> Self {
        let lower = text.to_lowercase();
        const SATURATION: f32 = 8.0;
        Self {
            chemistry:     (count_hits(&lower, CHEM_KEYWORDS) as f32 / SATURATION).min(1.0),
            mathematics:   (count_hits(&lower, MATH_KEYWORDS) as f32 / SATURATION).min(1.0),
            digital_logic: (count_hits(&lower, LOGIC_KEYWORDS) as f32 / SATURATION).min(1.0),
        }
    }

    /// Return as a `[chemistry, mathematics, digital_logic]` channel array.
    pub fn as_channels(&self) -> [f32; 3] {
        [self.chemistry, self.mathematics, self.digital_logic]
    }
}

fn count_hits(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(**kw)).count()
}

// ── SNN router ────────────────────────────────────────────────────────────

/// Number of input channels (one per verification domain).
pub const AHL_NUM_CHANNELS: usize = 3;

/// Integration timesteps per routing decision (more → more stable, higher latency).
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / `ROUTING_TIMESTEPS`) to activate a domain.
/// 3/16 ≈ 0.1875 — a neuron must fire at least 3 out of 16 steps.
const MIN_FIRE_RATE: f32 = 0.1875;

/// Sparse activation decision from the SNN router.
#[derive(Debug, Clone, Default)]
pub struct RoutingDecision {
    /// Which domains should be verified (sparse — usually 1, sometimes 2).
    pub active_domains: Vec<VerificationDomain>,
    /// Per-domain firing rates (for diagnostics and STDP feedback).
    pub firing_rates: [f32; AHL_NUM_CHANNELS],
    /// Raw domain signals fed into the router.
    pub input_signals: DomainSignals,
}

impl RoutingDecision {
    pub fn is_active(&self, domain: VerificationDomain) -> bool {
        self.active_domains.contains(&domain)
    }

    /// True when no domain was activated (no detectable technical claims).
    pub fn is_empty(&self) -> bool {
        self.active_domains.is_empty()
    }
}

/// Anti-Hallucination Layer SNN Router.
///
/// Contains `AHL_NUM_CHANNELS` LIF neurons (one per verification domain) that
/// integrate keyword-derived Poisson spike trains over `ROUTING_TIMESTEPS` to
/// produce a sparse activation mask.
#[derive(Clone, Serialize, Deserialize)]
pub struct AhlRouter {
    neurons: Vec<LifNeuron>,
    /// Cumulative routing decisions since creation.
    pub total_routes: u64,
}

impl Default for AhlRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AhlRouter {
    pub fn new() -> Self {
        let neurons = (0..AHL_NUM_CHANNELS).map(|i| {
            let mut n = LifNeuron::new();
            // Strong self-affinity; weak cross-domain inhibition.
            n.weights = vec![-0.15; AHL_NUM_CHANNELS];
            n.weights[i] = 0.9;
            n.threshold = 0.22;
            n.decay_rate = 0.12;
            n
        }).collect();

        Self { neurons, total_routes: 0 }
    }

    /// Route `text` through the SNN — returns the sparse `RoutingDecision`.
    pub fn route(&mut self, text: &str) -> RoutingDecision {
        let signals = DomainSignals::from_text(text);
        self.route_from_signals(signals)
    }

    /// Route from pre-computed domain signals (useful for testing).
    pub fn route_from_signals(&mut self, signals: DomainSignals) -> RoutingDecision {
        let channels = signals.as_channels();
        let mut spike_counts = [0u32; AHL_NUM_CHANNELS];

        // Reset membrane potentials for a fresh routing decision.
        for n in &mut self.neurons {
            n.membrane_potential = 0.0;
            n.last_spike = false;
        }

        // Integrate over ROUTING_TIMESTEPS.
        for _ in 0..ROUTING_TIMESTEPS {
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                let stimulus: f32 = channels.iter()
                    .zip(neuron.weights.iter())
                    .map(|(ch, w)| ch * w)
                    .sum();
                neuron.integrate(stimulus);
                if neuron.check_fire().is_some() {
                    spike_counts[i] += 1;
                    neuron.last_spike = true;
                } else {
                    neuron.last_spike = false;
                }
            }
        }

        let mut firing_rates = [0.0f32; AHL_NUM_CHANNELS];
        let mut active_domains = Vec::new();
        for i in 0..AHL_NUM_CHANNELS {
            firing_rates[i] = spike_counts[i] as f32 / ROUTING_TIMESTEPS as f32;
            if firing_rates[i] >= MIN_FIRE_RATE {
                if let Some(domain) = VerificationDomain::from_index(i) {
                    active_domains.push(domain);
                }
            }
        }

        self.total_routes += 1;
        RoutingDecision { active_domains, firing_rates, input_signals: signals }
    }

    /// Apply verification outcome as a reward/punishment signal (simplified STDP).
    ///
    /// - `reward > 0` — verification succeeded → LTP on the primary routing weight.
    /// - `reward < 0` — verification failed → LTD (wrong domain classified).
    ///
    /// Winner-take-all cross-inhibition: when `reward > 0` the other domains'
    /// weights for this channel are gently depressed.
    pub fn apply_feedback(&mut self, domain: VerificationDomain, reward: f32) {
        let idx = domain.index();
        let delta = reward * 0.01; // small LR for stability

        self.neurons[idx].weights[idx] =
            (self.neurons[idx].weights[idx] + delta).clamp(0.1, 2.0);

        if reward > 0.0 {
            for j in 0..AHL_NUM_CHANNELS {
                if j != idx {
                    self.neurons[j].weights[idx] =
                        (self.neurons[j].weights[idx] - delta * 0.3).clamp(0.05, 1.5);
                }
            }
        }
    }

    /// Current routing weight matrix (row = neuron, col = input channel).
    pub fn weight_matrix(&self) -> [[f32; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS] {
        let mut m = [[0.0; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS];
        for (i, n) in self.neurons.iter().enumerate() {
            for (j, &w) in n.weights.iter().enumerate() {
                m[i][j] = w;
            }
        }
        m
    }
}
