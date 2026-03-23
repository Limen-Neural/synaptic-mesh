//! # spikenaut-router
//!
//! SNN-based Anti-Hallucination Layer (AHL) router for LLM output verification.
//!
//! Uses a bank of Leaky Integrate-and-Fire (LIF) neurons to sparsely map
//! free-text LLM claims to the minimum set of Julia/Rust verification backends
//! required for the detected domain (Chemistry, Mathematics, Digital Logic).
//!
//! ## Quick start
//!
//! ```rust
//! use spikenaut_router::{AhlRouter, VerificationDomain};
//!
//! let mut router = AhlRouter::new();
//!
//! let decision = router.route(
//!     "Balance: NaOH + HCl → NaCl + H₂O. Find the limiting reactant."
//! );
//!
//! assert!(decision.is_active(VerificationDomain::Chemistry));
//!
//! // After verification succeeds, reinforce the routing weight:
//! router.apply_feedback(VerificationDomain::Chemistry, 1.0);
//! ```
//!
//! ## References
//!
//! **LIF neuron model:**
//! - Lapicque, L. (1907). *Recherches quantitatives sur l'excitation électrique des
//!   nerfs traitée comme une polarisation.* Journal de Physiologie et de Pathologie
//!   Générale, 9, 620–635.
//! - Stein, R. B. (1967). *Some models of neuronal variability.* Biophysical
//!   Journal, 7(1), 37–68.
//!
//! **STDP / Hebbian plasticity:**
//! - Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
//! - Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
//!   hippocampal neurons: dependence on spike timing, synaptic strength, and
//!   postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464–10472.
//!
//! **Winner-take-all lateral inhibition:**
//! - Maass, W. (2000). On the computational power of winner-take-all.
//!   *Neural Computation*, 12(11), 2519–2535.

pub mod lif;
pub mod router;

pub use lif::LifNeuron;
pub use router::{
    AhlRouter, AHL_NUM_CHANNELS, DomainSignals, RoutingDecision, VerificationDomain,
};

#[cfg(test)]
mod tests;
