<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">spikenaut-router</h1>
<p align="center">SNN-based sparse domain routing — the Anti-Hallucination Layer</p>

<p align="center">
  <a href="https://crates.io/crates/spikenaut-router"><img src="https://img.shields.io/crates/v/spikenaut-router" alt="crates.io"></a>
  <a href="https://docs.rs/spikenaut-router"><img src="https://docs.rs/spikenaut-router/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

A small LIF network acts as a neural router: it classifies incoming signals into domain
categories and activates only the relevant processing pipelines. Only neurons that
actually fire trigger downstream computation — sparse activation by design.

## Features

- `AhlRouter` — 3-neuron LIF bank routing over `Chemistry`, `Mathematics`, `DigitalLogic`
- `DomainSignals::from_text(text)` — keyword-density feature extraction
- `RoutingDecision` — activation mask + per-domain confidence scores
- STDP-based online refinement (correct dispatches potentiate, failures depress)
- Winner-take-all lateral inhibition between domain neurons
- Configurable `MIN_FIRE_RATE` sparse-activation floor

## Installation

```toml
spikenaut-router = "0.1"
```

## Quick Start

```rust
use spikenaut_router::AhlRouter;

let mut router = AhlRouter::new();

let decision = router.route("solve the differential equation dy/dx = sin(x)");
// decision.domain   → Mathematics
// decision.confidence → 0.87
// decision.mask     → [false, true, false]

// Reinforce correct routing (potentiates the Mathematics neuron)
router.reinforce(decision.domain, true);
```

## Routing Model

Each domain neuron integrates keyword-density features from the input:

```
V_i[t+1] = V_i[t] · (1 - β) + Σ_j W_ij · x_j[t] - Σ_k≠i W_inh · V_k[t]
```

Fires when `V_i ≥ θ_i`. STDP reward: `ΔW += η·(1-W)` on correct fire, `ΔW -= η·W` on miss.

*Bi & Poo (1998); Maass (2000) — winner-take-all; Lapicque (1907)*

## Extending to Custom Domains

```rust
use spikenaut_router::{AhlRouter, VerificationDomain};

// Add your own domain by implementing VerificationDomain
// and providing keyword sets for feature extraction
```

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), where it served
as the Anti-Hallucination Layer (AHL) routing verification queries to the correct
neural lobe. Decoupled from lobe-specific logic so it works as a general sparse router
for any classification task.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [spikenaut-encoder](https://github.com/rmems/spikenaut-encoder) | Feature → spike encoding |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | SNN backend abstraction |

## License

GPL-3.0-or-later
