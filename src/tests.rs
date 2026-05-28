use crate::router::{ChannelRouter, RouterConfig, NeuromodState};
use crate::neuromod::NeuromodNeuron;

#[test]
fn channel_0_pulse_activates_channel_0() {
    let mut router = ChannelRouter::new();
    let d = router.route(&[1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0), "Channel 0 should be active, firing rate was {:?}", d.firing_rates[0]);
    assert!(!d.is_active(1));
    assert!(!d.is_active(2));
}

#[test]
fn channel_1_pulse_activates_channel_1() {
    let mut router = ChannelRouter::new();
    let d = router.route(&[0.0, 1.0, 0.0]).unwrap();
    assert!(d.is_active(1));
    assert!(!d.is_active(0));
}

#[test]
fn background_noise_routes_nowhere() {
    let mut router = ChannelRouter::new();
    let d = router.route(&[0.05, 0.05, 0.05]).unwrap();
    assert!(d.is_empty());
}

#[test]
fn firing_rates_in_range() {
    let mut router = ChannelRouter::new();
    let d = router.route(&[0.8, 0.2, 0.1]).unwrap();
    for &rate in &d.firing_rates {
        assert!(rate >= 0.0 && rate <= 1.0, "firing rate out of range: {rate}");
    }
}

#[test]
fn positive_feedback_increases_weight() {
    let mut router = ChannelRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, 1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after > w_before);
}

#[test]
fn negative_feedback_decreases_weight() {
    let mut router = ChannelRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, -1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after < w_before);
}

#[test]
fn global_gain_inhibits_firing() {
    let mut router = ChannelRouter::new();
    let d1 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    assert!(d1.is_active(0));

    router.set_global_gain(0.1);
    let d2 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    assert!(d2.is_empty(), "Reduced gain should have inhibited firing");
}

#[test]
fn total_routes_increments() {
    let mut router = ChannelRouter::new();
    assert_eq!(router.total_routes, 0);
    router.route(&[0.0, 0.0, 0.0]).unwrap();
    router.route(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(router.total_routes, 2);
}

#[test]
fn neuromod_neuron_fires_above_threshold() {
    let mut n = NeuromodNeuron::new();
    n.threshold = 0.1;
    n.leak = 0.0;
    n.integrate(0.5);
    assert!(n.check_fire().is_some());
    assert_eq!(n.v, 0.0);
}

#[test]
fn neuromod_neuron_no_fire_below_threshold() {
    let mut n = NeuromodNeuron::new();
    n.threshold = 1.0;
    n.integrate(0.05);
    assert!(n.check_fire().is_none());
    assert!(n.v > 0.0);
}

// ── Generic channel count tests ───────────────────────────────────────────────

#[test]
fn five_channel_router_routes_correctly() {
    let config = RouterConfig {
        channel_count: 5,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router.route(&[1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
    assert!(!d.is_active(1));
    assert!(!d.is_active(4));
    assert_eq!(d.firing_rates.len(), 5);
}

#[test]
fn eight_channel_router_routes_correctly() {
    let config = RouterConfig {
        channel_count: 8,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router.route(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(3));
    assert!(!d.is_active(0));
    assert_eq!(d.firing_rates.len(), 8);
}

#[test]
fn single_channel_router_always_routes() {
    let config = RouterConfig {
        channel_count: 1,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let d = router.route(&[1.0]).unwrap();
    assert!(d.is_active(0));
    assert_eq!(d.firing_rates.len(), 1);
}

#[test]
fn custom_config_weights_applied() {
    let config = RouterConfig {
        channel_count: 3,
        self_weight: 1.2,
        cross_weight: -0.2,
        ..RouterConfig::default()
    };
    let router = ChannelRouter::with_config(config);
    let m = router.weight_matrix();
    assert!((m[0][0] - 1.2).abs() < 1e-6);
    assert!((m[0][1] - (-0.2)).abs() < 1e-6);
}

#[test]
fn backward_compat_ahl_router_still_works() {
    let mut router = crate::router::AhlRouter::new();
    let d = router.route(&[1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
}

#[test]
fn route_accepts_array_by_value() {
    let mut router = ChannelRouter::new();
    let d = router.route([1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
}

#[test]
fn route_rejects_mismatched_length() {
    let mut router = ChannelRouter::new();
    let result = router.route(&[1.0, 0.0]);
    assert!(result.is_err());
}

#[test]
#[should_panic(expected = "routing_timesteps must be > 0")]
fn zero_routing_timesteps_panics() {
    let config = RouterConfig {
        routing_timesteps: 0,
        ..RouterConfig::default()
    };
    let _router = ChannelRouter::with_config(config);
}

// ── Neuromodulatory routing tests ─────────────────────────────────────────────

#[test]
fn dopamine_increases_channel_conductance() {
    let mut router = ChannelRouter::new();
    // Baseline: weak signal should not activate.
    let d1 = router.route(&[0.15, 0.0, 0.0]).unwrap();
    let baseline_active = d1.is_active(0);

    // With dopamine: same weak signal should have lower threshold.
    let mods = NeuromodState { dopamine: 0.8, ..NeuromodState::default() };
    let d2 = router.route_modulated(&[0.15, 0.0, 0.0], &mods).unwrap();
    // Dopamine makes it easier to fire, so if it wasn't active before,
    // it might be now. If it was active before, it should still be.
    if !baseline_active {
        // Dopamine should help weak signal cross threshold.
        assert!(d2.is_active(0) || d2.firing_rates[0] > d1.firing_rates[0],
            "Dopamine should increase conductance");
    }
}

#[test]
fn cortisol_increases_channel_resistance() {
    let mut router = ChannelRouter::new();
    // Baseline: moderate signal should activate.
    let d1 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    let baseline_rate = d1.firing_rates[0];

    // With cortisol: same signal should have higher threshold.
    let mods = NeuromodState { cortisol: 0.8, ..NeuromodState::default() };
    let d2 = router.route_modulated(&[0.5, 0.0, 0.0], &mods).unwrap();
    // Cortisol should reduce firing rate.
    assert!(d2.firing_rates[0] <= baseline_rate,
        "Cortisol should increase resistance (reduce firing rate)");
}

#[test]
fn serotonin_increases_leak_reduces_persistence() {
    let mut router = ChannelRouter::new();
    // Baseline firing rate.
    let d1 = router.route(&[0.8, 0.0, 0.0]).unwrap();
    let baseline_rate = d1.firing_rates[0];

    // With serotonin: higher leak should reduce firing.
    let mods = NeuromodState { serotonin: 0.8, ..NeuromodState::default() };
    let d2 = router.route_modulated(&[0.8, 0.0, 0.0], &mods).unwrap();
    // Serotonin should reduce firing rate due to higher leak.
    assert!(d2.firing_rates[0] <= baseline_rate,
        "Serotonin should increase leak (reduce persistence)");
}

#[test]
fn active_channel_strengthens_with_dopamine() {
    let config = RouterConfig {
        plasticity_potentiate: 0.1,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);
    let w_before = router.weight_matrix()[0][0];

    // Route with high dopamine — channel 0 should activate and strengthen.
    let mods = NeuromodState { dopamine: 1.0, ..NeuromodState::default() };
    let d = router.route_modulated(&[1.0, 0.0, 0.0], &mods).unwrap();
    assert!(d.is_active(0), "Channel 0 should be active");

    let w_after = router.weight_matrix()[0][0];
    assert!(w_after > w_before,
        "Active channel should strengthen with dopamine: {w_before} -> {w_after}");
}

#[test]
fn inactive_channel_weakens_over_time() {
    let config = RouterConfig {
        plasticity_decay: 0.1,
        plasticity_potentiate: 0.2,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);

    // First activate channel 1 to strengthen it above baseline.
    let _ = router.route(&[0.0, 1.0, 0.0]).unwrap();
    let w_strengthened = router.weight_matrix()[1][1];
    assert!(w_strengthened > 0.9, "Channel 1 should strengthen after activation");

    // Now route multiple times with signal only on channel 0.
    // Channel 1 is inactive and should decay toward baseline.
    for _ in 0..10 {
        let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();
    }

    let w_after = router.weight_matrix()[1][1];
    assert!(w_after < w_strengthened,
        "Inactive channel should weaken (use-it-or-lose-it): {w_strengthened} -> {w_after}");
}

#[test]
fn fatigue_accumulates_with_use() {
    let mut router = ChannelRouter::new();
    let fatigue_before = router.channel_fatigue[0];

    // Route with strong signal on channel 0.
    let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();

    let fatigue_after = router.channel_fatigue[0];
    assert!(fatigue_after > fatigue_before,
        "Fatigue should accumulate with activation: {fatigue_before} -> {fatigue_after}");
}

#[test]
fn fatigue_recovery_when_inactive() {
    let mut router = ChannelRouter::new();
    // Activate channel 0.
    let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();
    let fatigue_after_active = router.channel_fatigue[0];
    assert!(fatigue_after_active > 0.0);

    // Route with signal on channel 1 (channel 0 inactive).
    let _ = router.route(&[0.0, 1.0, 0.0]).unwrap();
    let fatigue_after_inactive = router.channel_fatigue[0];
    assert!(fatigue_after_inactive < fatigue_after_active,
        "Fatigue should recover when inactive: {fatigue_after_active} -> {fatigue_after_inactive}");
}

#[test]
fn cortisol_amplifies_fatigue_effect() {
    let mut router = ChannelRouter::new();
    // Build up some fatigue on channel 0.
    for _ in 0..3 {
        let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();
    }
    let fatigue = router.channel_fatigue[0];
    assert!(fatigue > 0.0);

    // Baseline rate without cortisol.
    let d1 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    let rate_no_stress = d1.firing_rates[0];

    // With cortisol: fatigue should have stronger effect.
    let mods = NeuromodState { cortisol: 1.0, ..NeuromodState::default() };
    let d2 = router.route_modulated(&[0.5, 0.0, 0.0], &mods).unwrap();
    let rate_stressed = d2.firing_rates[0];

    assert!(rate_stressed <= rate_no_stress,
        "Cortisol should amplify fatigue effect: {rate_no_stress} -> {rate_stressed}");
}

#[test]
fn dopamine_counteracts_fatigue() {
    let mut router = ChannelRouter::new();
    // Build up fatigue on channel 0.
    for _ in 0..5 {
        let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();
    }
    let fatigue = router.channel_fatigue[0];
    assert!(fatigue > 0.3, "Should have significant fatigue");

    // Baseline rate with fatigue.
    let d1 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    let rate_no_dopamine = d1.firing_rates[0];

    // With dopamine: should counteract fatigue.
    let mods = NeuromodState { dopamine: 1.0, ..NeuromodState::default() };
    let d2 = router.route_modulated(&[0.5, 0.0, 0.0], &mods).unwrap();
    let rate_dopamine = d2.firing_rates[0];

    assert!(rate_dopamine >= rate_no_dopamine,
        "Dopamine should counteract fatigue: {rate_no_dopamine} -> {rate_dopamine}");
}

#[test]
fn least_resistance_pathway_routing() {
    let config = RouterConfig {
        channel_count: 3,
        fatigue_accumulation: 0.4,
        fatigue_recovery: 0.02,
        threshold: 0.15,
        ..RouterConfig::default()
    };
    let mut router = ChannelRouter::with_config(config);

    let low_cortisol = NeuromodState { cortisol: 0.3, ..NeuromodState::default() };

    let d_baseline = router.route_modulated(&[0.3, 0.3, 0.3], &low_cortisol).unwrap();
    let rate_baseline = d_baseline.firing_rates[0];

    for _ in 0..10 {
        let _ = router.route(&[1.0, 0.0, 0.0]).unwrap();
    }
    let fatigue_ch0 = router.channel_fatigue[0];
    assert!(fatigue_ch0 > 0.8, "Channel 0 should be heavily fatigued: {fatigue_ch0}");

    let d_fatigued = router.route_modulated(&[0.3, 0.3, 0.3], &low_cortisol).unwrap();
    let rate_fatigued = d_fatigued.firing_rates[0];

    assert!(rate_fatigued < rate_baseline,
        "Fatigued channel 0 should have lower firing rate: {rate_baseline} -> {rate_fatigued}");
}
