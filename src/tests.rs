use crate::router::{ChannelRouter, RouterConfig};
use crate::neuromod::NeuromodNeuron;

#[test]
fn channel_0_pulse_activates_channel_0() {
    let mut router = ChannelRouter::new();
    // Provide a strong pulse on channel 0
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
    // Weak signals below threshold
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
    // Normal routing (gain 1.0)
    let d1 = router.route(&[0.5, 0.0, 0.0]).unwrap();
    assert!(d1.is_active(0));

    // Reduced gain should inhibit routing
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
    n.leak = 0.0; // no leak for this test
    n.integrate(0.5);
    assert!(n.check_fire().is_some());
    assert_eq!(n.v, 0.0); // hard reset
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
    // AhlRouter is a type alias for ChannelRouter
    let mut router = crate::router::AhlRouter::new();
    let d = router.route(&[1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
}

#[test]
fn route_accepts_array_by_value() {
    // AsRef<[f32]> allows passing [f32; 3] directly
    let mut router = ChannelRouter::new();
    let d = router.route([1.0, 0.0, 0.0]).unwrap();
    assert!(d.is_active(0));
}

#[test]
fn route_rejects_mismatched_length() {
    let mut router = ChannelRouter::new();
    let result = router.route(&[1.0, 0.0]); // 2 elements, expected 3
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
