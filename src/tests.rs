use crate::router::AhlRouter;
use crate::neuromod::NeuromodNeuron;

#[test]
fn channel_0_pulse_activates_channel_0() {
    let mut router = AhlRouter::new();
    // Provide a strong pulse on channel 0
    let d = router.route([1.0, 0.0, 0.0]);
    assert!(d.is_active(0), "Channel 0 should be active, firing rate was {:?}", d.firing_rates[0]);
    assert!(!d.is_active(1));
    assert!(!d.is_active(2));
}

#[test]
fn channel_1_pulse_activates_channel_1() {
    let mut router = AhlRouter::new();
    let d = router.route([0.0, 1.0, 0.0]);
    assert!(d.is_active(1));
    assert!(!d.is_active(0));
}

#[test]
fn background_noise_routes_nowhere() {
    let mut router = AhlRouter::new();
    // Weak signals below threshold
    let d = router.route([0.05, 0.05, 0.05]);
    assert!(d.is_empty());
}

#[test]
fn firing_rates_in_range() {
    let mut router = AhlRouter::new();
    let d = router.route([0.8, 0.2, 0.1]);
    for &rate in &d.firing_rates {
        assert!(rate >= 0.0 && rate <= 1.0, "firing rate out of range: {rate}");
    }
}

#[test]
fn positive_feedback_increases_weight() {
    let mut router = AhlRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, 1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after > w_before);
}

#[test]
fn negative_feedback_decreases_weight() {
    let mut router = AhlRouter::new();
    let w_before = router.weight_matrix()[0][0];
    router.apply_feedback(0, -1.0);
    let w_after = router.weight_matrix()[0][0];
    assert!(w_after < w_before);
}

#[test]
fn global_gain_inhibits_firing() {
    let mut router = AhlRouter::new();
    // Normal routing (gain 1.0)
    let d1 = router.route([0.5, 0.0, 0.0]);
    assert!(d1.is_active(0));

    // Reduced gain should inhibit routing
    router.set_global_gain(0.1);
    let d2 = router.route([0.5, 0.0, 0.0]);
    assert!(d2.is_empty(), "Reduced gain should have inhibited firing");
}

#[test]
fn total_routes_increments() {
    let mut router = AhlRouter::new();
    assert_eq!(router.total_routes, 0);
    router.route([0.0, 0.0, 0.0]);
    router.route([0.0, 0.0, 0.0]);
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
