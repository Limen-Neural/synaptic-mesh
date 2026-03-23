#[cfg(test)]
mod tests {
    use crate::{AhlRouter, DomainSignals, VerificationDomain};

    #[test]
    fn chemistry_text_activates_chemistry() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Balance the equation: NaOH + HCl → NaCl + H₂O. \
             Find the limiting reactant given 2.5 mol NaOH and 3.0 mol HCl. \
             Calculate the theoretical yield of NaCl."
        );
        assert!(d.is_active(VerificationDomain::Chemistry));
        assert!(!d.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn math_text_activates_mathematics() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Find the derivative of f(x) = sin(x²) at x = π/4. \
             Compute the definite integral from 0 to 1."
        );
        assert!(d.is_active(VerificationDomain::Mathematics));
    }

    #[test]
    fn logic_text_activates_digital_logic() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Simplify F(A,B,C) = A'BC + AB'C + ABC' + ABC using a Karnaugh map. \
             Verify the FSM transition table for determinism."
        );
        assert!(d.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn empty_text_routes_nowhere() {
        let mut router = AhlRouter::new();
        let d = router.route("Hello, how are you today?");
        assert!(d.is_empty());
    }

    #[test]
    fn firing_rates_in_range() {
        let mut router = AhlRouter::new();
        let d = router.route("The stoichiometry of NaOH + HCl reaction involves moles.");
        for &rate in &d.firing_rates {
            assert!(rate >= 0.0 && rate <= 1.0, "firing rate out of range: {rate}");
        }
    }

    #[test]
    fn positive_feedback_increases_weight() {
        let mut router = AhlRouter::new();
        let w_before = router.weight_matrix()[0][0];
        router.apply_feedback(VerificationDomain::Chemistry, 1.0);
        let w_after = router.weight_matrix()[0][0];
        assert!(w_after > w_before);
    }

    #[test]
    fn negative_feedback_decreases_weight() {
        let mut router = AhlRouter::new();
        let w_before = router.weight_matrix()[0][0];
        router.apply_feedback(VerificationDomain::Chemistry, -1.0);
        let w_after = router.weight_matrix()[0][0];
        assert!(w_after < w_before);
    }

    #[test]
    fn domain_signals_extraction_chemistry_dominant() {
        let s = DomainSignals::from_text(
            "The molarity of the NaOH solution is 0.5 M. \
             Calculate the moles needed for the stoichiometry."
        );
        assert!(s.chemistry > 0.0);
        assert!(s.chemistry > s.mathematics);
        assert!(s.chemistry > s.digital_logic);
    }

    #[test]
    fn domain_signals_empty_text_all_zero() {
        let s = DomainSignals::from_text("The sky is blue.");
        assert_eq!(s.chemistry, 0.0);
        assert_eq!(s.mathematics, 0.0);
        assert_eq!(s.digital_logic, 0.0);
    }

    #[test]
    fn total_routes_increments() {
        let mut router = AhlRouter::new();
        assert_eq!(router.total_routes, 0);
        router.route("hello");
        router.route("world");
        assert_eq!(router.total_routes, 2);
    }

    #[test]
    fn verification_domain_index_roundtrip() {
        for d in VerificationDomain::ALL {
            let idx = d.index();
            assert_eq!(VerificationDomain::from_index(idx), Some(d));
        }
        assert_eq!(VerificationDomain::from_index(99), None);
    }

    #[test]
    fn lif_neuron_fires_above_threshold() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.threshold = 0.1;
        n.decay_rate = 0.0; // no leak for this test
        n.integrate(0.5);
        assert!(n.check_fire().is_some());
        assert_eq!(n.membrane_potential, 0.0); // hard reset
    }

    #[test]
    fn lif_neuron_no_fire_below_threshold() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.threshold = 1.0;
        n.integrate(0.05);
        assert!(n.check_fire().is_none());
        assert!(n.membrane_potential > 0.0);
    }
}
