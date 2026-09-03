from __future__ import annotations

import unittest

from engineering import (
    SimulationRequest,
    parameter_sweep,
    simulate_thermal_diffusion,
    validate_request,
)


class ThermalDiffusionTests(unittest.TestCase):
    def test_reference_case_passes_physics_validation(self) -> None:
        result = simulate_thermal_diffusion(SimulationRequest(points=51))
        self.assertTrue(result.validation_passed)
        self.assertLess(result.relative_l2_error, 0.01)
        self.assertLessEqual(result.achieved_fourier_number, 0.5)
        self.assertEqual(result.maximum_principle_violation_k, 0.0)

    def test_out_of_domain_request_is_rejected(self) -> None:
        request = SimulationRequest(diffusivity_m2_s=0.1)
        self.assertTrue(validate_request(request))
        with self.assertRaisesRegex(ValueError, "diffusivity"):
            simulate_thermal_diffusion(request)

    def test_campaign_is_deterministic_and_parameterized(self) -> None:
        first = parameter_sweep((6e-6, 1.2e-5), template=SimulationRequest(points=31))
        second = parameter_sweep((6e-6, 1.2e-5), template=SimulationRequest(points=31))
        self.assertEqual(first, second)
        self.assertEqual(first[0].request.diffusivity_m2_s, 6e-6)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
