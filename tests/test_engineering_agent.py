from __future__ import annotations

import unittest

from engineering_agent import (
    AgentDecision,
    ground_decision_arguments,
    parse_decision,
    run_engineering_agent,
)


class EngineeringAgentTests(unittest.TestCase):
    def test_parser_accepts_only_allowlisted_numeric_simulation_arguments(self) -> None:
        decision = parse_decision(
            '{"action":"simulate_thermal_diffusion","arguments":'
            '{"diffusivity_m2_s":0.000012,"points":51}}'
        )
        self.assertEqual(decision.action, "simulate_thermal_diffusion")
        self.assertEqual(decision.arguments["points"], 51)
        with self.assertRaisesRegex(ValueError, "Unsupported simulation arguments"):
            parse_decision(
                '{"action":"simulate_thermal_diffusion","arguments":{"command":"rm"}}'
            )

    def test_agent_executes_validated_tool_from_injected_plan(self) -> None:
        outcome = run_engineering_agent(
            "Run the reference heat-diffusion case.",
            planner=lambda _: (
                '{"action":"simulate_thermal_diffusion","arguments":{"points":51}}'
            ),
        )
        self.assertEqual(outcome.decision.action, "simulate_thermal_diffusion")
        self.assertIsNotNone(outcome.simulation)
        self.assertTrue(outcome.simulation["validation_passed"])

    def test_agent_drops_planner_parameters_not_stated_by_user(self) -> None:
        decision = AgentDecision(
            action="simulate_thermal_diffusion",
            arguments={"points": 51, "left_temperature_k": 300.0},
        )
        grounded = ground_decision_arguments("Use 51 grid points.", decision)
        self.assertEqual(grounded.arguments, {"points": 51})

    def test_agent_rejects_out_of_domain_tool_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "diffusivity"):
            run_engineering_agent(
                "Run an impossible case with diffusivity 0.1 m2/s.",
                planner=lambda _: (
                    '{"action":"simulate_thermal_diffusion",'
                    '"arguments":{"diffusivity_m2_s":0.1}}'
                ),
            )


if __name__ == "__main__":
    unittest.main()
