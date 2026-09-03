"""Command-line interface for the validated physics tool."""

from __future__ import annotations

import argparse
import json

from engineering import SimulationRequest, parameter_sweep, simulate_thermal_diffusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusivity", type=float, default=1.2e-5)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--sweep", type=float, nargs="*")
    args = parser.parse_args()
    request = SimulationRequest(
        diffusivity_m2_s=args.diffusivity,
        length_m=args.length,
        duration_s=args.duration,
        points=args.points,
    )
    if args.sweep:
        payload = [result.summary() for result in parameter_sweep(args.sweep, template=request)]
    else:
        payload = simulate_thermal_diffusion(request).summary()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
