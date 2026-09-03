"""Validated one-dimensional transient thermal-diffusion simulation.

The tool intentionally uses a small, inspectable finite-difference model.  It
is fast enough for an interactive agent, rejects unsupported parameter ranges,
and validates the numerical result against a Fourier-series solution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationRequest:
    diffusivity_m2_s: float = 1.2e-5
    length_m: float = 1.0
    duration_s: float = 3600.0
    points: int = 101
    initial_temperature_k: float = 293.15
    left_temperature_k: float = 373.15
    right_temperature_k: float = 293.15
    fourier_number: float = 0.45


@dataclass(frozen=True)
class SimulationResult:
    request: SimulationRequest
    position_m: tuple[float, ...]
    numerical_temperature_k: tuple[float, ...]
    analytical_temperature_k: tuple[float, ...]
    time_step_s: float
    steps: int
    achieved_fourier_number: float
    relative_l2_error: float
    max_abs_error_k: float
    discrete_residual_l2_k_s: float
    maximum_principle_violation_k: float
    validation_passed: bool

    def summary(self) -> dict[str, float | int | bool | dict[str, float | int]]:
        return {
            "request": asdict(self.request),
            "time_step_s": self.time_step_s,
            "steps": self.steps,
            "achieved_fourier_number": self.achieved_fourier_number,
            "relative_l2_error": self.relative_l2_error,
            "max_abs_error_k": self.max_abs_error_k,
            "discrete_residual_l2_k_s": self.discrete_residual_l2_k_s,
            "maximum_principle_violation_k": self.maximum_principle_violation_k,
            "validation_passed": self.validation_passed,
        }


def validate_request(request: SimulationRequest) -> tuple[str, ...]:
    """Return explicit out-of-domain or numerical-safety violations."""
    errors: list[str] = []
    if not 1e-8 <= request.diffusivity_m2_s <= 1e-3:
        errors.append("diffusivity_m2_s must be between 1e-8 and 1e-3")
    if not 0.01 <= request.length_m <= 20.0:
        errors.append("length_m must be between 0.01 and 20")
    if not 1.0 <= request.duration_s <= 2.0e7:
        errors.append("duration_s must be between 1 and 2e7")
    if not 11 <= request.points <= 501:
        errors.append("points must be between 11 and 501")
    if request.points % 2 == 0:
        errors.append("points must be odd so the centerline is represented")
    for name, value in (
        ("initial_temperature_k", request.initial_temperature_k),
        ("left_temperature_k", request.left_temperature_k),
        ("right_temperature_k", request.right_temperature_k),
    ):
        if not 1.0 <= value <= 2000.0:
            errors.append(f"{name} must be between 1 and 2000 K")
    if not 0.05 <= request.fourier_number <= 0.5:
        errors.append("fourier_number must be between 0.05 and the explicit stability limit 0.5")
    if request.diffusivity_m2_s > 0 and request.length_m > 0 and request.points >= 2:
        dx = request.length_m / (request.points - 1)
        stable_dt = request.fourier_number * dx * dx / request.diffusivity_m2_s
        required_steps = int(np.ceil(request.duration_s / stable_dt))
        if required_steps > 250_000:
            errors.append("request would require more than 250,000 explicit time steps")
    return tuple(errors)


def _analytical_solution(
    request: SimulationRequest, position: np.ndarray, *, modes: int = 400
) -> np.ndarray:
    """Fourier solution for a uniform initial field and fixed end temperatures."""
    x_over_l = position / request.length_m
    steady = request.left_temperature_k + (
        request.right_temperature_k - request.left_temperature_k
    ) * x_over_l
    delta = request.initial_temperature_k - steady
    # Numerically integrate the general initial residual.  This also covers an
    # initial temperature that differs from both boundary temperatures.
    coefficients = []
    for mode in range(1, modes + 1):
        basis = np.sin(mode * np.pi * x_over_l)
        coefficients.append(2.0 * np.trapezoid(delta * basis, x_over_l))
    transient = np.zeros_like(position)
    for mode, coefficient in enumerate(coefficients, start=1):
        transient += coefficient * np.sin(mode * np.pi * x_over_l) * np.exp(
            -request.diffusivity_m2_s
            * (mode * np.pi / request.length_m) ** 2
            * request.duration_s
        )
    result = steady + transient
    result[0] = request.left_temperature_k
    result[-1] = request.right_temperature_k
    return result


def simulate_thermal_diffusion(request: SimulationRequest) -> SimulationResult:
    """Run a stable explicit solve and compare it with an analytical reference."""
    errors = validate_request(request)
    if errors:
        raise ValueError("; ".join(errors))

    position = np.linspace(0.0, request.length_m, request.points)
    dx = float(position[1] - position[0])
    stable_dt = request.fourier_number * dx * dx / request.diffusivity_m2_s
    steps = max(1, int(np.ceil(request.duration_s / stable_dt)))
    dt = request.duration_s / steps
    fourier = request.diffusivity_m2_s * dt / (dx * dx)

    temperature = np.full(request.points, request.initial_temperature_k, dtype=float)
    temperature[0] = request.left_temperature_k
    temperature[-1] = request.right_temperature_k
    previous = temperature.copy()
    for _ in range(steps):
        previous = temperature.copy()
        temperature[1:-1] = previous[1:-1] + fourier * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        )
        temperature[0] = request.left_temperature_k
        temperature[-1] = request.right_temperature_k

    analytical = _analytical_solution(request, position)
    scale = max(
        abs(request.left_temperature_k - request.initial_temperature_k),
        abs(request.right_temperature_k - request.initial_temperature_k),
        1.0,
    )
    relative_l2_error = float(
        np.linalg.norm(temperature - analytical) / (np.sqrt(request.points) * scale)
    )
    max_abs_error = float(np.max(np.abs(temperature - analytical)))
    residual = (
        (temperature[1:-1] - previous[1:-1]) / dt
        - request.diffusivity_m2_s
        * (previous[2:] - 2.0 * previous[1:-1] + previous[:-2])
        / (dx * dx)
    )
    residual_l2 = float(np.linalg.norm(residual) / np.sqrt(residual.size))
    lower = min(
        request.initial_temperature_k,
        request.left_temperature_k,
        request.right_temperature_k,
    )
    upper = max(
        request.initial_temperature_k,
        request.left_temperature_k,
        request.right_temperature_k,
    )
    maximum_principle_violation = float(
        max(0.0, lower - float(temperature.min()), float(temperature.max()) - upper)
    )
    passed = (
        fourier <= 0.5 + 1e-12
        and relative_l2_error <= 0.01
        and residual_l2 <= 1e-9
        and maximum_principle_violation <= 1e-10
    )
    return SimulationResult(
        request=request,
        position_m=tuple(float(value) for value in position),
        numerical_temperature_k=tuple(float(value) for value in temperature),
        analytical_temperature_k=tuple(float(value) for value in analytical),
        time_step_s=float(dt),
        steps=steps,
        achieved_fourier_number=float(fourier),
        relative_l2_error=relative_l2_error,
        max_abs_error_k=max_abs_error,
        discrete_residual_l2_k_s=residual_l2,
        maximum_principle_violation_k=maximum_principle_violation,
        validation_passed=passed,
    )


def parameter_sweep(
    diffusivities_m2_s: Iterable[float],
    *,
    template: SimulationRequest | None = None,
) -> tuple[SimulationResult, ...]:
    """Execute a deterministic simulation campaign over material diffusivity."""
    base = template or SimulationRequest()
    return tuple(
        simulate_thermal_diffusion(
            SimulationRequest(
                diffusivity_m2_s=float(diffusivity),
                length_m=base.length_m,
                duration_s=base.duration_s,
                points=base.points,
                initial_temperature_k=base.initial_temperature_k,
                left_temperature_k=base.left_temperature_k,
                right_temperature_k=base.right_temperature_k,
                fourier_number=base.fourier_number,
            )
        )
        for diffusivity in diffusivities_m2_s
    )
