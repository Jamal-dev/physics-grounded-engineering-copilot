"""Bounded LLM routing between scoped RAG and a validated engineering tool."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import Settings, settings
from engineering import SimulationRequest, simulate_thermal_diffusion
from rag import Answer, ask

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route an engineering request to exactly one allowlisted action. "
            "Return one JSON object and no prose. Use action 'simulate_thermal_diffusion' "
            "only when the user asks to run or predict a one-dimensional heat-diffusion "
            "case. Otherwise use action 'answer_from_evidence'. Simulation arguments may "
            "contain only diffusivity_m2_s, length_m, duration_s, points, "
            "initial_temperature_k, left_temperature_k, and right_temperature_k. Omit "
            "unknown arguments so validated defaults are used.",
        ),
        (
            "human",
            "Request:\n{question}\n\nReturn: "
            '{{"action":"answer_from_evidence","arguments":{{}}}} or '
            '{{"action":"simulate_thermal_diffusion","arguments":{{...}}}}',
        ),
    ]
)

ALLOWED_ACTIONS = {"answer_from_evidence", "simulate_thermal_diffusion"}
SIMULATION_ARGUMENTS = {
    "diffusivity_m2_s",
    "length_m",
    "duration_s",
    "points",
    "initial_temperature_k",
    "left_temperature_k",
    "right_temperature_k",
}
ARGUMENT_CUES = {
    "diffusivity_m2_s": ("diffusivity", "thermal diffusivity"),
    "length_m": ("length", "long"),
    "duration_s": ("duration", "seconds", "minutes", "hours"),
    "points": ("grid points", "nodes", "points"),
    "initial_temperature_k": ("initial temperature", "starting temperature"),
    "left_temperature_k": ("left temperature", "left boundary"),
    "right_temperature_k": ("right temperature", "right boundary"),
}


@dataclass(frozen=True)
class AgentDecision:
    action: str
    arguments: dict[str, float | int]


@dataclass(frozen=True)
class AgentOutcome:
    decision: AgentDecision
    answer: str
    rag_answer: Answer | None = None
    simulation: dict[str, Any] | None = None


def parse_decision(text: str) -> AgentDecision:
    """Parse and strictly validate an untrusted model routing decision."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Planner output must be a JSON object.")
    action = str(value.get("action", ""))
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported agent action: {action!r}")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Planner arguments must be a JSON object.")
    unknown = set(arguments) - SIMULATION_ARGUMENTS
    if unknown:
        raise ValueError(f"Unsupported simulation arguments: {', '.join(sorted(unknown))}")
    if action == "answer_from_evidence" and arguments:
        raise ValueError("Evidence-answer routing cannot contain simulation arguments.")
    checked: dict[str, float | int] = {}
    for key, value_item in arguments.items():
        if isinstance(value_item, bool) or not isinstance(value_item, int | float):
            raise ValueError(f"Simulation argument {key} must be numeric.")
        checked[key] = int(value_item) if key == "points" else float(value_item)
    return AgentDecision(action=action, arguments=checked)


def ground_decision_arguments(question: str, decision: AgentDecision) -> AgentDecision:
    """Drop parameters the planner introduced without a cue in the user request."""
    if decision.action != "simulate_thermal_diffusion":
        return decision
    lowered = question.lower()
    grounded = {
        key: value
        for key, value in decision.arguments.items()
        if any(cue in lowered for cue in ARGUMENT_CUES[key])
    }
    return AgentDecision(action=decision.action, arguments=grounded)


def _planner_completion(question: str, config: Settings) -> str:
    model = ChatOllama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        temperature=0,
        reasoning=False,
        format="json",
        num_ctx=4096,
        num_predict=256,
    )
    response = (PLANNER_PROMPT | model).invoke({"question": question})
    content = response.content
    return content if isinstance(content, str) else json.dumps(content, default=str)


def run_engineering_agent(
    question: str,
    *,
    config: Settings = settings,
    scope: str | None = None,
    k: int = 4,
    min_relevance_score: float | None = None,
    planner: Callable[[str], str] | None = None,
) -> AgentOutcome:
    """Plan one safe action, execute it, and retain inspectable provenance."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    decision_text = planner(question) if planner else _planner_completion(question, config)
    decision = ground_decision_arguments(question, parse_decision(decision_text))
    if decision.action == "answer_from_evidence":
        grounded = ask(
            question,
            provider="Ollama",
            k=k,
            scope=scope,
            min_relevance_score=min_relevance_score,
            config=config,
        )
        return AgentOutcome(decision=decision, answer=grounded.text, rag_answer=grounded)

    request = SimulationRequest(**decision.arguments)
    result = simulate_thermal_diffusion(request)
    summary = result.summary()
    answer = (
        f"The validated thermal-diffusion tool ran {result.steps} stable steps "
        f"(Fourier number {result.achieved_fourier_number:.3f}). The numerical "
        f"solution passed={result.validation_passed}, with relative L2 error "
        f"{result.relative_l2_error:.3%}, maximum error {result.max_abs_error_k:.3f} K, "
        f"and maximum-principle violation {result.maximum_principle_violation_k:.3g} K."
    )
    return AgentOutcome(
        decision=decision,
        answer=answer,
        simulation={"request": asdict(request), **summary},
    )
