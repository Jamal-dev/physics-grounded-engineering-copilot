"""Build the reviewed 50-question benchmark from private evidence passages."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from experiments.common import DEFAULT_CORPUS, DEFAULT_QUESTIONS, read_jsonl, write_jsonl
from experiments.generate_questions import select_evidence

# Each item was manually checked against its selected source passage. Terms are
# exact, case-insensitive substrings used as a reproducible evidence sanity check.
CURATED = [
    (
        "How are the inverse coefficients 1/H and 1/R interpreted in isotropic soil consolidation?",
        "1/H measures soil compressibility under water-pressure change; 1/R measures water-content change under that pressure.",
        ["compressibility", "water content", "water pressure"],
    ),
    (
        "How did Heinrich and Desoyer change their porous-skeleton assumptions between their 1955 and 1956 groundwater-flow models?",
        "They moved from grains touching only at points to a general solid-matrix description.",
        ["grains", "solid matrices", "ground-water flow"],
    ),
    (
        "Why does the treatment of the unjacketed test in Biot and Willis raise concerns about its constitutive parameters?",
        "It mixes partial strain and pore-fluid quantities with real fluid compressibility, obscuring their mechanical correspondence.",
        ["unjacketed test", "partial strain", "real compressibility"],
    ),
    (
        "How are grain-induced forces represented when Heinrich and Desoyer pass from micro-flow to macro-flow?",
        "They distribute the grain forces over the mixture volume as quasi-volume forces.",
        ["micro flow", "quasi-volume forces", "whole volume"],
    ),
    (
        "What physical behavior of soil particles is omitted when water motion is modeled only with Darcy's law?",
        "Darcy-only models omit elastic compression and extension of the soil particles.",
        ["darcy's law", "elastically compressed", "extended"],
    ),
    (
        "Why are Derski's solid and fluid velocities easier to validate experimentally than Biot's fluid velocity variable?",
        "Derski's phase velocities are directly measurable, whereas Biot's variable is a barycentric velocity of free and trapped fluid.",
        ["directly measured", "barycentric velocity", "trapped fluid"],
    ),
    (
        "What quantity did Schleicher use to characterize the limit of the elastic range in frictional materials?",
        "He used the total deformation work stored per unit volume.",
        ["elastic range", "mechanical work", "space unit"],
    ),
    (
        "Why can the plastic strain tensor fail to represent deformation history?",
        "Loading and unloading along the same path can leave zero plastic strain despite a nontrivial history.",
        ["unloading", "loading", "turn out to be zero"],
    ),
    (
        "What limitation of open yield surfaces was the cap model designed to address?",
        "It closes the hardening range along the hydrostatic direction in the triaxial plane.",
        ["cap model", "hydrostatic", "hardening range"],
    ),
    (
        "How is constituent incompressibility imposed in Mills's two-fluid mixture formulation?",
        "Real constituent densities are constant and the volume fractions sum to one.",
        ["real densities", "constant", "sum of the volume fractions"],
    ),
    (
        "What classical stress system illustrates the balance of equilibrated force in voided elastic materials?",
        "A double-force system without moments: two opposite forces at the same point with no net force or moment.",
        ["double force", "no net force", "no resulting moment"],
    ),
    (
        "What constraint follows from combining incompressible constituents with the volume-fraction concept in saturated porous media?",
        "The constituent volume fractions must sum to one, introducing a saturation constraint.",
        ["incompressible", "saturation condition", "equal to one"],
    ),
    (
        "How does volume averaging connect microscale constituent fields to a continuum porous-medium description?",
        "It integrates microscopic quantities over an averaging volume to produce macroscopic field quantities.",
        ["volume averaging", "microscopic quantity", "macroscopic quantities"],
    ),
    (
        "When must fluid reference positions belong to the solid phase's reference placement?",
        "Only for deformation processes in which the fluid phases leave the solid control space.",
        ["reference positions", "fluid phases leave", "control space"],
    ),
    (
        "Why does incompressibility of individual grains not make a granular skeleton macroscopically incompressible?",
        "Contact forces can change pore structure and volume fraction without changing individual grain volume.",
        ["contact forces", "pore structure", "volume fraction"],
    ),
    (
        "Why is the porous-solid deformation gradient decomposed multiplicatively into two parts?",
        "The decomposition transfers incompatible microscale structural changes into an integrable macroscale deformation description.",
        ["multiplicatively", "microscale", "macroscale"],
    ),
    (
        "Which balance laws must be written for every constituent in a porous-medium mixture?",
        "Mass, momentum, moment of momentum, and energy balances, including external and interaction effects.",
        ["balance of mass", "balance of momentum", "balance of energy"],
    ),
    (
        "Why is entropy especially difficult to interpret for irreversible processes?",
        "Its reversible-process meaning is clearer, while irreversible formulations remain conceptually obscure and admit competing interpretations.",
        ["irreversible processes", "mysterious concept", "reversible processes"],
    ),
    (
        "Why does introducing solid and fluid volume fractions create a closure problem in porous-media theory?",
        "The added volume-fraction fields leave two field equations missing from the otherwise closed mixture theory.",
        ["two field equations", "volume fractions", "closure"],
    ),
    (
        "Which four continuum-mechanics principles should porous-medium constitutive equations satisfy beyond matching experiments?",
        "Determinism, local action, material objectivity, and dissipation.",
        ["determinism", "local action", "material objectivity", "dissipation"],
    ),
    (
        "Why can a purely formal evaluation of the entropy inequality yield an unreliable porous-material model?",
        "It may satisfy the inequality mathematically while failing to reproduce experimentally observed physical phenomena.",
        ["purely stereotyped", "experiment", "physical phenomena"],
    ),
    (
        "Which constituent-compressibility combinations require separate entropy-inequality treatments in a binary porous medium?",
        "Both incompressible, both compressible, solid-only compressible, and fluid-only compressible cases.",
        ["both constituents", "incompressible", "compressible"],
    ),
    (
        "Why must the saturation-constraint multiplier be determined constitutively in the compressible binary porous model?",
        "A constitutive multiplier is needed to impose saturation without leaving the field-equation system unclosed.",
        ["saturation condition", "multiplier", "closure problem"],
    ),
    (
        "What constituent assumptions define the first-type hybrid binary model considered for thermoelastic porous solids?",
        "It combines a compressible thermoelastic porous solid with an incompressible viscous fluid.",
        ["compressible thermoelastic", "incompressible viscous fluid", "hybrid binary"],
    ),
    (
        "Which developments provided successful constitutive descriptions of finite elastic distortions?",
        "The finite-elasticity work of Mooney, Rivlin, and Rivlin with Saunders supplied successful nonlinear laws.",
        ["mooney", "rivlin", "finite"],
    ),
    (
        "Why is a separate history tensor introduced for hardening rather than using plastic strain alone?",
        "Plastic strain can be displacement-determined or return to zero after reverse loading, losing history information.",
        ["history", "plastic strain", "unloading path"],
    ),
    (
        "Why do granular and brittle skeletons require different plasticity models from metals?",
        "Their extension and compression responses differ strongly and their yielding may include failure and isotropic hardening.",
        ["granular", "brittle", "extension", "compression"],
    ),
    (
        "What practical advantage does the proposed single-surface yield condition have over a cap model?",
        "One surface covers onset, hardening, and failure without a separate cap in the hydrostatic plane.",
        ["single-surface", "cap-model", "hydrostatic plane"],
    ),
    (
        "How does the information needed for elastic response differ from that needed for plastic response in porous media?",
        "Elastic response uses the current deformation gradient; plastic response requires the complete deformation process.",
        ["deformation gradient", "plastic response", "total deformation process"],
    ),
    (
        "What four ingredients form the classical plasticity framework described for porous skeletons?",
        "A yield condition, consistency condition, loading criteria, and flow rule.",
        ["yield condition", "consistency condition", "loading criteria", "flow rule"],
    ),
    (
        "Which Mohr-Coulomb parameter can be used to describe the hardening process?",
        "The angle of internal friction can serve as a hardening parameter.",
        ["angle of internal friction", "hardening process", "mohr-coulomb"],
    ),
    (
        "Why is a plastic-potential flow rule common in geomechanical software despite its drawbacks?",
        "It is easy to fit to measured strain-rate directions and widely implemented, though complex for nonlinear boundary problems.",
        ["plastic potential", "software", "boundary-value problems"],
    ),
    (
        "What transport processes follow the constitutive relation for an incompressible viscous fluid in a rigid porous solid?",
        "The formulation next treats heat transport and fluid mass transport.",
        ["incompressible viscous fluid", "heat transport", "mass transport"],
    ),
    (
        "How should capillary and friction forces enter the local momentum balance of a saturated porous medium?",
        "Both are phase-interaction forces; friction is a volume force aligned with relative fluid-solid velocity.",
        ["capillary force", "interaction force", "relative velocity"],
    ),
    (
        "What relation between phase velocity and pressure gradient yields Fick's first diffusion law for an ideal gas?",
        "The gas-solid difference velocity is proportional to the gas-pressure gradient.",
        ["difference velocity", "gradient of the gas pressure", "fick's first"],
    ),
    (
        "What is the standard approach to model the poroelastic material?",
        "Most practical problems use Biot's original linear poroelasticity theory or parts of it.",
        ["practical problems", "biot's original poroelasticity", "parts of this theory"],
    ),
    (
        "Which three interaction effects are fundamental in liquid-saturated porous solids?",
        "Uplift, friction, and capillarity.",
        ["uplift", "friction", "capillarity"],
    ),
    (
        "What wave modes does Biot's compressible-constituent theory conventionally predict in fluid-saturated porous media?",
        "Two dilatational waves and one rotational wave.",
        ["two dilatational waves", "one rotational wave", "biot's"],
    ),
    (
        "Why does the one-dimensional saturated-medium response resemble viscoelastic behavior?",
        "Water drainage and internal friction make responses depend on time and prior loading history.",
        ["loading history", "squeezing out of water", "internal friction"],
    ),
    (
        "How many independent dilatational waves remain for sinusoidal loading when both phases are incompressible?",
        "Only one; solid displacement can be expressed through fluid displacement and vice versa.",
        ["only one", "dilatational wave", "solid displacement"],
    ),
    (
        "How do the solid and pore liquid respond over time during step-load consolidation?",
        "The solid moves downward while liquid is squeezed from the pores, with frictional viscoelastic behavior.",
        ["moves downwards", "liquid is squeezed out", "internal friction"],
    ),
    (
        "How do solid extra stress and pore pressure evolve during the reported step-load consolidation?",
        "Extra stress increases with time at fixed depth, while pore pressure eventually falls to zero.",
        ["increase with time", "pore pressure", "zero"],
    ),
    (
        "What indicates elastic recovery after an impulse load on the saturated porous column?",
        "Near-surface pore-liquid pressure changes from positive to negative as the skeleton recovers and absorbs water.",
        ["positive to negative", "elastic recovery", "water absorption"],
    ),
    (
        "Which assumptions make the exact transient column solution produce one shared dilatational-wave speed?",
        "Both constituents are incompressible and the model is geometrically linear with nearly fixed volume fractions.",
        ["two incompressible", "geometrically-linear", "one independent dilatational wave"],
    ),
    (
        "Why is load-induced effective pressure treated as an additional configuration pressure in one-dimensional consolidation?",
        "That interpretation simplifies the boundary conditions because the additional configuration pressure is zero in the control space.",
        ["configuration pressure", "boundary conditions", "zero"],
    ),
    (
        "Why must a von-Mises-type model be extended for metallic-powder compaction?",
        "Powder porosity permits large plastic volume changes, unlike classical incompressible metal plasticity.",
        ["von-mises", "volume changes", "porosity"],
    ),
    (
        "Why is positive definiteness of the elastic-plastic response tensor important for boundary-value problems?",
        "It supports uniqueness proofs and minimum or maximum principles for their solutions.",
        ["positive definite", "uniqueness", "minimum/maximum"],
    ),
    (
        "When can gas-filled powder compaction be interpreted as deformation of an empty porous solid?",
        "When gas density is made negligible so the calculated pore-gas pressure is zero.",
        ["poregas pressure", "equal to zero", "empty porous solid"],
    ),
    (
        "What displacement pattern identifies a Rayleigh surface wave at a free boundary?",
        "Two extreme peaks in horizontal solid displacement travel along the free boundary.",
        ["two extreme peaks", "free boundary", "rayleigh-wave"],
    ),
    (
        "What is the main theoretical difficulty that a consistent porous-media formulation must overcome?",
        "It must close the field equations while preserving mechanics, thermodynamics, and experimentally known phenomena.",
        ["closure problem", "thermodynamics", "experiment"],
    ),
]


def curate(corpus_path: Path, output: Path) -> list[dict[str, object]]:
    corpus = read_jsonl(corpus_path)
    selected = select_evidence(corpus, 50, 298, 483)
    # The user's motivating question is grounded in the explicit Applications
    # statement on page 430 rather than the automatically selected nearby passage.
    selected[35] = next(row for row in corpus if row["sequence"] == 1360)
    if len(CURATED) != len(selected):
        raise RuntimeError("Curated question count does not match selected evidence")

    rows: list[dict[str, object]] = []
    for index, (evidence, item) in enumerate(zip(selected, CURATED, strict=False), start=1):
        question, answer, terms = item
        evidence_lower = re.sub(
            r"\s+", " ", evidence["text"].lower().replace("\u00ad ", "").replace("\u00ad", "")
        )
        missing = [
            term
            for term in terms
            if re.sub(r"\s+", " ", term.lower()) not in evidence_lower
        ]
        if missing:
            raise RuntimeError(f"Q{index:03d} has missing evidence terms: {missing}")
        rows.append(
            {
                "id": f"Q{index:03d}",
                "question": question,
                "answer_summary": answer,
                "source": evidence["source"],
                "expected_pages": evidence["pages"],
                "evidence_terms": terms,
                "evidence_sha256": hashlib.sha256(
                    evidence["text"].encode("utf-8")
                ).hexdigest(),
                "split": "test" if index % 10 in {0, 3, 7} else "development",
            }
        )
    write_jsonl(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    rows = curate(args.corpus.expanduser().resolve(), args.output.expanduser().resolve())
    dev = sum(row["split"] == "development" for row in rows)
    test = len(rows) - dev
    print(f"Wrote {len(rows)} reviewed questions ({dev} development, {test} test)")


if __name__ == "__main__":
    main()
