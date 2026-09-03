"""Streamlit interface for document-grounded answer comparison."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from config import settings
from engineering import SimulationRequest, parameter_sweep, simulate_thermal_diffusion
from engineering_agent import run_engineering_agent
from rag import (
    ask,
    ask_without_rag,
    indexed_chunk_count,
    ingest,
    retrieve,
    validate_local_models,
)

st.set_page_config(
    page_title="Physics-Grounded Engineering Copilot", page_icon="🧪", layout="wide"
)


@st.cache_resource(show_spinner=False)
def _load_adapter_model(config_path: str, adapter_path: str):
    from fine_tuning.config import load_config
    from fine_tuning.inference import AdapterChatModel

    config = load_config(config_path)
    return AdapterChatModel.from_config(config, adapter_path=adapter_path)


settings.ensure_directories()

st.title("Physics-Grounded Engineering Copilot")
st.caption(
    "Compare document-grounded engineering answers, inspect retrieval evidence, "
    "and run a numerically validated physics tool."
)

with st.sidebar:
    st.header("System")
    problems = validate_local_models()
    if problems:
        for problem in problems:
            st.error(problem)
    else:
        st.success("Ollama and local models are ready")

    try:
        chunk_count = indexed_chunk_count()
        st.metric("Indexed chunks", chunk_count)
    except Exception as exc:  # Keep the UI usable while setup is incomplete.
        chunk_count = 0
        st.error(f"Cannot open ChromaDB: {exc}")

    provider_options = ["Ollama"]
    if os.getenv("OPENAI_API_KEY"):
        provider_options.append("OpenAI")
    provider = st.selectbox("Answer model", provider_options)
    top_k = st.slider("Retrieved passages", 1, 10, 5)
    min_relevance_score = st.slider(
        "Minimum relevance", 0.0, 1.0, float(settings.min_relevance_score), 0.01
    )
    scope = st.text_input("Evidence scope", value=settings.default_scope)
    st.caption(
        f"Operational retrieval: {settings.embedding_model}; "
        f"{settings.chunk_words}-word chunks / "
        f"{settings.chunk_overlap_words}-word overlap"
    )

ask_tab, agent_tab, physics_tab, study_tab, fine_tune_tab = st.tabs(
    [
        "Ask documents",
        "Agent workflow",
        "Physics tool",
        "Retrieval study",
        "Fine-tuning comparison",
    ]
)

with ask_tab:
    st.subheader("1. Build the document index")
    uploads = st.file_uploader(
        "PDF, Word, PowerPoint, Excel, HTML, Markdown, text, or CSV",
        type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "csv"],
        accept_multiple_files=True,
    )

    documents: list[Path] = []
    if uploads:
        for upload in uploads:
            safe_name = Path(upload.name).name
            payload = upload.getvalue()
            digest = hashlib.sha256(payload).hexdigest()[:12]
            destination = settings.uploads_dir / f"{digest}-{safe_name}"
            destination.write_bytes(payload)
            documents.append(destination)

    example_document = settings.example_document
    if example_document:
        use_example = st.checkbox(
            f"Use configured document: {example_document.name}", value=not uploads
        )
        if use_example:
            documents.append(example_document)

    if st.button("Index selected documents", type="primary", disabled=not documents):
        with st.spinner("Docling is parsing and embedding the selected documents…"):
            try:
                count = ingest(documents, scope=scope)
            except Exception as exc:
                st.exception(exc)
            else:
                st.success(f"Indexed {count} chunks from {len(documents)} document(s).")
                st.rerun()

    st.subheader("2. Compare the model before and after retrieval")
    question = st.text_input(
        "Question",
        value="What is the standard approach to model the poroelastic material?",
    )
    if st.button(
        "Compare answers",
        disabled=not question or bool(problems) or chunk_count == 0,
    ):
        with st.spinner("Generating the ungrounded and document-grounded answers…"):
            try:
                baseline = ask_without_rag(question, provider=provider)
                grounded = ask(
                    question,
                    provider=provider,
                    k=top_k,
                    retrieval_hint=baseline,
                    scope=scope,
                    min_relevance_score=min_relevance_score,
                )
            except Exception as exc:
                st.exception(exc)
            else:
                before, after = st.columns(2)
                with before:
                    st.markdown("### Before RAG")
                    st.caption("Model knowledge only; no retrieved document context")
                    st.markdown(baseline)
                with after:
                    st.markdown("### After RAG")
                    st.caption("The same model, grounded in retrieved passages")
                    st.markdown(grounded.text)

                for warning in grounded.warnings:
                    st.warning(warning)

                st.subheader("Evidence used after RAG")
                for source in grounded.sources:
                    page_label = f", page(s) {source.pages}" if source.pages else ""
                    label = (
                        f"[{source.number}] {Path(source.source).name}{page_label} - "
                        f"relevance {source.score:.3f}"
                    )
                    with st.expander(label):
                        st.write(source.excerpt)

with agent_tab:
    st.subheader("Bounded engineering agent")
    st.write(
        "A local model selects one allowlisted action: answer from the scoped "
        "document index or run the validated thermal-diffusion tool. Model output "
        "is parsed as untrusted JSON; arbitrary code and unknown arguments are rejected."
    )
    agent_question = st.text_area(
        "Engineering request",
        value=(
            "Run a one-dimensional thermal-diffusion simulation for 3600 seconds "
            "with diffusivity 1.2e-5 m²/s, length 1 m, and 101 grid points."
        ),
        key="agent_question",
    )
    if st.button(
        "Plan and execute one safe action",
        disabled=not agent_question or bool(problems),
    ):
        with st.spinner("Planning and validating the engineering action…"):
            try:
                outcome = run_engineering_agent(
                    agent_question,
                    scope=scope,
                    k=top_k,
                    min_relevance_score=min_relevance_score,
                )
            except Exception as exc:
                st.exception(exc)
            else:
                st.markdown(f"**Selected action:** `{outcome.decision.action}`")
                st.markdown(outcome.answer)
                if outcome.simulation:
                    st.json(outcome.simulation)
                if outcome.rag_answer:
                    for source in outcome.rag_answer.sources:
                        st.caption(
                            f"[{source.number}] {source.source} — relevance {source.score:.3f}"
                        )

with physics_tab:
    st.subheader("Validated thermal-diffusion simulation")
    st.write(
        "Run a bounded finite-difference simulation campaign. Every result is "
        "checked against a Fourier-series reference, the explicit stability "
        "condition, the discrete residual, and the maximum principle."
    )
    input_columns = st.columns(4)
    diffusivity = input_columns[0].number_input(
        "Diffusivity (m²/s)", 1e-8, 1e-3, 1.2e-5, format="%.2e"
    )
    length = input_columns[1].number_input("Length (m)", 0.01, 20.0, 1.0)
    duration = input_columns[2].number_input(
        "Duration (s)", 1.0, 2.0e7, 3600.0
    )
    points = input_columns[3].number_input(
        "Grid points", 11, 501, 101, step=2
    )
    if st.button("Run validated simulation", type="primary"):
        request = SimulationRequest(
            diffusivity_m2_s=float(diffusivity),
            length_m=float(length),
            duration_s=float(duration),
            points=int(points),
        )
        try:
            result = simulate_thermal_diffusion(request)
        except ValueError as exc:
            st.error(f"Out-of-domain request: {exc}")
        else:
            metrics = st.columns(4)
            metrics[0].metric("Validation", "PASS" if result.validation_passed else "FAIL")
            metrics[1].metric("Relative L2 error", f"{result.relative_l2_error:.3%}")
            metrics[2].metric("Max error", f"{result.max_abs_error_k:.3f} K")
            metrics[3].metric("Fourier number", f"{result.achieved_fourier_number:.3f}")
            frame = pd.DataFrame(
                {
                    "position_m": result.position_m,
                    "finite_difference_k": result.numerical_temperature_k,
                    "analytical_reference_k": result.analytical_temperature_k,
                }
            ).set_index("position_m")
            st.line_chart(frame)
            st.json(result.summary())

    st.markdown("#### Reproducible simulation campaign")
    if st.button("Sweep three material diffusivities"):
        template = SimulationRequest(
            length_m=float(length), duration_s=float(duration), points=int(points)
        )
        try:
            campaign = parameter_sweep((6e-6, 1.2e-5, 2.4e-5), template=template)
        except ValueError as exc:
            st.error(f"Out-of-domain request: {exc}")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "diffusivity_m2_s": result.request.diffusivity_m2_s,
                            "steps": result.steps,
                            "relative_l2_error": result.relative_l2_error,
                            "max_abs_error_k": result.max_abs_error_k,
                            "validation_passed": result.validation_passed,
                        }
                        for result in campaign
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

with study_tab:
    st.subheader("How do retrieval choices influence engineering-answer reliability?")
    st.write(
        "A 50-question, page-grounded benchmark compares four embedding models, "
        "three chunk sizes, three overlaps, and k = 1, 3, 5, and 10. The system "
        "selects a configuration on 35 development questions and reports its "
        "performance on 15 held-out questions."
    )
    results_dir = Path(__file__).resolve().parent / "results"
    selection_path = results_dir / "selected_configuration.json"
    if not selection_path.is_file():
        st.info("Run the retrieval benchmark to populate this study.")
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        chosen = selection["selected_configuration"]
        held_out = {row["k"]: row for row in selection["held_out_test"]}
        configuration_columns = st.columns(2)
        configuration_columns[0].markdown(
            f"**Selected embedding**\n\n`{chosen['model']}`"
        )
        configuration_columns[1].markdown(
            "**Chunk / overlap**\n\n"
            f"`{chosen['chunk_words']} / {chosen['overlap_words']} words`"
        )
        score_columns = st.columns(2)
        score_columns[0].metric(
            "Development Recall@5", f"{chosen['development_recall_at_5']:.1%}"
        )
        score_columns[1].metric(
            "Held-out Recall@5", f"{held_out[5]['recall_at_k']:.1%}"
        )

        for figure_name, caption in (
            ("recall_heatmaps.png", "Development evidence Recall@5 across chunking choices"),
            ("held_out_recall_at_k.png", "Selected configuration on development and held-out questions"),
            ("model_tradeoffs.png", "Held-out result for each model's development-selected chunking"),
        ):
            figure = results_dir / figure_name
            if figure.is_file():
                st.image(str(figure), caption=caption, width="stretch")

        summary_path = results_dir / "retrieval_summary.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            selected_rows = summary[
                (summary["model"] == chosen["model"])
                & (summary["chunk_words"] == chosen["chunk_words"])
                & (summary["overlap_words"] == chosen["overlap_words"])
                & (summary["split"].isin(["development", "test"]))
            ][["split", "k", "questions", "recall_at_k", "page_recall_at_k", "mrr_at_k"]]
            st.dataframe(
                selected_rows.style.format(
                    {
                        "recall_at_k": "{:.1%}",
                        "page_recall_at_k": "{:.1%}",
                        "mrr_at_k": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "Recall requires the retrieved chunk to overlap the reviewed page and "
            "contain every checked evidence term. It measures evidence delivery, "
            "not complete generated-answer correctness."
        )

with fine_tune_tab:
    st.subheader("Base, RAG, fine-tuning, and their combination")
    st.write(
        "This 2x2 comparison uses one Hugging Face base checkpoint, toggles its "
        "LoRA adapter, and reuses the same retrieved passages for both RAG answers."
    )
    aggregate_path = Path(__file__).parent / "results" / "fine_tuning_results.json"
    chart_path = Path(__file__).parent / "results" / "fine_tuning_comparison.png"
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        metric_rows = []
        labels = {
            "base": "Base model",
            "base_rag": "Base model + RAG",
            "fine_tuned": "Base model + fine-tuning",
            "fine_tuned_rag": "Base model + fine-tuning + RAG",
        }
        for key, label in labels.items():
            values = aggregate["metrics"]["variants"][key]
            metric_rows.append(
                {
                    "Condition": label,
                    "Token F1": values["token_f1"],
                    "Valid JSON": values["json_valid"],
                    "Status accuracy": values["json_field.status"],
                    "Tool decision": values["tool_decision_accuracy"],
                }
            )
        st.dataframe(
            pd.DataFrame(metric_rows).style.format(
                {
                    "Token F1": "{:.1%}",
                    "Valid JSON": "{:.1%}",
                    "Status accuracy": "{:.1%}",
                    "Tool decision": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Measured on 48 held-out questions; the retrieval index contains "
            "training-split examples only."
        )
        if chart_path.is_file():
            st.image(str(chart_path))
    config_path = settings.fine_tune_config
    if config_path is None:
        st.info(
            "Set FINE_TUNE_CONFIG (and optionally FINE_TUNE_ADAPTER_PATH) in .env "
            "after training an adapter. See docs/FINE_TUNING.md."
        )
    elif not config_path.is_file():
        st.warning(f"Fine-tuning config not found: {config_path}")
    else:
        from fine_tuning.config import load_config

        fine_tune_config = load_config(config_path)
        fine_collection = str(
            fine_tune_config.evaluation.get(
                "rag_collection_name", settings.collection_name
            )
        )
        fine_rag_config = replace(settings, collection_name=fine_collection)
        fine_scope = str(
            fine_tune_config.evaluation.get("rag_scope", settings.default_scope)
        )
        fine_top_k = int(fine_tune_config.evaluation.get("rag_top_k", 4))
        fine_threshold = fine_tune_config.evaluation.get("rag_min_relevance_score")
        try:
            fine_chunk_count = indexed_chunk_count(fine_rag_config)
        except Exception as exc:
            fine_chunk_count = 0
            st.warning(f"Cannot open the comparison retrieval index: {exc}")
        adapter_path = settings.fine_tune_adapter or fine_tune_config.adapter_dir
        if not adapter_path.is_dir():
            st.warning(f"Fine-tuned adapter not found: {adapter_path}")
        else:
            fine_tune_question = st.text_area(
                "Evaluation query",
                value=(
                    "Derive the weak form of the Poisson equation with mixed "
                    "Dirichlet and Neumann boundary conditions."
                ),
                key="fine_tune_question",
            )
            compare_fine_tuning = st.button(
                "Run four-way comparison",
                disabled=not fine_tune_question or fine_chunk_count == 0,
            )
            if fine_chunk_count == 0:
                st.caption("Build the training-only comparison index to enable RAG.")
            if compare_fine_tuning:
                with st.spinner("Loading the base model and running four answers…"):
                    try:
                        from fine_tuning.comparison import ComparisonRunner

                        model = _load_adapter_model(
                            str(config_path), str(adapter_path)
                        )
                        runner = ComparisonRunner(
                            model,
                            lambda query, k: retrieve(
                                query,
                                k=k,
                                scope=fine_scope,
                                min_relevance_score=fine_threshold,
                                config=fine_rag_config,
                            ),
                        )
                        comparison = runner.run(
                            fine_tune_question, top_k=fine_top_k
                        )
                    except Exception as exc:
                        st.exception(exc)
                    else:
                        answer_by_key = {
                            answer.variant: answer for answer in comparison.answers
                        }
                        for left_key, right_key in (
                            ("base", "base_rag"),
                            ("fine_tuned", "fine_tuned_rag"),
                        ):
                            left, right = st.columns(2)
                            for column, key in ((left, left_key), (right, right_key)):
                                answer = answer_by_key[key]
                                with column:
                                    st.markdown(f"### {answer.label}")
                                    st.caption(
                                        f"Generated in {answer.latency_seconds:.1f} s"
                                    )
                                    st.markdown(answer.text)
                        st.subheader("Shared retrieved evidence")
                        for source in comparison.sources:
                            pages = source.get("pages")
                            page_label = f", page(s) {pages}" if pages else ""
                            label = (
                                f"[{source.get('number')}] {source.get('source')}"
                                f"{page_label}"
                            )
                            with st.expander(label):
                                st.write(source.get("excerpt", ""))
