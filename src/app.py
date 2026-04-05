import os
import pickle
import queue
import re
import base64
import threading
import time
from collections.abc import Callable

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from agent import EDAAgent
from config import ModelConfig, load_model_config
from eda_agents import (
    run_multivariate_graphical_agent,
    run_multivariate_nongraphical_agent,
    run_univariate_graphical_agent,
    run_univariate_nongraphical_agent,
)
from pipeline import run_pipeline

load_dotenv()

STAGE_LABELS = {
    "planning": "Stage 1/5 — Planner: building analysis plan…",
    "hypothesizing": "Stage 2/5 — Hypothesis: generating testable hypotheses…",
    "executing": "Stage 3/5 — Executor: running analyses…",
    "critiquing": "Stage 4/5 — Critic: validating results…",
    "synthesizing": "Stage 5/5 — Synthesizer: crafting insights…",
}

STAGE_WEIGHTS = {
    "planning": 0.08,
    "hypothesizing": 0.18,
    "executing": 0.60,
    "critiquing": 0.78,
    "synthesizing": 0.92,
}

STAGE_ICONS = {
    "planning": "🗺",
    "hypothesizing": "💡",
    "executing": "⚙",
    "critiquing": "🔬",
    "synthesizing": "📝",
    "why_depth_1": "🔍",
    "why_depth_2": "🔎",
    "why_depth_3": "🧠",
}

SESSION_FILE = ".session_cache.pkl"

PERSIST_KEYS = {
    "df",
    "file_name",
    "business_context",
    "pipeline_question",
    "pipeline_result",
    "pipeline_error",
    "messages",
    "chat_display",
    "basic_eda_items",
    "basic_eda_done",
    "eda_uni_ng_items",
    "eda_uni_ng_done",
    "eda_uni_ng_error",
    "eda_uni_g_items",
    "eda_uni_g_done",
    "eda_uni_g_error",
    "eda_multi_ng_items",
    "eda_multi_ng_done",
    "eda_multi_ng_error",
    "eda_multi_g_items",
    "eda_multi_g_done",
    "eda_multi_g_error",
}


def save_session() -> None:
    """Pickle the persistent subset of session state to disk."""
    data = {k: st.session_state[k] for k in PERSIST_KEYS if k in st.session_state}
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass  # never crash the app due to a save failure


def load_session() -> dict:
    """Return the pickled session dict, or empty dict if none exists."""
    try:
        with open(SESSION_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError(
        f"Unsupported file type: {uploaded_file.name}. Upload a CSV or Excel file."
    )


def render_chat_item(item: dict):
    t = item["type"]

    if t == "text":
        st.markdown(item["content"])

    elif t == "code":
        with st.expander("Generated code", expanded=False):
            st.code(item["content"], language="python")

    elif t == "figure":
        img_bytes = base64.b64decode(item["content"])
        st.image(img_bytes, use_container_width=True)

    elif t == "error":
        st.error(item["content"])

    elif t == "stage_header":
        icon = STAGE_ICONS.get(item.get("stage", ""), "▶")
        st.markdown(f"### {icon} {item['content']}")
        st.divider()

    elif t == "plan_card":
        plan = item["content"]
        with st.container(border=True):
            st.markdown(f"**Objective:** {plan.get('objective', '')}")
            cols = plan.get("focus_columns", [])
            if cols:
                st.markdown(f"**Focus columns:** `{'`, `'.join(str(c) for c in cols)}`")
            atypes = plan.get("analysis_types", [])
            if atypes:
                st.markdown(f"**Analysis types:** {' · '.join(atypes)}")
            if plan.get("rationale"):
                with st.expander("Rationale"):
                    st.markdown(plan["rationale"])

    elif t == "hypothesis_card":
        h = item["content"]
        with st.container(border=True):
            priority_badge = f"P{h.get('priority', '?')}"
            st.markdown(
                f"**H{h.get('id', '?')} `{priority_badge}`** — {h.get('statement', '')}"
            )
            c1, c2 = st.columns(2)
            c1.markdown(f"**Test:** `{h.get('test_method', '')}`")
            cols_str = ", ".join(f"`{c}`" for c in h.get("columns_involved", []))
            c2.markdown(f"**Columns:** {cols_str}")
            if h.get("expected_direction"):
                st.caption(f"Expected: {h['expected_direction']}")

    elif t == "critique_card":
        c = item["content"]
        verdict = c.get("verdict", "inconclusive")
        confidence = c.get("confidence", 0.0)
        verdict_color = {
            "supported": "green",
            "refuted": "red",
            "inconclusive": "orange",
        }.get(verdict, "gray")
        with st.container(border=True):
            st.markdown(
                f"**Verdict:** :{verdict_color}[{verdict.upper()}]"
                f"  —  confidence: **{confidence:.0%}**"
            )
            issues = c.get("issues", [])
            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            if c.get("recommendation"):
                st.caption(f"Recommendation: {c['recommendation']}")

    elif t == "insight_box":
        s = item["content"]
        if s.get("headline"):
            st.success(f"**{s['headline']}**")

        score = s.get("insight_confidence_score")
        if score is not None:
            label = "High" if score >= 0.7 else ("Medium" if score >= 0.4 else "Low")
            color = "green" if score >= 0.7 else ("orange" if score >= 0.4 else "red")
            st.markdown(f"**Insight Confidence:** :{color}[{label}] ({score:.0%})")
            st.progress(float(score))

        if s.get("story"):
            with st.container(border=True):
                st.markdown(f"*{s['story']}*")

        insights = s.get("insights", [])
        if insights:
            with st.container(border=True):
                st.markdown("**Key Insights**")
                for i, insight in enumerate(insights, 1):
                    st.markdown(f"{i}. {insight}")
        caveats = s.get("caveats", [])
        if caveats:
            with st.expander("⚠ Caveats"):
                for caveat in caveats:
                    st.markdown(f"- {caveat}")
        next_steps = s.get("next_steps", [])
        if next_steps:
            with st.expander("→ Recommended next steps"):
                for ns in next_steps:
                    st.markdown(f"- {ns}")


def main():
    st.set_page_config(page_title="EDA Agent", page_icon="📊", layout="wide")
    st.title("📊 EDA Agent")

    # --- Restore persisted session on first load ---
    if "_session_loaded" not in st.session_state:
        for key, value in load_session().items():
            st.session_state[key] = value
        for key in list(st.session_state.keys()):
            if key.endswith("_running"):
                st.session_state[key] = False
        st.session_state["_session_loaded"] = True

    # --- Initialize session state defaults (only for keys not already restored) ---
    for key, default in [
        ("messages", []),
        ("df", None),
        ("file_name", None),
        ("chat_display", []),
        ("pipeline_result", None),
        ("business_context", ""),
        ("pipeline_running", False),
        ("pipeline_question", ""),
        ("pipeline_error", None),
        ("pipeline_thread", None),
        ("pipeline_queue", None),
        ("pipeline_live_items", []),
        ("basic_eda_items", []),
        ("basic_eda_running", False),
        ("basic_eda_done", False),
        ("eda_uni_ng_items", []),
        ("eda_uni_ng_done", False),
        ("eda_uni_ng_running", False),
        ("eda_uni_g_items", []),
        ("eda_uni_g_done", False),
        ("eda_uni_g_running", False),
        ("eda_multi_ng_items", []),
        ("eda_multi_ng_done", False),
        ("eda_multi_ng_running", False),
        ("eda_multi_g_items", []),
        ("eda_multi_g_done", False),
        ("eda_multi_g_running", False),
        ("eda_uni_ng_error", None),
        ("eda_uni_g_error", None),
        ("eda_multi_ng_error", None),
        ("eda_multi_g_error", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar ---
    with st.sidebar:
        st.header("🤖 LLM Settings")
        model = st.text_input(
            "Model",
            value=os.getenv("LLM_MODEL", "openrouter/google/gemma-4-31b-it"),
            help=(
                "Examples: openrouter/anthropic/claude-sonnet-4-6 · "
                "openrouter/openai/gpt-4o · "
                "openrouter/google/gemini-2.0-flash-001 · "
                "anthropic/claude-sonnet-4-6 · openai/gpt-4o · "
                "ollama/llama3.1 · groq/llama-3.1-70b-versatile"
            ),
        )
        api_base_input = st.text_input(
            "API Base URL (optional)",
            value=os.getenv("OPEN_ROUTER_BASE_URL", ""),
            help="Leave blank for cloud providers. For local models: http://localhost:11434",
        )
        api_base = api_base_input.strip() or None

    models = load_model_config(model, api_base)

    with st.sidebar:
        st.divider()
        st.header("📁 Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.file_name:
                try:
                    df = load_dataframe(uploaded_file)
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.chat_display = []
                    st.session_state.pipeline_result = None
                    st.session_state.basic_eda_items = []
                    st.session_state.basic_eda_done = False
                    st.session_state.basic_eda_running = True
                    save_session()
                    st.success(f"Loaded **{uploaded_file.name}**")
                except ValueError as e:
                    st.error(str(e))

        if st.session_state.df is not None:
            df = st.session_state.df
            st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10), use_container_width=True, height=200)

            if st.button("🗑️ Clear all", use_container_width=True):
                for key in PERSIST_KEYS:
                    if key in st.session_state:
                        del st.session_state[key]
                try:
                    os.remove(SESSION_FILE)
                except FileNotFoundError:
                    pass
                st.rerun()

        st.divider()
        st.header("🔬 Hypothesis Analysis")

        st.session_state.business_context = st.text_area(
            "Business context (optional)",
            value=st.session_state.business_context,
            placeholder=(
                "e.g. 'E-commerce transactions. "
                "We want to understand why churn increased in Q3.'"
            ),
            height=90,
        )
        pipeline_question = st.text_input(
            "Analysis question",
            placeholder="e.g. 'What drives high customer lifetime value?'",
            key="pipeline_question_input",
        )

        run_disabled = st.session_state.df is None or st.session_state.pipeline_running
        if st.button(
            "🚀 Run Full Analysis",
            use_container_width=True,
            type="primary",
            disabled=run_disabled,
        ):
            if not pipeline_question.strip():
                st.error("Enter an analysis question first.")
            else:
                st.session_state.pipeline_running = True
                st.session_state.pipeline_question = pipeline_question
                st.session_state.pipeline_result = None
                st.session_state.pipeline_error = None
                st.session_state.pipeline_thread = None
                st.session_state.pipeline_live_items = []
                st.rerun()

    # --- Main area ---
    if st.session_state.df is None:
        st.info("Upload a CSV or Excel file in the sidebar to get started.")
        return

    tab_overview, tab_deep_eda = st.tabs(["Overview", "Deep EDA"])

    with tab_overview:
        _render_overview(models)

    with tab_deep_eda:
        _render_deep_eda(models)


def _render_overview(models: ModelConfig) -> None:
    """Renders the Overview tab: Basic EDA, Pipeline, and Chat."""

    # --- Basic EDA runner (auto-triggered on upload) ---
    if st.session_state.basic_eda_running:
        from basic_eda import run_basic_eda

        eda_status = st.empty()
        eda_status.markdown("**Running basic EDA…**")
        eda_container = st.container()
        live_items: list[dict] = []

        def eda_render_cb(item: dict) -> None:
            live_items.append(item)
            with eda_container:
                render_chat_item(item)

        try:
            run_basic_eda(st.session_state.df, eda_render_cb)
            st.session_state.basic_eda_items = live_items
            st.session_state.basic_eda_done = True
            eda_status.markdown(
                "✓ Basic EDA complete. "
                "Ask a question below or run a full hypothesis analysis from the sidebar."
            )
            save_session()
        except Exception as e:
            eda_status.markdown(f"❌ EDA failed: {e}")
        finally:
            st.session_state.basic_eda_running = False

        st.rerun()

    # --- Basic EDA results ---
    if st.session_state.basic_eda_done and st.session_state.basic_eda_items:
        st.markdown("## 📋 Exploratory Data Analysis")
        for item in st.session_state.basic_eda_items:
            render_chat_item(item)
        st.markdown("---")

    # --- Pipeline runner (background thread, polls every 0.3s) ---
    if st.session_state.pipeline_running:
        # Start the thread on first entry
        if st.session_state.pipeline_thread is None:
            q: queue.Queue = queue.Queue()
            st.session_state.pipeline_queue = q
            st.session_state.pipeline_live_items = []

            _models = models
            _df = st.session_state.df
            _question = st.session_state.pipeline_question
            _context = st.session_state.business_context

            def _run_pipeline_thread() -> None:
                try:
                    result = run_pipeline(
                        df=_df,
                        user_question=_question,
                        business_context=_context,
                        models=_models,
                        progress_cb=lambda stage, prog: q.put(
                            ("progress", (stage, prog))
                        ),
                        render_cb=lambda item: q.put(("render", item)),
                    )
                    q.put(("done", result))
                except Exception as exc:
                    q.put(("error", str(exc)))

            t = threading.Thread(target=_run_pipeline_thread, daemon=True)
            st.session_state.pipeline_thread = t
            t.start()

        # Drain the queue
        q = st.session_state.pipeline_queue
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        status_text.markdown("**Running hypothesis analysis…**")
        finished = False

        while not q.empty():
            kind, payload = q.get_nowait()
            if kind == "render":
                st.session_state.pipeline_live_items.append(payload)
            elif kind == "progress":
                stage, prog = payload
                label = STAGE_LABELS.get(stage, stage)
                status_text.markdown(f"**{label}**")
                if stage == "executing" and prog.get("total", 0) > 0:
                    pct = 0.18 + 0.42 * (prog["done"] / prog["total"])
                else:
                    pct = STAGE_WEIGHTS.get(stage, 0.5)
                progress_bar.progress(min(pct, 0.99))
            elif kind == "done":
                st.session_state.pipeline_result = payload
                st.session_state.pipeline_error = None
                st.session_state.pipeline_running = False
                st.session_state.pipeline_thread = None
                st.session_state.pipeline_live_items = []
                save_session()
                finished = True
                break
            elif kind == "error":
                st.session_state.pipeline_error = payload
                st.session_state.pipeline_running = False
                st.session_state.pipeline_thread = None
                st.session_state.pipeline_live_items = []
                finished = True
                break

        if not finished:
            # Render accumulated live items and poll again
            results_container = st.container()
            with results_container:
                for item in st.session_state.pipeline_live_items:
                    render_chat_item(item)
            time.sleep(0.3)
            st.rerun()
        else:
            st.rerun()

    # --- Pipeline error ---
    if st.session_state.pipeline_error:
        st.error(f"❌ Pipeline failed: {st.session_state.pipeline_error}")

    # --- Pipeline results ---
    if st.session_state.pipeline_result is not None:
        result = st.session_state.pipeline_result
        st.markdown("---")
        st.markdown(f"## 🔬 Analysis: *{st.session_state.pipeline_question}*")
        for item in result.render_items:
            render_chat_item(item)
        st.markdown("---")

        # Seed chat history with synthesis context
        if result.synthesis and not st.session_state.messages:
            headline = result.synthesis.get("headline", "")
            if headline:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"I've completed a hypothesis-driven analysis of your dataset. "
                            f"Key finding: {headline} "
                            f"Ask me follow-up questions about any specific aspect of the data."
                        ),
                    }
                )

    # --- Replay chat history ---
    for entry in st.session_state.chat_display:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        else:
            with st.chat_message("assistant"):
                for item in entry["items"]:
                    render_chat_item(item)

    # --- Chat input ---
    if prompt := st.chat_input("Ask anything about your dataset…"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_display.append({"role": "user", "content": prompt})

        is_why = bool(re.search(r"\bwhy\b", prompt, re.IGNORECASE))

        with st.chat_message("assistant"):
            with st.spinner("Investigating…" if is_why else "Thinking…"):
                try:
                    agent = EDAAgent(models=models)
                    if is_why:
                        response_items = agent.run_why_loop(
                            st.session_state.messages,
                            st.session_state.df,
                        )
                    else:
                        response_items = agent.run(
                            st.session_state.messages,
                            st.session_state.df,
                        )
                except Exception as e:
                    response_items = [{"type": "error", "content": f"Agent error: {e}"}]

            for item in response_items:
                render_chat_item(item)

        st.session_state.chat_display.append(
            {"role": "assistant", "items": response_items}
        )
        save_session()


_DEEP_EDA_AGENTS: list[tuple] = [
    (
        "Univariate Non-Graphical",
        "uni_ng",
        run_univariate_nongraphical_agent,
        "Frequency tables, descriptive stats, missing value analysis — per column.",
    ),
    (
        "Univariate Graphical",
        "uni_g",
        run_univariate_graphical_agent,
        "Histograms, box plots, bar charts, stem-and-leaf — per column.",
    ),
    (
        "Multivariate Non-Graphical",
        "multi_ng",
        run_multivariate_nongraphical_agent,
        "Pearson/Spearman correlations, ANOVA, cross-tabs, covariance.",
    ),
    (
        "Multivariate Graphical",
        "multi_g",
        run_multivariate_graphical_agent,
        "Pairplots, heatmaps, grouped bar charts, bubble and run charts.",
    ),
]


def _make_render_cb(
    container: "st.delta_generator.DeltaGenerator",
    live_items: list[dict],
) -> Callable[[dict], None]:
    """Returns a render callback that streams items into container and live_items."""

    def _cb(item: dict) -> None:
        live_items.append(item)
        with container:
            render_chat_item(item)

    return _cb


def _render_deep_eda(models: ModelConfig) -> None:
    """Renders the Deep EDA tab with one expander per specialized agent."""
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
        return

    st.markdown("## Specialized EDA Agents")
    st.caption(
        "Each agent runs a focused analysis independently. "
        "Results persist for the session."
    )

    for label, key, agent_fn, description in _DEEP_EDA_AGENTS:
        done_key = f"eda_{key}_done"
        running_key = f"eda_{key}_running"
        items_key = f"eda_{key}_items"

        is_active = st.session_state[running_key] or st.session_state[done_key]
        with st.expander(f"**{label}**  —  {description}", expanded=is_active):
            if st.session_state[running_key]:
                container = st.container()
                live_items: list[dict] = []
                try:
                    agent_fn(
                        df=st.session_state.df,
                        models=models,
                        render_cb=_make_render_cb(container, live_items),
                    )
                    st.session_state[items_key] = live_items
                    st.session_state[done_key] = True
                    st.session_state[running_key] = False
                    save_session()
                    st.rerun()
                except Exception as e:
                    st.session_state[running_key] = False
                    st.session_state[f"eda_{key}_error"] = str(e)
                    st.rerun()

            elif st.session_state[done_key]:
                for item in st.session_state[items_key]:
                    render_chat_item(item)

            else:
                error_key = f"eda_{key}_error"
                if st.session_state.get(error_key):
                    st.error(f"Agent error: {st.session_state[error_key]}")
                if st.button(
                    f"Run {label}",
                    key=f"btn_deep_{key}",
                    use_container_width=True,
                ):
                    st.session_state[error_key] = None
                    st.session_state[running_key] = True
                    st.rerun()


if __name__ == "__main__":
    main()
