import os
import re
import base64

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from agent import EDAAgent
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
    "executing": 0.60,  # updated dynamically per hypothesis
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
                f"**Verdict:** :{verdict_color}[{verdict.upper()}]  —  confidence: **{confidence:.0%}**"
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

    # --- Initialize session state ---
    for key, default in [
        ("messages", []),
        ("df", None),
        ("file_name", None),
        ("chat_display", []),
        ("pipeline_result", None),
        ("business_context", ""),
        ("pipeline_running", False),
        ("pipeline_question", ""),
        ("basic_eda_items", []),
        ("basic_eda_running", False),
        ("basic_eda_done", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar ---
    with st.sidebar:
        st.header("🤖 LLM Settings")
        model = st.text_input(
            "Model",
            value=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            help="Examples: claude-sonnet-4-6 · openai/gpt-4o · ollama/llama3.1 · groq/llama-3.1-70b-versatile",
        )
        api_base_input = st.text_input(
            "API Base URL (optional)",
            value=os.getenv("LLM_API_BASE", ""),
            help="Leave blank for cloud providers. For local models: http://localhost:11434",
        )
        api_base = api_base_input.strip() or None

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
                    st.success(f"Loaded **{uploaded_file.name}**")
                except ValueError as e:
                    st.error(str(e))

        if st.session_state.df is not None:
            df = st.session_state.df
            st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10), use_container_width=True, height=200)

            if st.button("🗑️ Clear all", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_display = []
                st.session_state.pipeline_result = None
                st.rerun()

        st.divider()
        st.header("🔬 Hypothesis Analysis")

        st.session_state.business_context = st.text_area(
            "Business context (optional)",
            value=st.session_state.business_context,
            placeholder="e.g. 'E-commerce transactions. We want to understand why churn increased in Q3.'",
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
                st.rerun()

    # --- Main area ---
    if st.session_state.df is None:
        st.info("Upload a CSV or Excel file in the sidebar to get started.")
        return

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
        except Exception as e:
            eda_status.markdown(f"❌ EDA failed: {e}")
        finally:
            st.session_state.basic_eda_running = False

        st.rerun()

    # --- Basic EDA results (persisted after runner finishes) ---
    if st.session_state.basic_eda_done and st.session_state.basic_eda_items:
        st.markdown("## 📋 Exploratory Data Analysis")
        for item in st.session_state.basic_eda_items:
            render_chat_item(item)
        st.markdown("---")

    # --- Pipeline runner (streams results in real time) ---
    if st.session_state.pipeline_running:
        question = st.session_state.pipeline_question

        status_text = st.empty()
        progress_bar = st.progress(0.0)
        results_container = st.container()  # items stream into here as they're produced

        def progress_cb(stage: str, prog: dict):
            label = STAGE_LABELS.get(stage, stage)
            status_text.markdown(f"**{label}**")
            if stage == "executing" and prog.get("total", 0) > 0:
                pct = 0.18 + 0.42 * (prog["done"] / prog["total"])
            else:
                pct = STAGE_WEIGHTS.get(stage, 0.5)
            progress_bar.progress(min(pct, 0.99))

        def render_cb(item: dict):
            with results_container:
                render_chat_item(item)

        try:
            result = run_pipeline(
                df=st.session_state.df,
                user_question=question,
                business_context=st.session_state.business_context,
                model=model,
                api_base=api_base,
                progress_cb=progress_cb,
                render_cb=render_cb,
            )
            st.session_state.pipeline_result = result
            progress_bar.progress(1.0)
            if result.errors:
                status_text.markdown("⚠ Analysis completed with warnings.")
            else:
                status_text.markdown("✓ Analysis complete.")
        except Exception as e:
            status_text.markdown(f"❌ Pipeline failed: {e}")
            st.session_state.pipeline_result = None
        finally:
            st.session_state.pipeline_running = False

        st.rerun()

    # --- Pipeline results ---
    if st.session_state.pipeline_result is not None:
        result = st.session_state.pipeline_result
        st.markdown("---")
        st.markdown(f"## 🔬 Analysis: *{st.session_state.pipeline_question}*")
        for item in result.render_items:
            render_chat_item(item)
        st.markdown("---")

        # Seed chat history with synthesis context so EDAAgent knows what was found
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
                    agent = EDAAgent(model=model, api_base=api_base)
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


if __name__ == "__main__":
    main()
