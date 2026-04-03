"""
Hypothesis-Driven EDA Pipeline

5-stage pipeline with per-hypothesis streaming:
  1. Planner    — understands schema + question → analysis plan
  2. Hypothesis — generates ALL testable hypotheses (shown before any code runs)
  3. Executor   — runs Python code per hypothesis, streams results immediately
  4. Critic     — validates each result inline (interleaved with stage 3)
  5. Synthesizer — converts all results into business narrative
"""

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm
import pandas as pd
from agents import Agent, Runner, RunContextWrapper, function_tool
from dotenv import load_dotenv

import tools as _tools
from agent import get_model
from prompts import (
    get_critic_messages,
    get_executor_messages,
    get_hypothesis_messages,
    get_planner_messages,
    get_synthesizer_messages,
    EXECUTOR_SYSTEM,
)

load_dotenv(override=True)
litellm.suppress_debug_info = True

MAX_EXECUTOR_TURNS = 6


# ── State ─────────────────────────────────────────────────────────────────────


@dataclass
class EDAState:
    # Inputs
    df: pd.DataFrame
    user_question: str
    business_context: str
    model: str
    api_base: str | None

    # Stage outputs
    schema_json: str = ""
    plan: dict = field(default_factory=dict)
    hypotheses: list[dict] = field(default_factory=list)
    execution_results: list[dict] = field(default_factory=list)
    # each: {hypothesis_index, stdout, error, figures:[base64], figures_count}
    critique_results: list[dict] = field(default_factory=list)
    # each: {hypothesis_index, verdict, confidence, issues, recommendation}
    synthesis: dict = field(default_factory=dict)
    insight_confidence_score: float = 0.0

    # Progress / error tracking
    current_stage: str = "idle"
    errors: list[str] = field(default_factory=list)

    # Accumulated render items for the UI (in pipeline order, for persistence)
    render_items: list[dict] = field(default_factory=list)


# ── Executor Agent tools ──────────────────────────────────────────────────────


@dataclass
class ExecutorContext:
    df: pd.DataFrame
    emit_cb: Callable = field(default=lambda _: None)  # called immediately per artifact
    response_items: list = field(default_factory=list)
    stdout_text: str = ""
    error_text: str = ""
    figures: list[str] = field(default_factory=list)  # base64 strings


@function_tool
def get_dataset_info_tool(ctx: RunContextWrapper[ExecutorContext]) -> str:
    """Returns structural information about the loaded dataset: shape, column names,
    dtypes, null counts, numeric summary statistics, categorical value counts, and a
    5-row sample. Call this first before writing analysis code."""
    result = _tools.get_dataset_info(ctx.context.df)
    return json.dumps(result)


@function_tool
def execute_python_tool(ctx: RunContextWrapper[ExecutorContext], code: str) -> str:
    """Executes Python code for data analysis or visualization.
    df, pd, np, plt, sns, stats, sm, sklearn are pre-injected.
    Use print() for text output. Charts are captured automatically.
    Do NOT reassign df.

    Args:
        code: Python code to execute.
    """
    code_item = {"type": "code", "content": code}
    ctx.context.response_items.append(code_item)
    ctx.context.emit_cb(code_item)  # stream immediately

    result = _tools.execute_python(code, ctx.context.df)

    if result["error"]:
        ctx.context.error_text = result["error"]
        err_item = {"type": "error", "content": result["error"]}
        ctx.context.response_items.append(err_item)
        ctx.context.emit_cb(err_item)  # stream immediately

    if result["stdout"]:
        ctx.context.stdout_text += result["stdout"]

    for fig_b64 in result["figures"]:
        ctx.context.figures.append(fig_b64)
        fig_item = {"type": "figure", "content": fig_b64}
        ctx.context.response_items.append(fig_item)
        ctx.context.emit_cb(fig_item)  # stream immediately

    return json.dumps(
        {
            "stdout": result["stdout"],
            "error": result["error"],
            "figures_generated": len(result["figures"]),
        }
    )


# ── LLM helper ────────────────────────────────────────────────────────────────

_KNOWN_PROVIDERS = {
    "openai/",
    "anthropic/",
    "ollama/",
    "groq/",
    "vertex_ai/",
    "huggingface/",
    "cohere/",
    "mistral/",
    "together_ai/",
    "azure/",
}


def _resolve_model(model: str, api_base: str | None) -> str:
    """Prefix with 'openai/' when using a custom base URL and no known provider prefix."""
    if api_base and not any(model.startswith(p) for p in _KNOWN_PROVIDERS):
        return f"openai/{model}"
    return model


def _build_llm_kwargs(model: str, messages: list[dict], api_base: str | None) -> dict:
    kwargs: dict = {
        "model": _resolve_model(model, api_base),
        "messages": messages,
        "max_tokens": 4096,
    }
    if api_base:
        kwargs["base_url"] = api_base
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LOCAL_MODEL_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a text string (handles markdown code fences)."""
    import re

    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Malformed JSON in response")


def call_llm_json(
    messages: list[dict],
    model: str,
    api_base: str | None,
    expected_keys: list[str],
    fallback: dict,
) -> dict:
    """
    Calls litellm and parses a JSON response. Never raises — returns fallback on failure.

    Strategy:
    1. Try with response_format=json_object (native JSON mode).
    2. If that fails (unsupported by model), retry without it and extract JSON from raw text.
    """
    last_error: str = ""

    # Attempt 1: JSON mode
    try:
        kwargs = _build_llm_kwargs(model, messages, api_base)
        kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        if all(k in parsed for k in expected_keys):
            return parsed
        last_error = f"Missing expected keys {expected_keys} in: {raw[:300]}"
    except Exception as e:
        last_error = str(e)

    # Attempt 2: Plain completion, extract JSON from text
    try:
        kwargs = _build_llm_kwargs(model, messages, api_base)
        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content or ""
        parsed = _extract_json(raw)
        if all(k in parsed for k in expected_keys):
            return parsed
        last_error = f"Missing expected keys {expected_keys} in: {raw[:300]}"
    except Exception as e:
        last_error = f"{last_error} | fallback attempt: {e}"

    return {**fallback, "_error": last_error}


# ── Helpers: emit an item to both render_items and the live render callback ───


def _emit(state: EDAState, emit: Callable, item: dict) -> None:
    state.render_items.append(item)
    emit(item)


# ── Stage 1: Planner ──────────────────────────────────────────────────────────


def stage_plan(state: EDAState, cb: Callable, emit: Callable) -> None:
    cb("planning", {})

    schema = _tools.get_dataset_info(state.df)
    state.schema_json = json.dumps(schema, indent=2)

    messages = get_planner_messages(
        schema_json=state.schema_json,
        business_context=state.business_context,
        user_question=state.user_question,
    )
    plan = call_llm_json(
        messages=messages,
        model=state.model,
        api_base=state.api_base,
        expected_keys=["objective", "focus_columns", "analysis_types"],
        fallback={
            "objective": state.user_question,
            "focus_columns": [c["name"] for c in schema["columns"][:5]],
            "analysis_types": ["distribution", "correlation"],
            "rationale": "Fallback plan — planner LLM call failed.",
        },
    )
    if "_error" in plan:
        state.errors.append(f"Planner: {plan['_error']}")
        _emit(
            state,
            emit,
            {"type": "error", "content": f"Planner LLM error: {plan['_error']}"},
        )
    state.plan = {k: v for k, v in plan.items() if k != "_error"}

    _emit(
        state,
        emit,
        {"type": "stage_header", "stage": "planning", "content": "Analysis Plan"},
    )
    _emit(state, emit, {"type": "plan_card", "content": state.plan})


# ── Stage 2: Hypothesis Agent ─────────────────────────────────────────────────


def stage_hypothesize(state: EDAState, cb: Callable, emit: Callable) -> None:
    cb("hypothesizing", {})

    messages = get_hypothesis_messages(
        plan_json=json.dumps(state.plan, indent=2),
        schema_json=state.schema_json,
        business_context=state.business_context,
    )
    result = call_llm_json(
        messages=messages,
        model=state.model,
        api_base=state.api_base,
        expected_keys=["hypotheses"],
        fallback={"hypotheses": []},
    )
    if "_error" in result:
        state.errors.append(f"Hypothesis agent: {result['_error']}")
        _emit(
            state,
            emit,
            {"type": "error", "content": f"Hypothesis agent error: {result['_error']}"},
        )

    hypotheses = result.get("hypotheses", [])
    hypotheses.sort(key=lambda h: h.get("priority", 99))
    for i, h in enumerate(hypotheses):
        h["id"] = i + 1
    state.hypotheses = hypotheses

    _emit(
        state,
        emit,
        {
            "type": "stage_header",
            "stage": "hypothesizing",
            "content": f"Hypotheses ({len(hypotheses)} generated)",
        },
    )
    # Emit ALL hypothesis cards immediately — user sees them before any code runs
    for h in hypotheses:
        _emit(state, emit, {"type": "hypothesis_card", "content": h})


# ── Stage 3 (per-hypothesis): Executor ───────────────────────────────────────


def _execute_hypothesis(
    state: EDAState,
    executor_agent: Agent,
    idx: int,
    cb: Callable,
    emit: Callable,
) -> None:
    """Execute one hypothesis and stream its artifacts immediately."""
    hypothesis = state.hypotheses[idx]
    total = len(state.hypotheses)
    cb("executing", {"done": idx, "total": total})

    # Header appears before any code runs
    _emit(
        state,
        emit,
        {
            "type": "stage_header",
            "stage": "executing",
            "content": f"H{hypothesis['id']}: {hypothesis['statement']}",
        },
    )

    # Pass emit into context — execute_python_tool streams each artifact immediately
    ctx = ExecutorContext(df=state.df, emit_cb=emit)
    messages = get_executor_messages(hypothesis, state.schema_json)

    try:
        result = Runner.run_sync(
            executor_agent,
            messages,
            context=ctx,
            max_turns=MAX_EXECUTOR_TURNS,
        )
        final_text = result.final_output or ""
    except Exception as e:
        final_text = f"Executor error: {e}"
        ctx.error_text = str(e)

    # Persist artifacts to state.render_items — do NOT emit again (already streamed live)
    for artifact in ctx.response_items:
        state.render_items.append(artifact)
    if final_text:
        _emit(state, emit, {"type": "text", "content": final_text})

    state.execution_results.append(
        {
            "hypothesis_index": idx,
            "stdout": ctx.stdout_text,
            "error": ctx.error_text or None,
            "figures": ctx.figures,
            "figures_count": len(ctx.figures),
        }
    )
    cb("executing", {"done": idx + 1, "total": total})


# ── Stage 4 (per-hypothesis): Critic ─────────────────────────────────────────


def _critique_hypothesis(
    state: EDAState,
    idx: int,
    cb: Callable,
    emit: Callable,
) -> None:
    """Critique one hypothesis result and emit the verdict inline."""
    cb("critiquing", {})
    hypothesis = state.hypotheses[idx]
    exec_result = state.execution_results[idx]

    messages = get_critic_messages(
        hypothesis,
        {
            "stdout": exec_result.get("stdout", ""),
            "error": exec_result.get("error"),
            "figures_count": exec_result.get("figures_count", 0),
        },
    )
    critique = call_llm_json(
        messages=messages,
        model=state.model,
        api_base=state.api_base,
        expected_keys=["verdict", "confidence"],
        fallback={
            "verdict": "inconclusive",
            "confidence": 0.0,
            "issues": ["Critic LLM call failed — unable to validate result."],
            "recommendation": "Review execution output manually.",
        },
    )
    critique["hypothesis_index"] = idx
    state.critique_results.append(critique)
    _emit(state, emit, {"type": "critique_card", "content": critique})


# ── Stage 5: Synthesizer ──────────────────────────────────────────────────────


def _compute_insight_confidence(
    df: pd.DataFrame, critique_results: list[dict]
) -> float:
    """Compute a 0–1 insight confidence score.

    Combines three factors:
    - data_size   (25%): log-normalized row count
    - stat_strength (50%): mean critic confidence across hypotheses
    - consistency  (25%): fraction of hypotheses with a decisive verdict
    """
    n_rows = len(df)
    # log10(10)=1, log10(100k)=5 → normalise to [0,1]
    data_size_score = min(1.0, math.log10(max(n_rows, 10)) / 5.0)

    if not critique_results:
        return round(data_size_score * 0.5, 2)

    stat_strength = sum(c.get("confidence", 0.0) for c in critique_results) / len(
        critique_results
    )
    decisive = sum(1 for c in critique_results if c.get("verdict") != "inconclusive")
    consistency = decisive / len(critique_results)

    score = 0.25 * data_size_score + 0.50 * stat_strength + 0.25 * consistency
    return round(min(score, 1.0), 2)


def stage_synthesize(state: EDAState, cb: Callable, emit: Callable) -> None:
    cb("synthesizing", {})

    messages = get_synthesizer_messages(
        hypotheses=state.hypotheses,
        execution_results=state.execution_results,
        critique_results=state.critique_results,
        business_context=state.business_context,
    )
    synthesis = call_llm_json(
        messages=messages,
        model=state.model,
        api_base=state.api_base,
        expected_keys=["headline", "insights", "story"],
        fallback={
            "headline": "Analysis complete — see individual hypothesis results above.",
            "story": "",
            "insights": [],
            "caveats": [],
            "next_steps": [],
        },
    )
    state.insight_confidence_score = _compute_insight_confidence(
        state.df, state.critique_results
    )
    synthesis["insight_confidence_score"] = state.insight_confidence_score
    state.synthesis = synthesis

    _emit(
        state,
        emit,
        {
            "type": "stage_header",
            "stage": "synthesizing",
            "content": "Business Insights",
        },
    )
    _emit(state, emit, {"type": "insight_box", "content": synthesis})


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_pipeline(
    df: pd.DataFrame,
    user_question: str,
    business_context: str,
    model: str,
    api_base: str | None,
    progress_cb: Callable[[str, dict], None] | None = None,
    render_cb: Callable[[dict], None] | None = None,
) -> EDAState:
    """
    Runs the full pipeline, streaming render items as they are produced.

    Args:
        render_cb: Called immediately for each render item as it is produced.
                   Also stored in state.render_items for persistence.
    """
    state = EDAState(
        df=df,
        user_question=user_question,
        business_context=business_context,
        model=model,
        api_base=api_base,
    )
    cb = progress_cb or (lambda stage, prog: None)
    emit = render_cb or (lambda item: None)

    # Stage 1: Plan
    state.current_stage = "planning"
    try:
        stage_plan(state, cb, emit)
    except Exception as e:
        _emit(
            state,
            emit,
            {"type": "error", "content": f"Pipeline error in planning: {e}"},
        )
        state.errors.append(f"planning: {e}")
        state.current_stage = "error"
        return state

    # Stage 2: Hypothesize — all cards stream before any code runs
    state.current_stage = "hypothesizing"
    try:
        stage_hypothesize(state, cb, emit)
    except Exception as e:
        _emit(
            state,
            emit,
            {"type": "error", "content": f"Pipeline error in hypothesizing: {e}"},
        )
        state.errors.append(f"hypothesizing: {e}")
        state.current_stage = "error"
        return state

    if not state.hypotheses:
        state.current_stage = "done"
        return state

    # Build executor agent once, reuse for all hypotheses
    executor_agent = Agent(
        name="EDA Executor",
        model=get_model(state.model, api_base=state.api_base),
        instructions=EXECUTOR_SYSTEM,
        tools=[get_dataset_info_tool, execute_python_tool],
    )

    # Stages 3+4: Execute then critique each hypothesis before moving to the next
    for i in range(len(state.hypotheses)):
        state.current_stage = "executing"
        try:
            _execute_hypothesis(state, executor_agent, i, cb, emit)
        except Exception as e:
            _emit(
                state,
                emit,
                {"type": "error", "content": f"Executor error for H{i + 1}: {e}"},
            )
            state.errors.append(f"executing H{i + 1}: {e}")
            # Placeholder so critique can still run at the same index
            state.execution_results.append(
                {
                    "hypothesis_index": i,
                    "stdout": "",
                    "error": str(e),
                    "figures": [],
                    "figures_count": 0,
                }
            )

        state.current_stage = "critiquing"
        try:
            _critique_hypothesis(state, i, cb, emit)
        except Exception as e:
            _emit(
                state,
                emit,
                {"type": "error", "content": f"Critic error for H{i + 1}: {e}"},
            )
            state.errors.append(f"critiquing H{i + 1}: {e}")

    # Stage 5: Synthesize
    state.current_stage = "synthesizing"
    try:
        stage_synthesize(state, cb, emit)
    except Exception as e:
        _emit(
            state,
            emit,
            {"type": "error", "content": f"Pipeline error in synthesizing: {e}"},
        )
        state.errors.append(f"synthesizing: {e}")

    state.current_stage = "done"
    return state
