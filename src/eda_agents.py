"""
Four specialized EDA agents, one per industry-standard EDA type:

- Univariate Non-Graphical: per-column statistics, no charts
- Univariate Graphical: per-column visualizations
- Multivariate Non-Graphical: cross-column statistics, no charts
- Multivariate Graphical: cross-column visualizations
"""

from collections.abc import Callable

import pandas as pd
from agents import Agent, ModelSettings, Runner

from agent import get_model
from config import ModelConfig
from pipeline import ExecutorContext, execute_python, get_dataset_info
from prompts import (
    MULTI_GRAPHICAL_SYSTEM,
    MULTI_NONGRAPHICAL_SYSTEM,
    UNI_GRAPHICAL_SYSTEM,
    UNI_NONGRAPHICAL_SYSTEM,
)

_USER_MESSAGE = {"role": "user", "content": "Perform your analysis on this dataset."}
MAX_TURNS = 5

def _run_eda_agent(
    name: str,
    instructions: str,
    df: pd.DataFrame,
    model: str,
    api_base: str | None,
    render_cb: Callable[[dict], None],
    max_turns: int = MAX_TURNS,
) -> list[dict]:
    """Shared runner: builds an Agent SDK agent, wires ExecutorContext, runs it,
    and streams all artifacts (code, figures, errors, final text) via render_cb.

    Args:
        name: Agent name for Agent SDK identification.
        instructions: System prompt for the agent.
        df: Dataset to analyse.
        model: LLM model identifier.
        api_base: Optional API base URL for local/custom models.
        render_cb: Called immediately for each emitted artifact dict.
        max_turns: Maximum Agent SDK turns before stopping.

    Returns:
        List of all artifact dicts emitted during the run.
    """
    ctx = ExecutorContext(df=df, emit_cb=render_cb)
    agent = Agent(
        name=name,
        instructions=instructions,
        tools=[get_dataset_info, execute_python],
        model=get_model(model, api_base),
        model_settings=ModelSettings(extra_args={"timeout": 120}),
    )
    result = Runner.run_sync(
        agent,
        [_USER_MESSAGE],
        context=ctx,
        max_turns=max_turns,
    )
    if result.final_output:
        text_item: dict = {"type": "text", "content": result.final_output}
        render_cb(text_item)
        ctx.response_items.append(text_item)
    return ctx.response_items


def run_univariate_nongraphical_agent(
    df: pd.DataFrame,
    models: ModelConfig,
    render_cb: Callable[[dict], None],
) -> list[dict]:
    """Runs the univariate non-graphical EDA agent.

    Produces per-column descriptive statistics (mean, median, std, quartiles,
    skewness, kurtosis, frequency tables, missing %) with no charts.

    Args:
        df: Dataset to analyse.
        models: Per-agent model configuration.
        render_cb: Called immediately for each emitted artifact dict.

    Returns:
        List of all artifact dicts emitted during the run.
    """
    return _run_eda_agent(
        name="univariate_nongraphical",
        instructions=UNI_NONGRAPHICAL_SYSTEM,
        df=df,
        model=models.uni_ng,
        api_base=models.api_base,
        render_cb=render_cb,
        max_turns=MAX_TURNS,
    )


def run_univariate_graphical_agent(
    df: pd.DataFrame,
    models: ModelConfig,
    render_cb: Callable[[dict], None],
) -> list[dict]:
    """Runs the univariate graphical EDA agent.

    Produces per-column visualizations: histograms, box plots, bar charts,
    and text stem-and-leaf plots.

    Args:
        df: Dataset to analyse.
        models: Per-agent model configuration.
        render_cb: Called immediately for each emitted artifact dict.

    Returns:
        List of all artifact dicts emitted during the run.
    """
    return _run_eda_agent(
        name="univariate_graphical",
        instructions=UNI_GRAPHICAL_SYSTEM,
        df=df,
        model=models.uni_g,
        api_base=models.api_base,
        render_cb=render_cb,
        max_turns=MAX_TURNS,
    )


def run_multivariate_nongraphical_agent(
    df: pd.DataFrame,
    models: ModelConfig,
    render_cb: Callable[[dict], None],
) -> list[dict]:
    """Runs the multivariate non-graphical EDA agent.

    Produces cross-column statistics: Pearson/Spearman correlation matrices,
    ANOVA tests, cross-tabulations, and covariance matrix with no charts.

    Args:
        df: Dataset to analyse.
        models: Per-agent model configuration.
        render_cb: Called immediately for each emitted artifact dict.

    Returns:
        List of all artifact dicts emitted during the run.
    """
    return _run_eda_agent(
        name="multivariate_nongraphical",
        instructions=MULTI_NONGRAPHICAL_SYSTEM,
        df=df,
        model=models.multi_ng,
        api_base=models.api_base,
        render_cb=render_cb,
        max_turns=MAX_TURNS,
    )


def run_multivariate_graphical_agent(
    df: pd.DataFrame,
    models: ModelConfig,
    render_cb: Callable[[dict], None],
) -> list[dict]:
    """Runs the multivariate graphical EDA agent.

    Produces cross-column visualizations: pairplots, correlation heatmap,
    grouped bar charts, bubble chart, and run charts.

    Args:
        df: Dataset to analyse.
        models: Per-agent model configuration.
        render_cb: Called immediately for each emitted artifact dict.

    Returns:
        List of all artifact dicts emitted during the run.
    """
    return _run_eda_agent(
        name="multivariate_graphical",
        instructions=MULTI_GRAPHICAL_SYSTEM,
        df=df,
        model=models.multi_g,
        api_base=models.api_base,
        render_cb=render_cb,
        max_turns=MAX_TURNS,
    )
