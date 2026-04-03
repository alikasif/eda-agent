"""
EDA Agent

Exploratory Data Analysis agent using OpenAI Agent SDK with LiteLLM.
"""

import json
from dataclasses import dataclass, field

import pandas as pd
from agents import Agent, Runner, RunContextWrapper, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

import tools as _tools

load_dotenv(override=True)

MAX_TURNS = 10
MAX_WHY_DEPTH = 3

WHY_FOLLOW_UP_TEMPLATE = (
    "Based on this finding:\n\n{finding}\n\n"
    "Now investigate one level deeper: WHY does this happen? "
    "Look for root causes in the data. Investigate the next layer of causation."
)

SYSTEM_PROMPT = """You are an expert data analyst assistant. The user has uploaded a dataset \
which is available as a pandas DataFrame called `df`.

Your workflow:
1. When first asked about a dataset, call get_dataset_info to understand its structure.
2. For any analysis, write and execute Python code using execute_python.
3. Always use print() to surface computed values you want the user to see.
4. For visualizations, create matplotlib/seaborn figures — they are captured automatically.
5. After getting tool results, summarize your findings in plain language.
6. If code fails, read the error, fix it, and retry once before reporting the issue.

Guidelines:
- Prefer seaborn for statistical plots, matplotlib for custom plots.
- Always label axes and add titles to charts.
- For correlation analysis, use df.corr(numeric_only=True).
- When showing distributions, consider data type: hist for numeric, countplot for categorical.
- Be concise in explanations — let the data speak.
"""


@dataclass
class EDAContext:
    df: pd.DataFrame
    response_items: list = field(default_factory=list)


@function_tool
def get_dataset_info(ctx: RunContextWrapper[EDAContext]) -> str:
    """Returns structural information about the loaded dataset: shape, column names,
    dtypes, null counts, numeric summary statistics, categorical value counts, and a
    5-row sample. Call this first before writing any analysis code."""
    result = _tools.get_dataset_info(ctx.context.df)
    return json.dumps(result)


@function_tool
def execute_python(ctx: RunContextWrapper[EDAContext], code: str) -> str:
    """Executes Python code for data analysis or visualization.
    The variable `df` (a pandas DataFrame) is pre-loaded in scope.
    `pd`, `plt` (matplotlib.pyplot), and `sns` (seaborn) are also available.
    Use print() to surface computed values.
    Create charts with plt.figure() and standard plot calls — they are captured automatically.
    Do NOT call pd.read_csv() or reassign df; the dataset is already loaded.

    Args:
        code: Valid Python code to execute. Use print() for text output.
    """
    ctx.context.response_items.append({"type": "code", "content": code})

    result = _tools.execute_python(code, ctx.context.df)

    if result["error"]:
        ctx.context.response_items.append({"type": "error", "content": result["error"]})

    for fig_b64 in result["figures"]:
        ctx.context.response_items.append({"type": "figure", "content": fig_b64})

    # Return only text summary — NOT the raw base64 blobs
    return json.dumps(
        {
            "stdout": result["stdout"],
            "error": result["error"],
            "figures_generated": len(result["figures"]),
        }
    )


def _extract_key_finding(items: list[dict]) -> str:
    """Return the last meaningful text item from an agent response (capped at 500 chars)."""
    for item in reversed(items):
        if item.get("type") == "text" and len(item.get("content", "")) > 50:
            return item["content"][:500]
    return ""


def get_model(model: str, api_base: str | None = None) -> LitellmModel:
    """Get LitellmModel configured from the given model string and optional base URL."""
    import os

    # When using a custom base_url (local/Docker endpoint), the endpoint is
    # OpenAI-compatible. Prefix with "openai/" if the model doesn't already start
    # with a known LiteLLM provider prefix.
    KNOWN_PROVIDERS = {
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
    if api_base and not any(model.startswith(p) for p in KNOWN_PROVIDERS):
        model = f"openai/{model}"
    kwargs: dict = {"model": model}
    if api_base:
        kwargs["base_url"] = api_base
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LOCAL_MODEL_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return LitellmModel(**kwargs)


class EDAAgent:
    """EDA Agent."""

    def __init__(self, model: str, api_base: str | None = None):
        self.agent = Agent(
            name="EDA Agent",
            model=get_model(model, api_base=api_base),
            instructions=SYSTEM_PROMPT,
            tools=[get_dataset_info, execute_python],
        )

    def run(self, messages: list[dict], df: pd.DataFrame) -> list[dict]:
        """
        Run one agent turn (potentially multiple tool calls).

        Args:
            messages: Conversation history in OpenAI format. Updated in place.
            df: The loaded pandas DataFrame.

        Returns:
            A list of response item dicts for Streamlit to render:
                {"type": "text",   "content": str}
                {"type": "figure", "content": str}   # base64 PNG
                {"type": "error",  "content": str}
                {"type": "code",   "content": str}
        """
        context = EDAContext(df=df)

        result = Runner.run_sync(
            self.agent,
            messages,
            context=context,
            max_turns=MAX_TURNS,
        )

        # Update messages in place with the full updated history
        messages[:] = result.to_input_list()

        # Combine tool-generated items (code/figures/errors) with final text output
        response_items = context.response_items.copy()
        if result.final_output:
            response_items.append({"type": "text", "content": result.final_output})

        return response_items

    def run_why_loop(self, messages: list[dict], df: pd.DataFrame) -> list[dict]:
        """Recursively investigate a 'why' question up to MAX_WHY_DEPTH layers.

        Each layer runs a full agent turn then automatically formulates the next
        level of investigation based on the findings.

        Args:
            messages: Conversation history, updated in place across all layers.
            df: The loaded pandas DataFrame.

        Returns:
            All render items from all investigation layers, with depth headers.
        """
        all_items: list[dict] = []

        for depth in range(1, MAX_WHY_DEPTH + 1):
            all_items.append(
                {
                    "type": "stage_header",
                    "stage": f"why_depth_{depth}",
                    "content": f"Investigation Layer {depth}/{MAX_WHY_DEPTH}",
                }
            )

            items = self.run(messages, df)
            all_items.extend(items)

            if depth == MAX_WHY_DEPTH:
                break

            finding = _extract_key_finding(items)
            if not finding:
                break

            follow_up = WHY_FOLLOW_UP_TEMPLATE.format(finding=finding)
            messages.append({"role": "user", "content": follow_up})

        return all_items
