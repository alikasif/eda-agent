"""
Model configuration for EDA agents.

Each agent can be configured with its own LLM model via environment variables.
All agent-specific vars fall back to LLM_MODEL if not set.
"""

import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Per-agent model configuration."""

    planner: str
    hypothesis: str
    executor: str
    critic: str
    synthesizer: str
    eda_agent: str
    uni_ng: str
    uni_g: str
    multi_ng: str
    multi_g: str
    api_base: str | None


def load_model_config(default_model: str, api_base: str | None) -> ModelConfig:
    """Build a ModelConfig from environment variables, falling back to default_model.

    Args:
        default_model: Fallback model used for any agent without a specific env var.
        api_base: Shared API base URL (e.g. for OpenRouter or local endpoints).

    Returns:
        ModelConfig with per-agent model strings resolved from the environment.
    """

    def get(key: str) -> str:
        return os.getenv(key, default_model)

    return ModelConfig(
        planner=get("PLANNER_MODEL"),
        hypothesis=get("HYPOTHESIS_MODEL"),
        executor=get("EXECUTOR_MODEL"),
        critic=get("CRITIC_MODEL"),
        synthesizer=get("SYNTHESIZER_MODEL"),
        eda_agent=get("EDA_AGENT_MODEL"),
        uni_ng=get("UNI_NG_MODEL"),
        uni_g=get("UNI_G_MODEL"),
        multi_ng=get("MULTI_NG_MODEL"),
        multi_g=get("MULTI_G_MODEL"),
        api_base=api_base,
    )
