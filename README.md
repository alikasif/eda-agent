# Exploratory Data Analysis(EDA) Agent

An AI-powered exploratory data analysis tool that automatically profiles your dataset and lets you investigate it through natural language.

## Features

- **Auto-EDA on upload** — instantly runs a 3-step profile: descriptive statistics, feature visualizations, and a data quality report
- **Hypothesis-driven analysis** — 5-stage pipeline (Plan → Hypothesize → Execute → Critique → Synthesize) generates and tests falsifiable hypotheses about your data
- **Deep EDA tab** — four specialized agents that each run an independent focused analysis:
  - *Univariate Non-Graphical*: per-column descriptive stats (mean, median, skewness, kurtosis, frequency tables)
  - *Univariate Graphical*: histograms, box plots, bar charts, stem-and-leaf plots per column
  - *Multivariate Non-Graphical*: Pearson/Spearman correlation matrices, ANOVA, cross-tabulations
  - *Multivariate Graphical*: pairplots, heatmaps, grouped bar charts, bubble charts
- **Insight Confidence Score** — rates the reliability of findings based on data size, statistical strength, and verdict consistency
- **Auto Story Mode** — synthesizes results into a 2–3 sentence board-level narrative with concrete numbers
- **Ask Why Loop** — type any "why" question and the agent recursively investigates 3 layers of root causes
- **Multi-LLM support** — works with Anthropic, OpenAI, Ollama, Groq, and any OpenAI-compatible endpoint via LiteLLM

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone <repo-url>
cd eda-agent
cp .env.example .env   # add your API key
uv sync
```

## Configuration

Edit `.env` with your chosen provider:

| Variable | Example | Purpose |
|---|---|---|
| `LLM_MODEL` | `openrouter/google/gemma-4-31b-it` | Default model for all agents |
| `OPEN_ROUTER_API_KEY` | `sk-or-...` | OpenRouter |
| `LLM_API_KEY` | `sk-ant-...` / `sk-...` | Anthropic / OpenAI direct |
| `OPEN_ROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `LLM_API_BASE` | `http://localhost:11434` | Local models (Ollama) |

### Per-agent model overrides

Each agent can use a different model. All fall back to `LLM_MODEL` if not set:

| Variable | Agent |
|---|---|
| `PLANNER_MODEL` | Planning stage |
| `HYPOTHESIS_MODEL` | Hypothesis generation |
| `EXECUTOR_MODEL` | Code execution agent |
| `CRITIC_MODEL` | Result critique |
| `SYNTHESIZER_MODEL` | Insight synthesis |
| `EDA_AGENT_MODEL` | Chat / Ask-Why agent |
| `UNI_NG_MODEL` | Univariate non-graphical agent |
| `UNI_G_MODEL` | Univariate graphical agent |
| `MULTI_NG_MODEL` | Multivariate non-graphical agent |
| `MULTI_G_MODEL` | Multivariate graphical agent |

## Running

```bash
uv run streamlit run src/app.py
```

## How to Use

**Step 1 — Upload your data**
Use the sidebar file uploader to upload a CSV or Excel file. A preview appears immediately.

**Step 2 — Auto-EDA runs**
As soon as the file loads, the app automatically runs:
1. Dataset overview (shape, dtypes, missing values, `describe()`)
2. Feature visualizations (distributions, category counts, Spearman correlation heatmap)
3. Data quality report (missing value severity, constant features, high correlations, class imbalance)

**Step 3 — Run Full Analysis**
Enter an optional business context (e.g. *"E-commerce sales, Q3 revenue dropped"*) and an analysis question (e.g. *"What drives customer churn?"*), then click **Run Full Analysis**. The 5-stage pipeline runs and produces:
- A hypothesis list with test methods and priorities
- Per-hypothesis execution results (code, charts, statistics)
- Critic verdicts (supported / refuted / inconclusive + confidence)
- A synthesis card with an Insight Confidence Score, a narrative story, key insights, caveats, and next steps

**Step 4 — Deep EDA tab**
Switch to the **Deep EDA** tab for four specialized agents. Each runs independently and results persist for the session:
- **Univariate Non-Graphical** — frequency tables, descriptive stats, missing value analysis per column
- **Univariate Graphical** — histograms, box plots, bar charts, stem-and-leaf per column
- **Multivariate Non-Graphical** — Pearson/Spearman correlations, ANOVA, cross-tabs, covariance
- **Multivariate Graphical** — pairplots, heatmaps, grouped bar charts, bubble and run charts

**Step 5 — Ask follow-up questions**
Use the chat at the bottom of the Overview tab to ask anything about your dataset. Include the word **"why"** to trigger the Ask Why Loop, which investigates 3 recursive layers of root causes:

> *"Why is churn higher in this segment?"*
> → Layer 1: identifies the pattern
> → Layer 2: finds what drives that pattern
> → Layer 3: uncovers the root cause

## Supported LLM Providers

| Provider | Model string example |
|---|---|
| OpenRouter | `openrouter/google/gemma-4-31b-it` |
| Anthropic | `anthropic/claude-sonnet-4-6` |
| OpenAI | `openai/gpt-4o` |
| Ollama (local) | `ollama/llama3.1` |
| Groq | `groq/llama-3.1-70b-versatile` |

Set the model string in the sidebar or via `LLM_MODEL` in `.env`. For local/custom endpoints, also set `LLM_API_BASE`.

## Project Structure

```
src/
  app.py          Streamlit UI — upload, rendering, pipeline + chat routing
  agent.py        EDAAgent — conversational analysis and Ask Why Loop
  pipeline.py     5-stage hypothesis pipeline orchestrator
  eda_agents.py   Four specialized Deep EDA agents
  basic_eda.py    Auto-run 3-step EDA triggered on file upload
  prompts.py      LLM system prompts for all agent personas
  tools.py        Python code execution sandbox and dataset info extractor
  config.py       Per-agent model configuration
pyproject.toml    Project metadata and dependencies
.env.example      Environment variable template
```
