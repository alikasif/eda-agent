# EDA Agent

An AI-powered exploratory data analysis tool that automatically profiles your dataset and lets you investigate it through natural language.

## Features

- **Auto-EDA on upload** — instantly runs a 3-step profile: descriptive statistics, feature visualizations, and a data quality report
- **Hypothesis-driven analysis** — 5-stage pipeline (Plan → Hypothesize → Execute → Critique → Synthesize) generates and tests falsifiable hypotheses about your data
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
| `LLM_MODEL` | `claude-sonnet-4-6` | Model to use |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Anthropic Claude |
| `OPENAI_API_KEY` | `sk-...` | OpenAI models |
| `LLM_API_BASE` | `http://localhost:11434` | Local models (Ollama) |
| `GROQ_API_KEY` | `gsk_...` | Groq models |

## Running

```bash
uv run streamlit run app.py
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

**Step 4 — Ask follow-up questions**
Use the chat at the bottom to ask anything about your dataset. Include the word **"why"** to trigger the Ask Why Loop, which investigates 3 recursive layers of root causes:

> *"Why is churn higher in this segment?"*
> → Layer 1: identifies the pattern
> → Layer 2: finds what drives that pattern
> → Layer 3: uncovers the root cause

## Supported LLM Providers

| Provider | Model string |
|---|---|
| Anthropic (default) | `claude-sonnet-4-6` |
| OpenAI | `openai/gpt-4o` |
| Ollama (local) | `ollama/llama3.1` |
| Groq | `groq/llama-3.1-70b-versatile` |

Set the model string in the sidebar or via `LLM_MODEL` in `.env`. For local/custom endpoints, also set `LLM_API_BASE`.

## Project Structure

```
app.py          Streamlit UI — upload, rendering, pipeline + chat routing
agent.py        EDAAgent — conversational analysis and Ask Why Loop
pipeline.py     5-stage hypothesis pipeline orchestrator
basic_eda.py    Auto-run 3-step EDA triggered on file upload
prompts.py      LLM system prompts for all 5 agent personas
tools.py        Python code execution sandbox and dataset info extractor
pyproject.toml  Project metadata and dependencies
.env.example    Environment variable template
```
