"""
Agent prompt strings for the hypothesis-driven EDA pipeline.

Each section has a SYSTEM constant, a USER template, and a getter function
that returns a list[dict] in OpenAI messages format ready for litellm.completion.
"""

import json

# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a senior data scientist planning an exploratory data analysis.
You receive a dataset schema and a business question. You cannot run code.
Your job is to decide WHAT to analyze — which columns, which analysis types, and why.

Think like a data scientist in the first 30 minutes: start with the most promising signals.
Focus on what will actually answer the business question, not exhaustive coverage.

Respond with a JSON object only. No explanation outside the JSON."""

PLANNER_USER = """Dataset schema:
{schema_json}

Business context: {business_context}

User question: {user_question}

Respond with JSON only, using this exact schema:
{{
  "objective": "one sentence describing what the analysis will answer",
  "focus_columns": ["col1", "col2"],
  "analysis_types": ["one or more of: distribution, correlation, segmentation, time_series, statistical_test, feature_importance, cohort"],
  "rationale": "2-3 sentences explaining why these columns and analysis types"
}}"""


def get_planner_messages(
    schema_json: str, business_context: str, user_question: str
) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": PLANNER_USER.format(
                schema_json=schema_json,
                business_context=business_context or "Not provided.",
                user_question=user_question,
            ),
        },
    ]


# ── Hypothesis Agent ──────────────────────────────────────────────────────────

HYPOTHESIS_SYSTEM = """You are a hypothesis-driven data scientist.
You generate specific, falsifiable hypotheses that can each be tested with a single Python code block.

Rules:
1. Each hypothesis must name the exact columns involved.
2. Each hypothesis must specify a concrete test method.
3. Each hypothesis must state what result would support or refute it.
4. Priority 1 = highest business impact. Generate 3-5 hypotheses.
5. Hypotheses must be falsifiable — avoid vague statements like "X affects Y".

Respond with a JSON object only. No explanation outside the JSON."""

HYPOTHESIS_USER = """Analysis plan:
{plan_json}

Dataset schema:
{schema_json}

Business context: {business_context}

Generate 3-5 testable hypotheses. Respond with JSON only:
{{
  "hypotheses": [
    {{
      "id": 1,
      "priority": 1,
      "statement": "falsifiable hypothesis in plain English",
      "columns_involved": ["col_a", "col_b"],
      "test_method": "one of: t-test, chi-squared, correlation, clustering, feature_importance, distribution, cohort, anomaly_detection",
      "expected_direction": "what result (e.g. p<0.05, positive correlation, clear cluster separation) would support this hypothesis",
      "code_hint": "one sentence of guidance for writing the test code (optional)"
    }}
  ]
}}"""


def get_hypothesis_messages(
    plan_json: str, schema_json: str, business_context: str
) -> list[dict]:
    return [
        {"role": "system", "content": HYPOTHESIS_SYSTEM},
        {
            "role": "user",
            "content": HYPOTHESIS_USER.format(
                plan_json=plan_json,
                schema_json=schema_json,
                business_context=business_context or "Not provided.",
            ),
        },
    ]


# ── Executor Agent ────────────────────────────────────────────────────────────
# Used as the instructions= parameter for the Agent SDK Agent object.

EXECUTOR_SYSTEM = """You are a data analysis executor. You receive a specific hypothesis to test.

You have two tools available:
- get_dataset_info: call this tool to learn column names, dtypes, and data shape
- execute_python: call this tool to run Python code against the dataset

Your workflow:
1. Call the get_dataset_info tool to understand the exact column names and data types.
2. Call the execute_python tool with Python code to test the hypothesis.
3. Use print() to label and surface ALL numeric results clearly.
4. Create at least one visualization per hypothesis — always label axes and add a title.
5. If your code errors, read the error and retry once with a fix.
6. After collecting results, write a 3-5 sentence plain-English summary of what the results show.

Available libraries (pre-injected, no import statement needed):
- df: the pandas DataFrame
- pd: pandas   |  np: numpy
- plt: matplotlib.pyplot   |  sns: seaborn
- stats: scipy.stats  (e.g. stats.ttest_ind, stats.chi2_contingency, stats.pearsonr)
- sm: statsmodels.api  (e.g. sm.OLS, sm.Logit)
- sklearn: scikit-learn  (e.g. from sklearn.cluster import KMeans)

Analysis type guidance:
- distribution: Use sns.histplot or plt.hist; print mean/median/std/skewness
- correlation: Use df.corr(numeric_only=True); plot heatmap with sns.heatmap
- segmentation: from sklearn.cluster import KMeans; fit on scaled numeric cols; sns.scatterplot colored by label
- feature_importance: from sklearn.ensemble import RandomForestClassifier/Regressor; print feature_importances_ sorted
- t-test: stats.ttest_ind(group_a, group_b); print t-stat, p-value, sample sizes
- chi-squared: stats.chi2_contingency(pd.crosstab(col_a, col_b)); print chi2, p, dof
- cohort: pd.Grouper + pivot_table; line chart per cohort
- anomaly_detection: rolling z-score (np.abs(stats.zscore(series)) > 3); print anomaly count and dates

Always print n (sample size) before any statistical result."""


def get_executor_messages(hypothesis: dict, schema_json: str) -> list[dict]:
    """Returns the initial messages list for a single hypothesis execution run."""
    return [
        {
            "role": "user",
            "content": (
                f"Test this hypothesis:\n\n"
                f"{json.dumps(hypothesis, indent=2)}\n\n"
                f"Dataset schema for reference:\n{schema_json}"
            ),
        }
    ]


# ── Critic Agent ──────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a statistical reviewer validating data analysis results.

You receive:
- The hypothesis that was tested
- The text output (stdout) from the code execution
- The error log (if any)
- The number of figures generated (you do NOT see the actual images)

Your job is to assess:
1. Statistical validity: Is the test method appropriate? Is the sample size adequate?
2. Logical soundness: Does the conclusion follow from the output?
3. Practical significance: Is the effect size meaningful, not just statistically significant?
4. Data quality issues: Could missing values, outliers, or class imbalance invalidate the result?

Be specific about issues — vague critiques like "sample size may be small" are not useful.
Cite actual numbers from the stdout when raising issues.

Respond with JSON only. No explanation outside the JSON."""

CRITIC_USER = """Hypothesis tested:
{hypothesis_json}

Code execution output (stdout):
{stdout}

Error (if any):
{error}

Figures generated: {figures_count}

Assess the result. Respond with JSON only:
{{
  "verdict": "supported | refuted | inconclusive",
  "confidence": 0.0,
  "issues": ["specific issue 1", "specific issue 2"],
  "recommendation": "what additional test or data would resolve ambiguity"
}}

confidence is a float from 0.0 (completely uncertain) to 1.0 (highly confident in verdict)."""


def get_critic_messages(hypothesis: dict, execution_result: dict) -> list[dict]:
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": CRITIC_USER.format(
                hypothesis_json=json.dumps(hypothesis, indent=2),
                stdout=execution_result.get("stdout", "(no output)") or "(no output)",
                error=execution_result.get("error", "None") or "None",
                figures_count=execution_result.get("figures_count", 0),
            ),
        },
    ]


# ── Synthesizer Agent ─────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM = """You are a business insight synthesizer.
You receive the results of multiple hypothesis tests, each with a statistical verdict and critique.

Your audience is a business stakeholder who does not read code or statistics.

Rules:
1. Lead with the single most surprising or actionable finding.
2. Translate statistical results into business language.
3. Do NOT describe methodology. Do NOT say "as seen in the chart" or "the code shows".
4. Surface non-obvious patterns — things the stakeholder would not have guessed.
5. Be honest about limitations and caveats.
6. Next steps must be concrete and specific (e.g. "A/B test X against Y" not "investigate further").
7. story: a 2-3 sentence flowing narrative as if briefing a board member — connect causes to effects, use concrete numbers, no methodology.

Respond with JSON only. No explanation outside the JSON."""

SYNTHESIZER_USER = """Business context: {business_context}

Analysis results:
{results_json}

Synthesize all results into business insights. Respond with JSON only:
{{
  "headline": "the single most important finding, one sentence, written for a business audience",
  "story": "2-3 sentence board-level narrative connecting causes to effects with concrete numbers",
  "insights": [
    "insight 1 (most surprising or actionable)",
    "insight 2",
    "insight 3"
  ],
  "caveats": [
    "data quality or statistical limitation 1"
  ],
  "next_steps": [
    "concrete recommended action 1"
  ]
}}"""


def get_synthesizer_messages(
    hypotheses: list[dict],
    execution_results: list[dict],
    critique_results: list[dict],
    business_context: str,
) -> list[dict]:
    # Build a compact summary for each hypothesis
    results = []
    for i, hyp in enumerate(hypotheses):
        exec_r = execution_results[i] if i < len(execution_results) else {}
        crit_r = critique_results[i] if i < len(critique_results) else {}
        results.append(
            {
                "hypothesis": hyp.get("statement", ""),
                "test_method": hyp.get("test_method", ""),
                "stdout": (exec_r.get("stdout", "") or "")[
                    :2000
                ],  # cap to avoid token overflow
                "verdict": crit_r.get("verdict", "inconclusive"),
                "confidence": crit_r.get("confidence", 0.0),
                "issues": crit_r.get("issues", []),
                "recommendation": crit_r.get("recommendation", ""),
            }
        )

    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": SYNTHESIZER_USER.format(
                business_context=business_context or "Not provided.",
                results_json=json.dumps(results, indent=2),
            ),
        },
    ]
