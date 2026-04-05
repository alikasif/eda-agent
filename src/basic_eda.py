"""
Basic EDA — automatically runs 3-step exploratory analysis on upload.

Steps follow the framework from:
https://medium.com/data-science/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6

  Step 1: Dataset Overview & Descriptive Statistics
  Step 2: Feature Assessment & Visualization (univariate + multivariate)
  Step 3: Data Quality Evaluation
"""

from collections.abc import Callable

import pandas as pd

from tools import execute_python

# ── EDA code blocks ───────────────────────────────────────────────────────────

_STEP1_CODE = """
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\\nColumn Types:")
for dtype_name, group in df.dtypes.groupby(df.dtypes.astype(str)):
    print(f"  {dtype_name}: {list(group.index)}")

print("\\nMissing Values:")
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing = pd.DataFrame({"count": null_counts, "pct_%": null_pct})
missing = missing[missing["count"] > 0].sort_values("pct_%", ascending=False)
if len(missing) > 0:
    print(missing.to_string())
else:
    print("  No missing values found.")

print(f"\\nDuplicate rows: {df.duplicated().sum():,}")

print("\\nDescriptive Statistics (Numeric):")
numeric = df.select_dtypes(include="number")
if not numeric.empty:
    print(numeric.describe().round(3).to_string())
else:
    print("  No numeric columns.")
"""

_STEP2_CODE = """
import numpy as _np

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# ── Univariate: numeric distributions ────────────────────────────────────────
if numeric_cols:
    n = min(len(numeric_cols), 12)
    cols_to_plot = numeric_cols[:n]
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = _np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col].dropna(), kde=True, ax=axes_flat[i])
        axes_flat[i].set_title(col)
        axes_flat[i].set_xlabel("")
    for j in range(len(cols_to_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Numeric Feature Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    print(f"Plotted distributions for {len(cols_to_plot)} numeric column(s).")

# ── Univariate: categorical counts ───────────────────────────────────────────
if categorical_cols:
    n = min(len(categorical_cols), 6)
    cols_to_plot = categorical_cols[:n]
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = _np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(cols_to_plot):
        top_vals = df[col].value_counts().head(10)
        sns.barplot(x=top_vals.values, y=top_vals.index.astype(str), ax=axes_flat[i])
        axes_flat[i].set_title(col)
        axes_flat[i].set_xlabel("Count")
    for j in range(len(cols_to_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Categorical Feature Counts (top 10)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    print(f"Plotted counts for {len(cols_to_plot)} categorical column(s).")

# ── Multivariate: Spearman correlation heatmap ───────────────────────────────
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(method="spearman")
    dim = max(8, len(numeric_cols))
    fig, ax = plt.subplots(figsize=(dim, max(6, dim - 1)))
    sns.heatmap(
        corr,
        annot=len(numeric_cols) <= 20,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.show()
    print("Spearman correlation matrix plotted.")
"""

_STEP3_CODE = """
import numpy as _np

print("=== Data Quality Report ===\\n")

# Missing value severity
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
cols_with_nulls = null_pct[null_pct > 0].sort_values(ascending=False)
if len(cols_with_nulls) > 0:
    print("Missing values:")
    for col, pct in cols_with_nulls.items():
        severity = "HIGH" if pct > 30 else ("MEDIUM" if pct > 5 else "LOW")
        print(f"  [{severity}] {col}: {pct:.1f}% missing")
else:
    print("Missing values: none found.")

# Constant / zero-variance features
print("\\nConstant / near-constant features:")
low_var = [col for col in df.columns if df[col].nunique() <= 1]
if low_var:
    for col in low_var:
        print(f"  {col}: {df[col].nunique()} unique value(s)")
else:
    print("  None found.")

# Highly correlated numeric pairs (|r| > 0.85)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(_np.triu(_np.ones(corr.shape), k=1).astype(bool))
    high_corr = [
        (col, row, corr.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > 0.85
    ]
    print("\\nHighly correlated feature pairs (|r| > 0.85):")
    if high_corr:
        for c1, c2, r in sorted(high_corr, key=lambda x: -x[2]):
            print(f"  {c1} ↔ {c2}: {r:.3f}")
    else:
        print("  None found.")

# Class imbalance check for low-cardinality categoricals
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
target_candidates = [c for c in categorical_cols if 2 <= df[c].nunique() <= 20]
if target_candidates:
    print("\\nCategorical class distribution:")
    for col in target_candidates[:5]:
        pcts = (df[col].value_counts(normalize=True) * 100).round(1)
        top_pct = pcts.iloc[0]
        label = "IMBALANCED" if top_pct > 70 else "OK"
        print(
            f"  [{label}] {col}: {df[col].nunique()} classes, "
            f"top = {pcts.index[0]!r} ({top_pct:.1f}%)"
        )
"""

# ── Step metadata ─────────────────────────────────────────────────────────────

_STEPS: list[tuple[str, str, str]] = [
    (
        "Step 1 — Dataset Overview & Descriptive Statistics",
        "overview",
        _STEP1_CODE,
    ),
    (
        "Step 2 — Feature Assessment & Visualization",
        "features",
        _STEP2_CODE,
    ),
    (
        "Step 3 — Data Quality Evaluation",
        "quality",
        _STEP3_CODE,
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────


def run_basic_eda(
    df: pd.DataFrame,
    render_cb: Callable[[dict], None],
) -> list[dict]:
    """Run the 3-step basic EDA and stream render items via render_cb.

    Args:
        df: The uploaded DataFrame to analyse.
        render_cb: Called immediately for each rendered item so the UI can
            display results in real time.

    Returns:
        Ordered list of all render items produced (for later replay).
    """
    items: list[dict] = []

    def emit(item: dict) -> None:
        items.append(item)
        render_cb(item)

    for title, stage, code in _STEPS:
        emit({"type": "stage_header", "stage": stage, "content": title})
        result = execute_python(code, df)

        if result["stdout"].strip():
            emit({"type": "text", "content": f"```\n{result['stdout'].strip()}\n```"})

        for fig_b64 in result["figures"]:
            emit({"type": "figure", "content": fig_b64})

        if result["error"]:
            emit({"type": "error", "content": result["error"]})

    return items
