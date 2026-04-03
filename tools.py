import io
import json
import base64
import contextlib
import traceback

import numpy as np
import scipy
import scipy.stats
import statsmodels.api as _sm
import sklearn

import matplotlib

matplotlib.use("Agg")  # must be before pyplot import — prevents GUI window creation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Returns structural information about the loaded dataset."""
    col_info = []
    for col in df.columns:
        col_info.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(df[col].isna().mean() * 100, 2),
            }
        )

    cat_summary = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        top = df[col].value_counts().head(5).to_dict()
        cat_summary.append(
            {
                "column": col,
                "unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top.items()},
            }
        )

    return {
        "shape": list(df.shape),
        "columns": col_info,
        "numeric_summary": df.describe().to_string(),
        "categorical_summary": cat_summary,
        "sample": df.head(5).to_string(),
    }


def execute_python(code: str, df: pd.DataFrame) -> dict:
    """
    Executes Python code with df, pd, plt, sns injected into scope.

    Returns:
        {
            "stdout": str,
            "error": str | None,
            "figures": list[str]   # base64-encoded PNG strings
        }
    """
    stdout_capture = io.StringIO()
    figures = []
    error = None

    local_scope = {
        "df": df.copy(),  # copy so agent code can't mutate session state df
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "json": json,
        "np": np,
        "scipy": scipy,
        "stats": scipy.stats,  # stats.ttest_ind(), stats.chi2_contingency(), etc.
        "sm": _sm,  # sm.OLS(), sm.Logit(), etc.
        "sklearn": sklearn,  # from sklearn.cluster import KMeans, etc.
    }

    try:
        plt.close("all")
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<agent>", "exec"), local_scope)

        # Capture any open figures after exec completes
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            figures.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close("all")

    except Exception:
        error = traceback.format_exc()
        plt.close("all")

    return {
        "stdout": stdout_capture.getvalue(),
        "error": error,
        "figures": figures,
    }
