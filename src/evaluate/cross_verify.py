# src/evaluate/cross_verify.py
"""
DL severity vs VLM confidence cross-verification utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.vlm_verify import vlm_verify


def build_cross_verification_table(
    severity_df: pd.DataFrame,
    img_dir: str | Path,
    vlm_model,
    vlm_processor,
    pred_col: str = "pred_severity_pct",
) -> pd.DataFrame:
    """
    Run VLM verification on every row of severity_df and return an enriched table.
    """
    img_dir = Path(img_dir)
    rows: List[Dict[str, Any]] = []

    for _, r in severity_df.iterrows():
        img_path = img_dir / r["image"]
        verdict = vlm_verify(
            image=img_path,
            dl_severity_pct=float(r[pred_col]),
            model=vlm_model,
            processor=vlm_processor,
        )

        conf = verdict.get("confidence", np.nan)
        agrees = verdict.get("agrees_with_dl_score", None)
        flags = verdict.get("flags", [])
        if isinstance(flags, list):
            flags_str = ",".join(str(f) for f in flags)
        else:
            flags_str = str(flags)

        rows.append(
            {
                "image": r["image"],
                "dl_severity_pct": r[pred_col],
                "gt_severity_pct": r.get("gt_severity_pct"),
                "vlm_confidence": conf,
                "vlm_agrees": agrees,
                "vlm_bracket": verdict.get("visual_severity_bracket"),
                "vlm_flags": flags_str,
            }
        )

    return pd.DataFrame(rows)


def flag_outliers(
    cross_df: pd.DataFrame,
    threshold: float = 0.25,
    dl_col: str = "dl_severity_pct",
    conf_col: str = "vlm_confidence",
) -> pd.DataFrame:
    """
    Mark rows where |dl_severity/100 - vlm_confidence| > threshold.
    """
    df = cross_df.copy()
    dl_norm = (df[dl_col] / 100.0).clip(0, 1)
    disagreement = (dl_norm - df[conf_col].fillna(0.5)).abs()
    df["disagreement"] = disagreement
    df["outlier"] = disagreement > threshold
    return df


def plot_cross_verification(
    cross_df: pd.DataFrame,
    threshold: float = 0.25,
    save_path: str | Path = "cross_verification_matrix.png",
    top_k_labels: int = 5,
) -> None:
    """
    Scatter plot of DL severity vs VLM confidence with outlier highlighting.
    Only the top-k worst outliers receive short text labels.
    """
    df = flag_outliers(cross_df, threshold=threshold)

    fig, ax = plt.subplots(figsize=(8, 6))

    ok = df[~df["outlier"]]
    out = df[df["outlier"]]

    ax.scatter(
        ok["dl_severity_pct"],
        ok["vlm_confidence"],
        c="#4C956C",
        s=70,
        edgecolor="k",
        label="Agreement",
        zorder=3,
    )
    ax.scatter(
        out["dl_severity_pct"],
        out["vlm_confidence"],
        c="#C1666B",
        s=90,
        edgecolor="k",
        label="Disagreement",
        zorder=4,
    )

    # Annotate only the worst top_k outliers
    if len(out):
        top = out.sort_values("disagreement", ascending=False).head(top_k_labels)
        for _, r in top.iterrows():
            short = r["image"][:18] + "…" if len(str(r["image"])) > 20 else r["image"]
            ax.annotate(
                short,
                (r["dl_severity_pct"], r["vlm_confidence"]),
                fontsize=8,
                xytext=(6, 6),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    ax.set_xlabel("DL-calculated severity (%)")
    ax.set_ylabel("VLM confidence (0-1)")
    ax.set_title("DL vs. VLM cross-verification\n(red = flagged disagreement)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    n_out = int(df["outlier"].sum())
    print(
        f"{n_out}/{len(df)} images flagged as DL/VLM disagreements "
        f"(threshold={threshold})."
    )


def bracket_distance(b1: Optional[str], b2: Optional[str]) -> Optional[int]:
    """
    Ordinal distance between two severity brackets.
    Returns None if either bracket is missing / unknown.
    """
    order = [
        "healthy(0-2%)",
        "low(2-10%)",
        "moderate(10-25%)",
        "high(25-50%)",
        "severe(>50%)",
    ]
    if b1 not in order or b2 not in order:
        return None
    return abs(order.index(b1) - order.index(b2))


def cross_verify_brackets(
    dl_bracket: str,
    vlm_bracket: Optional[str],
    vlm_confidence: Optional[float],
    confidence_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Deterministic agreement logic used by the serving API.
    """
    if vlm_bracket is None or vlm_confidence is None:
        return {"status": "uncertain", "bracket_distance": None}

    if vlm_confidence < confidence_threshold:
        return {
            "status": "uncertain",
            "bracket_distance": bracket_distance(dl_bracket, vlm_bracket),
        }

    dist = bracket_distance(dl_bracket, vlm_bracket)
    if dist == 0:
        status = "agree"
    elif dist == 1:
        status = "borderline"
    else:
        status = "disagree"

    return {"status": status, "bracket_distance": dist}
