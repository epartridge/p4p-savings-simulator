"""Model utilities for the P4P savings simulator."""
from __future__ import annotations

from typing import List, Sequence

import pandas as pd

# Column name constants
REGION = "Region"
DC_NUMBER = "DC Number"
DC_NAME = "DC Name"
DC_NUMBER_NAME = "DC Number Name"
MONTH = "Month"
PERCENT_COMMITMENT = "%  Commitment"
DOLLAR_IMPACT = "Dollar Impact"
LIVE = "Live?"

# Month ordering where Jan is treated as month 12 (after Dec)
MONTH_ORDER = [
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
]

REQUIRED_COLUMNS: Sequence[str] = (
    REGION,
    DC_NUMBER,
    DC_NAME,
    DC_NUMBER_NAME,
    MONTH,
    PERCENT_COMMITMENT,
    DOLLAR_IMPACT,
    LIVE,
)


def load_inputs(excel_path: str = "data/p4p_template.xlsx") -> pd.DataFrame:
    """
    Load the P4P template from Excel, validate required columns, and return a DataFrame.

    Args:
        excel_path: Path to the P4P Excel template.

    Raises:
        ValueError: If any required columns are missing from the sheet.

    Returns:
        DataFrame containing the template data.
    """

    df = pd.read_excel(excel_path, engine="openpyxl")

    missing_columns: List[str] = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df


def _normalize_live_column(series: pd.Series) -> pd.Series:
    """Normalize the Live? column to lowercase strings with whitespace stripped."""

    return series.fillna("").astype(str).str.strip().str.lower()


def month_order_value(month_label: str) -> int:
    """Return the ordering index for a month where Jan is treated as month 12."""

    try:
        return MONTH_ORDER.index(month_label) + 1
    except ValueError:
        return len(MONTH_ORDER) + 1


def calculate_scenario_savings(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Given a DataFrame that includes a 'Live?' column, compute total savings for rows marked Live.

    Works on a copy of the DataFrame and appends helper columns to make the calculation explicit.

    Args:
        df: Input DataFrame containing at least the Live? and Dollar Impact columns.

    Returns:
        A tuple of (DataFrame with helper columns, total savings as float).
    """

    df_copy = df.copy()
    df_copy[DOLLAR_IMPACT] = pd.to_numeric(df_copy[DOLLAR_IMPACT], errors="coerce").fillna(0.0)

    normalized_live = _normalize_live_column(df_copy[LIVE])
    df_copy["IsLive"] = normalized_live == "yes"
    df_copy["DollarImpactIncluded"] = df_copy[DOLLAR_IMPACT].where(df_copy["IsLive"], other=0.0)

    total_savings = float(df_copy["DollarImpactIncluded"].sum())

    return df_copy, total_savings


def build_greedy_schedule(df: pd.DataFrame, target_savings: float) -> tuple[pd.DataFrame, float]:
    """
    Starting from all Live? = 'No', greedily turn rows live by highest Dollar Impact until target is met.

    Args:
        df: Input DataFrame containing at least the Dollar Impact column.
        target_savings: Target cumulative savings to achieve.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(scheduled_df[DOLLAR_IMPACT], errors="coerce").fillna(0.0)

    cumulative_savings = 0.0
    for idx in scheduled_df.sort_values(by=DOLLAR_IMPACT, ascending=False).index:
        if cumulative_savings >= target_savings:
            break
        cumulative_savings += float(scheduled_df.at[idx, DOLLAR_IMPACT])
        scheduled_df.at[idx, LIVE] = "Yes"

    normalized_live = _normalize_live_column(scheduled_df[LIVE])
    scheduled_df["IsLive"] = normalized_live == "yes"
    scheduled_df["DollarImpactIncluded"] = scheduled_df[DOLLAR_IMPACT].where(
        scheduled_df["IsLive"], other=0.0
    )

    cumulative_savings = float(scheduled_df["DollarImpactIncluded"].sum())

    return scheduled_df, cumulative_savings


def build_region_grouped_schedule(df: pd.DataFrame, target_savings: float) -> tuple[pd.DataFrame, float]:
    """
    Greedily schedule savings by selecting whole regions in a month together.

    Regions are activated one month at a time (all buildings within a region for that
    month go live together) in descending order of monthly savings until the target is met.

    Args:
        df: Input DataFrame containing at least Region, Month, and Dollar Impact columns.
        target_savings: Target cumulative savings to achieve.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(scheduled_df[DOLLAR_IMPACT], errors="coerce").fillna(0.0)

    region_month_savings = (
        scheduled_df.groupby([REGION, MONTH], dropna=False)[DOLLAR_IMPACT]
        .sum()
        .reset_index()
    )
    region_month_savings["_month_order"] = region_month_savings[MONTH].apply(month_order_value)

    cumulative_savings = 0.0
    for _, row in region_month_savings.sort_values(
        by=[DOLLAR_IMPACT, "_month_order"], ascending=[False, True]
    ).iterrows():
        if cumulative_savings >= target_savings:
            break

        region_mask = scheduled_df[REGION] == row[REGION]
        month_mask = scheduled_df[MONTH] == row[MONTH]
        scheduled_df.loc[region_mask & month_mask, LIVE] = "Yes"
        cumulative_savings += float(row[DOLLAR_IMPACT])

    normalized_live = _normalize_live_column(scheduled_df[LIVE])
    scheduled_df["IsLive"] = normalized_live == "yes"
    scheduled_df["DollarImpactIncluded"] = scheduled_df[DOLLAR_IMPACT].where(
        scheduled_df["IsLive"], other=0.0
    )

    cumulative_savings = float(scheduled_df["DollarImpactIncluded"].sum())

    return scheduled_df, cumulative_savings
