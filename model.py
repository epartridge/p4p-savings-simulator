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

# Fixed go-live windows for specific DCs (DC Name keyed)
DC_LIVE_LOCKS = {
    "Houston": {"Feb", "Mar", "Apr", "May", "Jun", "Jul"},
    "Winchester": {"Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"},
}

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


def apply_dc_live_locks(df: pd.DataFrame, preserve_month_order: bool = False) -> pd.DataFrame:
    """Apply fixed go-live rules for DCs that have locked schedules.

    Args:
        df: DataFrame containing at least DC Name, Month, and Live? columns.
        preserve_month_order: If True, retain any existing ``_month_order`` helper
            column created inside the function.

    Returns:
        A DataFrame with Live? values forced according to ``DC_LIVE_LOCKS`` and
        all other rows left untouched.
    """

    locked_df = df.copy()
    added_month_order = False

    if "_month_order" not in locked_df.columns:
        locked_df["_month_order"] = locked_df[MONTH].apply(month_order_value)
        added_month_order = True

    for dc_name, allowed_live_months in DC_LIVE_LOCKS.items():
        dc_mask = locked_df[DC_NAME] == dc_name
        if not dc_mask.any():
            continue

        locked_df.loc[dc_mask, LIVE] = "No"
        locked_df.loc[dc_mask & locked_df[MONTH].isin(allowed_live_months), LIVE] = "Yes"

    if added_month_order and not preserve_month_order:
        locked_df = locked_df.drop(columns=["_month_order"], errors="ignore")

    return locked_df


def build_greedy_schedule(df: pd.DataFrame, target_savings: float) -> tuple[pd.DataFrame, float]:
    """
    Starting from all Live? = 'No', greedily turn rows live one building at a time.

    Buildings (DCs) are ordered by their total annual impact and, within each
    building, months are evaluated from latest to earliest. This ensures that if
    the target is not met with later months, the algorithm keeps pulling earlier
    months for the same building before moving on.

    Args:
        df: Input DataFrame containing at least the Dollar Impact column.
        target_savings: Target cumulative savings to achieve.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(
        scheduled_df[DOLLAR_IMPACT], errors="coerce"
    ).fillna(0.0)
    scheduled_df["_month_order"] = scheduled_df[MONTH].apply(month_order_value)

    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    # Prioritize the highest impact buildings so we fill their calendars first.
    dc_priority = (
        scheduled_df.groupby(DC_NUMBER)[DOLLAR_IMPACT]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    cumulative_savings = 0.0
    for dc_number in dc_priority:
        if cumulative_savings >= target_savings:
            break

        dc_rows = scheduled_df[scheduled_df[DC_NUMBER] == dc_number]
        for month_order in sorted(dc_rows["_month_order"].unique(), reverse=True):
            if cumulative_savings >= target_savings:
                break

            dc_mask = scheduled_df[DC_NUMBER] == dc_number
            scheduled_df.loc[
                dc_mask & (scheduled_df["_month_order"] >= month_order), LIVE
            ] = "Yes"

            scheduled_df = apply_dc_live_locks(
                scheduled_df, preserve_month_order=True
            )

            live_mask = _normalize_live_column(scheduled_df[LIVE]) == "yes"
            cumulative_savings = float(scheduled_df.loc[live_mask, DOLLAR_IMPACT].sum())

    scheduled_df = ensure_final_month_live(scheduled_df, preserve_month_order=True)
    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    normalized_live = _normalize_live_column(scheduled_df[LIVE])
    scheduled_df["IsLive"] = normalized_live == "yes"
    scheduled_df["DollarImpactIncluded"] = scheduled_df[DOLLAR_IMPACT].where(
        scheduled_df["IsLive"], other=0.0
    )

    cumulative_savings = float(scheduled_df["DollarImpactIncluded"].sum())

    scheduled_df = scheduled_df.drop(columns=["_month_order"], errors="ignore")

    return scheduled_df, cumulative_savings


def build_region_grouped_schedule(df: pd.DataFrame, target_savings: float) -> tuple[pd.DataFrame, float]:
    """
    Greedily schedule savings by selecting whole regions in a month together.

    Regions are activated one month at a time (all buildings within a region for that
    month go live together) starting from later months and then by descending monthly savings
    until the target is met. If later months do not reach the target, the algorithm keeps
    walking backward through earlier months for the same region before moving on to the next
    region.

    Args:
        df: Input DataFrame containing at least Region, Month, and Dollar Impact columns.
        target_savings: Target cumulative savings to achieve.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(
        scheduled_df[DOLLAR_IMPACT], errors="coerce"
    ).fillna(0.0)
    scheduled_df["_month_order"] = scheduled_df[MONTH].apply(month_order_value)

    region_month_savings = (
        scheduled_df.groupby([REGION, MONTH], dropna=False)[DOLLAR_IMPACT]
        .sum()
        .reset_index()
    )
    region_month_savings["_month_order"] = region_month_savings[MONTH].apply(month_order_value)

    # Order regions by their total impact so the highest contributors are evaluated first.
    region_priority = (
        region_month_savings.groupby(REGION)[DOLLAR_IMPACT]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    cumulative_savings = 0.0
    for region in region_priority:
        if cumulative_savings >= target_savings:
            break

        region_rows = region_month_savings[region_month_savings[REGION] == region]
        for _, row in region_rows.sort_values(
            by=["_month_order", DOLLAR_IMPACT], ascending=[False, False]
        ).iterrows():
            if cumulative_savings >= target_savings:
                break

            region_mask = scheduled_df[REGION] == row[REGION]
            month_mask = scheduled_df[MONTH] == row[MONTH]
            scheduled_df.loc[region_mask & month_mask, LIVE] = "Yes"

            # Enforce that once a DC goes live in a month, it stays live for the
            # remainder of the year.
            scheduled_df = enforce_forward_live_rows(
                scheduled_df, preserve_month_order=True
            )
            scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

            live_mask = _normalize_live_column(scheduled_df[LIVE]) == "yes"
            cumulative_savings = float(scheduled_df.loc[live_mask, DOLLAR_IMPACT].sum())

    scheduled_df = ensure_final_month_live(scheduled_df, preserve_month_order=True)
    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    normalized_live = _normalize_live_column(scheduled_df[LIVE])
    scheduled_df["IsLive"] = normalized_live == "yes"
    scheduled_df["DollarImpactIncluded"] = scheduled_df[DOLLAR_IMPACT].where(
        scheduled_df["IsLive"], other=0.0
    )

    cumulative_savings = float(scheduled_df["DollarImpactIncluded"].sum())

    scheduled_df = scheduled_df.drop(columns=["_month_order"], errors="ignore")

    return scheduled_df, cumulative_savings


def enforce_forward_live_rows(df: pd.DataFrame, preserve_month_order: bool = False) -> pd.DataFrame:
    """Force months after a go-live to remain live for each DC.

    Args:
        df: DataFrame containing at least Month, DC Number, and Live? columns.
        preserve_month_order: If True, keep an existing ``_month_order`` helper
            column so callers that rely on it for sorting can continue to do so.

    Returns:
        A DataFrame with Live? values adjusted so that, for each DC, once a
        month is live all subsequent months are also live.
    """

    enforced_df = df.copy()
    added_month_order = False

    if "_month_order" not in enforced_df.columns:
        enforced_df["_month_order"] = enforced_df[MONTH].apply(month_order_value)
        added_month_order = True

    # Iterate through each DC in chronological order to cascade live months
    # forward. Sorting before grouping preserves the month order inside each
    # group iteration.
    for _, group in enforced_df.sort_values("_month_order").groupby(DC_NUMBER, sort=False):
        live_seen = False
        for idx in group.index:
            live_value = str(enforced_df.at[idx, LIVE]).strip().lower() == "yes"
            live_seen = live_seen or live_value
            enforced_df.at[idx, LIVE] = "Yes" if live_seen else "No"

    if added_month_order and not preserve_month_order:
        enforced_df = enforced_df.drop(columns=["_month_order"], errors="ignore")

    return enforced_df


def ensure_final_month_live(
    df: pd.DataFrame, preserve_month_order: bool = False
) -> pd.DataFrame:
    """Guarantee that every DC is live in its latest month and cascade forward.

    Args:
        df: DataFrame containing at least Month, DC Number, and Live? columns.
        preserve_month_order: If True, keep an existing ``_month_order`` helper
            column so callers relying on it can continue to do so.

    Returns:
        A DataFrame with Live? adjusted so that each DC is live in its final
        month of the calendar, with forward-live enforcement applied.
    """

    ensured_df = df.copy()
    added_month_order = False

    if "_month_order" not in ensured_df.columns:
        ensured_df["_month_order"] = ensured_df[MONTH].apply(month_order_value)
        added_month_order = True

    final_month_per_dc = ensured_df.groupby(DC_NUMBER)["_month_order"].transform("max")
    ensured_df.loc[ensured_df["_month_order"] == final_month_per_dc, LIVE] = "Yes"

    ensured_df = enforce_forward_live_rows(ensured_df, preserve_month_order=True)

    if added_month_order and not preserve_month_order:
        ensured_df = ensured_df.drop(columns=["_month_order"], errors="ignore")

    return ensured_df
