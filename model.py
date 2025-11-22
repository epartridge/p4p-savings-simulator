"""Model utilities for the P4P savings simulator."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

import pandas as pd

# Column name constants
REGION = "Region"
DC_ID = "DC ID"
DC_NAME = "DC Name"
DC_NUMBER_NAME = "DC Number Name"
MONTH = "Month"
PERCENT_COMMITMENT = "%  Commitment"
DOLLAR_IMPACT = "Dollar Impact"
CPH_IMPACT = "CPH Impact"
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
    DC_ID,
    DC_NAME,
    DC_NUMBER_NAME,
    MONTH,
    PERCENT_COMMITMENT,
    DOLLAR_IMPACT,
    CPH_IMPACT,
    LIVE,
)

REQUIRED_INPUT_COLUMNS: Sequence[str] = tuple(
    column for column in REQUIRED_COLUMNS if column != DC_NUMBER_NAME
)


def _format_dc_identifier(value: object) -> str:
    """Normalize DC IDs read from Excel into clean strings.

    Integers or floats that represent whole numbers are rendered without a decimal to
    match the prior DC Number format. Any whitespace is stripped, and missing values
    become an empty string.
    """

    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def format_dc_number_name(dc_id: object, dc_name: object) -> str:
    """Concatenate DC ID and name for display in the UI."""

    dc_id_text = _format_dc_identifier(dc_id)
    dc_name_text = "" if pd.isna(dc_name) else str(dc_name).strip()

    if dc_id_text and dc_name_text:
        return f"{dc_id_text} - {dc_name_text}"
    return dc_id_text or dc_name_text


def load_inputs(excel_source: object = "data/p4p_template.xlsx") -> pd.DataFrame:
    """
    Load the P4P template from Excel, validate required columns, and return a DataFrame.

    Args:
        excel_source: Path or file-like object pointing to the P4P Excel template.

    Raises:
        ValueError: If any required columns are missing from the sheet.

    Returns:
        DataFrame containing the template data.
    """

    df = pd.read_excel(excel_source, engine="openpyxl")

    missing_columns: List[str] = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df[DC_NUMBER_NAME] = df.apply(lambda row: format_dc_number_name(row[DC_ID], row[DC_NAME]), axis=1)

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


def is_locked_dc(df: pd.DataFrame, dc_number: object) -> bool:
    """Return True if a DC is governed by a locked go-live schedule.

    The lookup is based on ``DC Name`` because the lock configuration is keyed by
    name. The helper is intentionally lightweight so it can be reused anywhere we
    need to treat locked DCs differently (e.g., when applying go-live caps).
    """

    dc_rows = df[df[DC_ID] == dc_number]
    if dc_rows.empty:
        return False
    return dc_rows[DC_NAME].iloc[0] in DC_LIVE_LOCKS


def build_greedy_schedule(
    df: pd.DataFrame,
    target_savings: float,
    max_initial_golives_per_month: int | None = None,
    use_late_month_bias: bool = False,
    late_month_bias: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    """
    Starting from all Live? = 'No', greedily turn rows live one building at a time.

    Buildings (DCs) are ordered by their total annual impact and, within each
    building, months are evaluated from latest to earliest. This ensures that if
    the target is not met with later months, the algorithm keeps pulling earlier
    months for the same building before moving on.

    Args:
        df: Input DataFrame containing at least the Dollar Impact column.
        target_savings: Target cumulative savings to achieve.
        use_late_month_bias: Toggle to enable soft boosting of later months when
            choosing go-live candidates. Defaults to False.
        late_month_bias: Optional multiplier that softly boosts later months when
            choosing go-live candidates. A value of 0.0 keeps the default
            behavior.
        max_initial_golives_per_month: Maximum number of DCs allowed to begin
            their go-live in the same month. If None, no limit is applied.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(
        scheduled_df[DOLLAR_IMPACT], errors="coerce"
    ).fillna(0.0)
    scheduled_df["_month_order"] = scheduled_df[MONTH].apply(month_order_value)

    # Apply any DC-specific live locks up front
    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    month_orders = sorted(scheduled_df["_month_order"].unique())
    month_index = {month_order: idx for idx, month_order in enumerate(month_orders)}
    num_months = len(month_orders)
    month_scale = np.arange(num_months, dtype=float) / max(1, num_months - 1)
    month_weight = 1.0 + late_month_bias * month_scale

    def first_live_month_index(dc_number: object) -> int | None:
        dc_rows = scheduled_df[scheduled_df[DC_ID] == dc_number].sort_values(
            "_month_order"
        )
        live_mask = _normalize_live_column(dc_rows[LIVE]) == "yes"
        if live_mask.any():
            first_month_order = dc_rows.loc[live_mask, "_month_order"].iloc[0]
            return month_index[first_month_order]
        return None

    initial_counts = np.zeros(len(month_orders), dtype=int)
    for dc_number in scheduled_df[DC_ID].unique():
        first_live_idx = first_live_month_index(dc_number)
        if first_live_idx is not None:
            initial_counts[first_live_idx] += 1

    # Prioritize the highest impact buildings so we fill their calendars first.
    dc_priority = (
        scheduled_df.groupby(DC_ID)[DOLLAR_IMPACT]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    cumulative_savings = 0.0
    for dc_number in dc_priority:
        if cumulative_savings >= target_savings:
            break

        dc_rows = scheduled_df[scheduled_df[DC_ID] == dc_number]
        candidate_month_orders = sorted(
            dc_rows["_month_order"].unique(), reverse=True
        )

        if not use_late_month_bias or late_month_bias == 0.0:
            for month_order in candidate_month_orders:
                if cumulative_savings >= target_savings:
                    break

                month_idx = month_index[month_order]
                current_first_idx = first_live_month_index(dc_number)
                if (
                    max_initial_golives_per_month is not None
                    and month_idx != current_first_idx
                    and initial_counts[month_idx] >= max_initial_golives_per_month
                ):
                    continue

                dc_mask = scheduled_df[DC_ID] == dc_number
                scheduled_df.loc[
                    dc_mask & (scheduled_df["_month_order"] >= month_order), LIVE
                ] = "Yes"

                # Re-apply DC live locks after each change
                scheduled_df = apply_dc_live_locks(
                    scheduled_df, preserve_month_order=True
                )

                live_mask = _normalize_live_column(scheduled_df[LIVE]) == "yes"
                cumulative_savings = float(
                    scheduled_df.loc[live_mask, DOLLAR_IMPACT].sum()
                )

                if max_initial_golives_per_month is not None:
                    new_first_idx = first_live_month_index(dc_number)
                    if new_first_idx is not None and new_first_idx != current_first_idx:
                        if current_first_idx is not None:
                            initial_counts[current_first_idx] -= 1
                        initial_counts[new_first_idx] += 1
        else:
            remaining_month_orders = candidate_month_orders
            while cumulative_savings < target_savings and remaining_month_orders:
                dc_mask = scheduled_df[DC_ID] == dc_number
                current_first_idx = first_live_month_index(dc_number)
                current_live_mask = (
                    _normalize_live_column(scheduled_df[LIVE]) == "yes"
                )

                best_month_order: int | None = None
                best_adjusted_value: float | None = None

                for month_order in remaining_month_orders:
                    month_idx = month_index[month_order]
                    if (
                        max_initial_golives_per_month is not None
                        and month_idx != current_first_idx
                        and initial_counts[month_idx] >= max_initial_golives_per_month
                    ):
                        continue

                    month_mask = dc_mask & (
                        scheduled_df["_month_order"] >= month_order
                    )
                    incremental_mask = month_mask & ~current_live_mask
                    base_value = float(
                        scheduled_df.loc[incremental_mask, DOLLAR_IMPACT].sum()
                    )
                    adjusted_value = base_value * month_weight[month_idx]

                    if (
                        best_adjusted_value is None
                        or adjusted_value > best_adjusted_value
                    ):
                        best_month_order = month_order
                        best_adjusted_value = adjusted_value

                if best_month_order is None or best_adjusted_value is None:
                    break

                month_idx = month_index[best_month_order]
                current_first_idx = first_live_month_index(dc_number)
                if (
                    max_initial_golives_per_month is not None
                    and month_idx != current_first_idx
                    and initial_counts[month_idx] >= max_initial_golives_per_month
                ):
                    remaining_month_orders = [
                        order for order in remaining_month_orders if order != best_month_order
                    ]
                    continue

                scheduled_df.loc[
                    dc_mask & (scheduled_df["_month_order"] >= best_month_order), LIVE
                ] = "Yes"

                # Re-apply DC live locks after each change
                scheduled_df = apply_dc_live_locks(
                    scheduled_df, preserve_month_order=True
                )

                live_mask = _normalize_live_column(scheduled_df[LIVE]) == "yes"
                cumulative_savings = float(
                    scheduled_df.loc[live_mask, DOLLAR_IMPACT].sum()
                )

                if max_initial_golives_per_month is not None:
                    new_first_idx = first_live_month_index(dc_number)
                    if new_first_idx is not None and new_first_idx != current_first_idx:
                        if current_first_idx is not None:
                            initial_counts[current_first_idx] -= 1
                        initial_counts[new_first_idx] += 1

                remaining_month_orders = [
                    order for order in remaining_month_orders if order < best_month_order
                ]

    # Ensure each DC is live in its final month and cascade forward, then re-lock
    scheduled_df = ensure_final_month_live(
        scheduled_df, preserve_month_order=True, only_if_currently_live=True
    )
    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    normalized_live = _normalize_live_column(scheduled_df[LIVE])
    scheduled_df["IsLive"] = normalized_live == "yes"
    scheduled_df["DollarImpactIncluded"] = scheduled_df[DOLLAR_IMPACT].where(
        scheduled_df["IsLive"], other=0.0
    )

    cumulative_savings = float(scheduled_df["DollarImpactIncluded"].sum())

    scheduled_df = scheduled_df.drop(columns=["_month_order"], errors="ignore")

    return scheduled_df, cumulative_savings



def build_region_grouped_schedule(
    df: pd.DataFrame,
    target_savings: float,
    max_initial_golives_per_month: int | None = None,
    use_late_month_bias: bool = False,
    late_month_bias: float = 0.0,
) -> tuple[pd.DataFrame, float]:
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
        use_late_month_bias: Toggle to enable soft boosting of later months when
            choosing go-live candidates. Defaults to False.
        late_month_bias: Optional multiplier that softly boosts later months when
            choosing go-live candidates. A value of 0.0 keeps the default
            behavior.
        max_initial_golives_per_month: Maximum number of DCs allowed to begin
            their go-live in the same month. If None, no limit is applied.

    Returns:
        A tuple of (scheduled DataFrame with updated Live? and helper columns, cumulative savings achieved).
    """

    scheduled_df = df.copy()
    scheduled_df[LIVE] = "No"
    scheduled_df[DOLLAR_IMPACT] = pd.to_numeric(
        scheduled_df[DOLLAR_IMPACT], errors="coerce"
    ).fillna(0.0)
    scheduled_df["_month_order"] = scheduled_df[MONTH].apply(month_order_value)

    # Apply DC-specific live locks up front so their schedules are preserved and the
    # go-live cap calculations are based on the correct baseline state.
    scheduled_df = apply_dc_live_locks(scheduled_df, preserve_month_order=True)

    month_orders = sorted(scheduled_df["_month_order"].unique())
    month_index = {month_order: idx for idx, month_order in enumerate(month_orders)}
    num_months = len(month_orders)
    month_scale = np.arange(num_months, dtype=float) / max(1, num_months - 1)
    month_weight = 1.0 + late_month_bias * month_scale

    def first_live_month_index(dc_number: object) -> int | None:
        dc_rows = scheduled_df[scheduled_df[DC_ID] == dc_number].sort_values(
            "_month_order"
        )
        live_mask = _normalize_live_column(dc_rows[LIVE]) == "yes"
        if live_mask.any():
            first_month_order = dc_rows.loc[live_mask, "_month_order"].iloc[0]
            return month_index[first_month_order]
        return None

    initial_counts = np.zeros(len(month_orders), dtype=int)
    for dc_number in scheduled_df[DC_ID].unique():
        first_live_idx = first_live_month_index(dc_number)
        if first_live_idx is not None:
            initial_counts[first_live_idx] += 1

    region_month_savings = (
        scheduled_df.groupby([REGION, MONTH], dropna=False)[DOLLAR_IMPACT]
        .sum()
        .reset_index()
    )
    region_month_savings["_month_order"] = region_month_savings[MONTH].apply(
        month_order_value
    )

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
        if not use_late_month_bias or late_month_bias == 0.0:
            iter_rows = region_rows.sort_values(
                by=["_month_order", DOLLAR_IMPACT], ascending=[False, False]
            ).iterrows()
        else:
            weighted_rows = region_rows.copy()
            weighted_rows["adjusted_value"] = weighted_rows.apply(
                lambda row: row[DOLLAR_IMPACT]
                * month_weight[month_index[row["_month_order"]]],
                axis=1,
            )
            iter_rows = weighted_rows.sort_values(
                by=["adjusted_value", "_month_order"], ascending=[False, False]
            ).iterrows()

        for _, row in iter_rows:
            if cumulative_savings >= target_savings:
                break

            region_mask = scheduled_df[REGION] == row[REGION]
            month_mask = scheduled_df[MONTH] == row[MONTH]

            if max_initial_golives_per_month is not None:
                candidate_idx = month_index[row["_month_order"]]
                proposed_counts = initial_counts.copy()
                can_schedule = True

                for dc_number in scheduled_df.loc[
                    region_mask & month_mask, DC_ID
                ].unique():
                    current_first_idx = first_live_month_index(dc_number)
                    if current_first_idx == candidate_idx:
                        continue
                    if proposed_counts[candidate_idx] >= max_initial_golives_per_month:
                        can_schedule = False
                        break
                    proposed_counts[candidate_idx] += 1
                    if current_first_idx is not None:
                        proposed_counts[current_first_idx] -= 1

                if not can_schedule:
                    continue

                initial_counts = proposed_counts

            scheduled_df.loc[region_mask & month_mask, LIVE] = "Yes"

            # Enforce that once a DC goes live in a month, it stays live for the remainder of the year.
            scheduled_df = enforce_forward_live_rows(
                scheduled_df, preserve_month_order=True
            )
            # Apply DC-specific live locks on top
            scheduled_df = apply_dc_live_locks(
                scheduled_df, preserve_month_order=True
            )

            live_mask = _normalize_live_column(scheduled_df[LIVE]) == "yes"
            cumulative_savings = float(
                scheduled_df.loc[live_mask, DOLLAR_IMPACT].sum()
            )

    # Ensure each DC is live in its final month and cascade forward, then re-lock
    scheduled_df = ensure_final_month_live(
        scheduled_df, preserve_month_order=True, only_if_currently_live=True
    )
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
        df: DataFrame containing at least Month, DC ID, and Live? columns.
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
    for _, group in enforced_df.sort_values("_month_order").groupby(DC_ID, sort=False):
        live_seen = False
        for idx in group.index:
            live_value = str(enforced_df.at[idx, LIVE]).strip().lower() == "yes"
            live_seen = live_seen or live_value
            enforced_df.at[idx, LIVE] = "Yes" if live_seen else "No"

    if added_month_order and not preserve_month_order:
        enforced_df = enforced_df.drop(columns=["_month_order"], errors="ignore")

    return enforced_df


def ensure_final_month_live(
    df: pd.DataFrame,
    preserve_month_order: bool = False,
    only_if_currently_live: bool = False,
) -> pd.DataFrame:
    """Guarantee that every DC is live in its latest month and cascade forward.

    Args:
        df: DataFrame containing at least Month, DC ID, and Live? columns.
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

    final_month_per_dc = ensured_df.groupby(DC_ID)["_month_order"].transform("max")
    if only_if_currently_live:
        live_mask = _normalize_live_column(ensured_df[LIVE]) == "yes"
        currently_live_dcs = set(ensured_df.loc[live_mask, DC_ID])
        locked_dcs = {dc_number for dc_number in ensured_df[DC_ID].unique() if is_locked_dc(ensured_df, dc_number)}
        eligible_dcs = currently_live_dcs | locked_dcs
        final_month_mask = ensured_df["_month_order"] == final_month_per_dc
        ensured_df.loc[final_month_mask & ensured_df[DC_ID].isin(eligible_dcs), LIVE] = "Yes"
    else:
        ensured_df.loc[ensured_df["_month_order"] == final_month_per_dc, LIVE] = "Yes"

    ensured_df = enforce_forward_live_rows(ensured_df, preserve_month_order=True)

    if added_month_order and not preserve_month_order:
        ensured_df = ensured_df.drop(columns=["_month_order"], errors="ignore")

    return ensured_df
