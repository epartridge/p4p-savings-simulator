"""Streamlit app to explore P4P savings scenarios."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from model import (
    DOLLAR_IMPACT,
    LIVE,
    MONTH,
    REGION,
    calculate_scenario_savings,
    load_inputs,
    month_order_value,
    build_greedy_schedule,
    build_region_grouped_schedule,
)


@st.cache_data
def _load_data(path: str = "data/p4p_template.xlsx") -> pd.DataFrame:
    """Load the P4P template data with caching."""

    return load_inputs(path)


def _sorted_months(df: pd.DataFrame) -> list[str]:
    """Return unique months sorted according to the fiscal ordering where Jan is last."""

    months = df[MONTH].dropna().astype(str).unique().tolist()
    return sorted(months, key=month_order_value)


def _manual_selector(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Render a manual selector grid for buildings and months, returning the updated DataFrame."""

    months = _sorted_months(df)
    buildings = df[[REGION, MONTH, "DC Number Name", "DC Number"]].copy()
    buildings["DC Number Name"] = buildings["DC Number Name"].fillna(buildings["DC Number"])
    building_keys = (
        buildings[["DC Number", "DC Number Name"]].drop_duplicates().to_dict("records")
    )

    st.caption(
        "Use the grid below to toggle specific building-month combinations. "
        "Each checkbox marks that building live for the selected month."
    )

    with st.form("manual_selection"):
        selections: Dict[Tuple[str, str], bool] = {}
        for building in building_keys:
            label = f"{building['DC Number Name']} (#{building['DC Number']})"
            st.write(f"**{label}**")
            cols = st.columns(len(months))
            for idx, month in enumerate(months):
                row_mask = (df["DC Number"] == building["DC Number"]) & (df[MONTH] == month)
                default_live = False
                if row_mask.any():
                    current = df.loc[row_mask, LIVE].iloc[0]
                    default_live = str(current).strip().lower() == "yes"
                selections[(building["DC Number"], month)] = cols[idx].checkbox(
                    month, value=default_live, key=f"{building['DC Number']}-{month}"
                )

        submitted = st.form_submit_button("Update scenario")

    updated = df.copy()
    if submitted:
        updated[LIVE] = "No"
        for (dc_number, month), is_live in selections.items():
            mask = (updated["DC Number"] == dc_number) & (updated[MONTH] == month)
            if is_live:
                updated.loc[mask, LIVE] = "Yes"

    return calculate_scenario_savings(updated)


def _optimizations(df: pd.DataFrame):
    """Render optimization helpers to reach a target savings amount."""

    st.subheader("Optimization helpers")
    target = st.number_input("Target savings", min_value=0.0, value=0.0, step=100000.0)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run greedy by Dollar Impact"):
            greedy_df, greedy_total = build_greedy_schedule(df, target)
            st.success(f"Greedy schedule total savings: ${greedy_total:,.2f}")
            st.dataframe(greedy_df)

    with col2:
        if st.button("Run region-grouped schedule"):
            grouped_df, grouped_total = build_region_grouped_schedule(df, target)
            st.success(
                "Region-grouped schedule total savings: " f"${grouped_total:,.2f}"
            )
            st.dataframe(grouped_df)


def main():
    st.title("P4P Savings Simulator")
    st.write(
        "Explore different ways to reach a target savings amount, including manual "
        "building-month toggles and automated optimizations."
    )
    df = _load_data()

    manual_tab, optim_tab = st.tabs(["Manual selection", "Optimizations"])

    with manual_tab:
        st.subheader("Manual building-month activation")
        manual_df, manual_total = _manual_selector(df)
        st.success(f"Current manual selection total savings: ${manual_total:,.2f}")
        st.dataframe(manual_df)

    with optim_tab:
        _optimizations(df)


if __name__ == "__main__":
    main()
