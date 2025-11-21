import pandas as pd
import streamlit as st

from model import (
    REGION,
    DC_NUMBER_NAME,
    MONTH,
    MONTH_ORDER,
    LIVE,
    load_inputs,
    calculate_scenario_savings,
    build_greedy_schedule,
    build_region_grouped_schedule,
)


@st.cache_data
def load_template_df() -> pd.DataFrame:
    return load_inputs()


def normalize_live_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower() == "yes"


def ordered_month_labels(months: pd.Index | list) -> list:
    month_set = set(months)
    ordered = [month for month in MONTH_ORDER if month in month_set]
    extras = [month for month in months if month not in ordered]
    return ordered + extras


def enforce_forward_month_selection(df: pd.DataFrame, month_columns: list[str]) -> pd.DataFrame:
    """Ensure that selecting a month also selects every later month in the year."""

    ordered_months = ordered_month_labels(month_columns)
    enforced_df = df.copy()
    for idx, row in enforced_df.iterrows():
        seen_selected = False
        for month in ordered_months:
            current_value = bool(row.get(month, False))
            seen_selected = seen_selected or current_value
            if seen_selected:
                enforced_df.at[idx, month] = True
    return enforced_df


def align_month_dtype(month_values: pd.Series, template_months: pd.Series) -> pd.Series:
    template_numeric = pd.to_numeric(template_months, errors="coerce")
    if not template_numeric.isna().any():
        return pd.to_numeric(month_values, errors="coerce")
    return month_values.astype(str)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("P4P Savings Simulator")
    st.write(
        "Interactively explore manual and optimized activation schedules to reach your target FY26 savings."
    )

    df = load_template_df()

    target_savings = st.number_input(
        "Target FY26 savings ($)", min_value=0.0, value=0.0, step=100000.0, format="%.0f"
    )

    base_df = df.copy()
    base_df["IsLiveBool"] = normalize_live_bool(base_df[LIVE])

    base_pivot = base_df.pivot_table(
        index=[REGION, DC_NUMBER_NAME],
        columns=MONTH,
        values="IsLiveBool",
        aggfunc="first",
    )

    base_pivot = base_pivot.reindex(columns=ordered_month_labels(base_pivot.columns), fill_value=False)
    base_pivot = base_pivot.fillna(False)
    base_pivot.columns.name = None
    base_pivot_reset = base_pivot.reset_index()
    month_columns = [col for col in base_pivot_reset.columns if col not in [REGION, DC_NUMBER_NAME]]

    column_config = {
        REGION: st.column_config.TextColumn(label=REGION, disabled=True, width="medium"),
        DC_NUMBER_NAME: st.column_config.TextColumn(
            label=DC_NUMBER_NAME, disabled=True, width="large"
        ),
    }
    column_config.update(
        {month: st.column_config.CheckboxColumn(label=month, width="small") for month in month_columns}
    )

    column_order = [REGION, DC_NUMBER_NAME] + month_columns

    manual_tab, optimizations_tab = st.tabs(["Manual selection", "Optimizations"])

    with manual_tab:
        if "manual_pivot_data" not in st.session_state:
            st.session_state["manual_pivot_data"] = base_pivot_reset

        edited_pivot = st.data_editor(
            st.session_state["manual_pivot_data"],
            use_container_width=True,
            num_rows="dynamic",
            column_config=column_config,
            column_order=column_order,
            key="manual_pivot_editor",
        )

        enforced_pivot = enforce_forward_month_selection(edited_pivot, month_columns)
        if not enforced_pivot.equals(st.session_state["manual_pivot_data"]):
            st.session_state["manual_pivot_data"] = enforced_pivot
            st.experimental_rerun()

        edited_long = enforced_pivot.melt(
            id_vars=[REGION, DC_NUMBER_NAME],
            value_vars=month_columns,
            var_name="MonthStr",
            value_name="IsLiveBool",
        )

        edited_long[MONTH] = align_month_dtype(edited_long["MonthStr"], df[MONTH])
        edited_long = edited_long.drop(columns=["MonthStr"])

        updated_df = df.copy()
        updated_df[LIVE] = "No"

        merged = updated_df.merge(
            edited_long,
            how="left",
            on=[REGION, DC_NUMBER_NAME, MONTH],
        )

        updated_df[LIVE] = merged["IsLiveBool"].fillna(False).map({True: "Yes", False: "No"})

        manual_result_df, manual_total = calculate_scenario_savings(updated_df)

        st.markdown(f"**Current manual selection total savings: ${manual_total:,.0f}**")
        if target_savings > 0:
            if manual_total >= target_savings:
                st.info(
                    f"Target reached! You exceed the target by ${manual_total - target_savings:,.0f}."
                )
            else:
                st.warning(
                    f"Still short of target by ${target_savings - manual_total:,.0f}. "
                    "Consider adjusting selections or running optimizations."
                )

        st.dataframe(manual_result_df, use_container_width=True)

    with optimizations_tab:
        if target_savings <= 0:
            st.warning("Enter a positive target savings above to run optimizations.")
            return

        st.write("Current DC / Month view")
        st.data_editor(
            base_pivot_reset,
            use_container_width=True,
            column_config=column_config,
            column_order=column_order,
            disabled=True,
            hide_index=True,
            key="optimization_dc_month_grid",
        )

        col1, col2 = st.columns(2)
        run_greedy = col1.button("Run greedy by Dollar Impact", use_container_width=True)
        run_region_grouped = col2.button("Run region-grouped schedule", use_container_width=True)

        if run_greedy:
            greedy_df, greedy_total = build_greedy_schedule(df, target_savings)
            st.success(f"Greedy schedule total savings: ${greedy_total:,.0f}")
            st.dataframe(greedy_df, use_container_width=True)

        if run_region_grouped:
            region_df, region_total = build_region_grouped_schedule(df, target_savings)
            st.success(f"Region-grouped schedule total savings: ${region_total:,.0f}")
            st.dataframe(region_df, use_container_width=True)


if __name__ == "__main__":
    main()
