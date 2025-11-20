import pandas as pd
import streamlit as st

from model import (
    REGION,
    DC_NUMBER,
    DC_NUMBER_NAME,
    MONTH,
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


def sorted_month_labels(months: pd.Index | list) -> list:
    months_list = list(months)
    numeric_months = pd.to_numeric(pd.Series(months_list), errors="coerce")
    if not numeric_months.isna().any():
        sorted_pairs = sorted(zip(numeric_months.tolist(), months_list), key=lambda x: x[0])
        return [original for _, original in sorted_pairs]
    return sorted(months_list, key=lambda x: str(x))


def align_month_dtype(month_values: pd.Series, template_months: pd.Series) -> pd.Series:
    template_numeric = pd.to_numeric(template_months, errors="coerce")
    if not template_numeric.isna().any():
        return pd.to_numeric(month_values, errors="coerce")
    return month_values.astype(str)


def main() -> None:
    st.title("P4P Savings Simulator")
    st.write(
        "Interactively explore manual and optimized activation schedules to reach your target FY26 savings."
    )

    df = load_template_df()

    target_savings = st.number_input(
        "Target FY26 savings ($)", min_value=0.0, value=0.0, step=100000.0, format="%.0f"
    )

    manual_tab, optimizations_tab = st.tabs(["Manual selection", "Optimizations"])

    with manual_tab:
        manual_df = df.copy()
        manual_df["IsLiveBool"] = normalize_live_bool(manual_df[LIVE])

        pivot = manual_df.pivot_table(
            index=[REGION, DC_NUMBER, DC_NUMBER_NAME],
            columns=MONTH,
            values="IsLiveBool",
            aggfunc="first",
        )

        pivot = pivot.reindex(columns=sorted_month_labels(pivot.columns), fill_value=False)
        pivot = pivot.fillna(False)
        pivot.columns.name = None
        pivot_reset = pivot.reset_index()

        edited_pivot = st.data_editor(pivot_reset, use_container_width=True, num_rows="dynamic")

        month_columns = [
            col for col in edited_pivot.columns if col not in [REGION, DC_NUMBER, DC_NUMBER_NAME]
        ]

        edited_long = edited_pivot.melt(
            id_vars=[REGION, DC_NUMBER, DC_NUMBER_NAME],
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
            on=[REGION, DC_NUMBER, DC_NUMBER_NAME, MONTH],
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
