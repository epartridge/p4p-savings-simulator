"""Streamlit application for exploring manual and optimized go-live schedules.

The app exposes two tabs:
1. **Manual selection** lets the user toggle live months for each DC.
2. **Optimizations** runs helper algorithms to propose schedules that hit a
   target savings goal.

Extensive comments are included throughout so that developers new to Python or
Streamlit can understand the intent behind each block of code.
"""

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
    """Convert a column containing "Yes"/"No" text into booleans.

    The data can contain missing values or mixed casing (e.g., " yes"). This
    helper strips whitespace, lower-cases the text, and maps "yes" to ``True``
    with everything else treated as ``False``.
    """

    return series.fillna("").astype(str).str.strip().str.lower() == "yes"


def build_calendar_pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Transform the long-form data into a pivoted calendar grid.

    The resulting table has a row per DC and a column per month with boolean
    cells indicating whether the DC is live. We also return the list of month
    columns so the caller can configure Streamlit widgets consistently.
    """

    calendar_df = df.copy()
    calendar_df["IsLiveBool"] = normalize_live_bool(calendar_df[LIVE])

    pivot = calendar_df.pivot_table(
        index=[REGION, DC_NUMBER_NAME],
        columns=MONTH,
        values="IsLiveBool",
        aggfunc="first",
    )

    pivot = pivot.reindex(columns=ordered_month_labels(pivot.columns), fill_value=False)
    pivot = pivot.fillna(False)
    pivot.columns.name = None
    pivot_reset = pivot.reset_index()
    month_columns = [col for col in pivot_reset.columns if col not in [REGION, DC_NUMBER_NAME]]

    return pivot_reset, month_columns


def calendar_height(row_count: int) -> int:
    """Provide a generous table height to avoid scroll bars for the calendar grid.

    Streamlit's data editor defaults can introduce vertical scroll bars when
    there are many rows. This helper uses a simple formula to compute an
    appropriate height based on the number of rows so the entire calendar is
    visible at once.
    """

    base_height = 120  # space for headers and padding
    row_height = 38
    return max(base_height + row_count * row_height, 400)


def ordered_month_labels(months: pd.Index | list) -> list:
    """Return month labels ordered according to the canonical MONTH_ORDER.

    Any unexpected months that are not in MONTH_ORDER are appended at the end to
    preserve the data while keeping known months in a predictable order.
    """

    month_set = set(months)
    ordered = [month for month in MONTH_ORDER if month in month_set]
    extras = [month for month in months if month not in ordered]
    return ordered + extras


def enforce_forward_month_selection(df: pd.DataFrame, month_columns: list[str]) -> pd.DataFrame:
    """Ensure that selecting a month also selects every later month in the year.

    When a user marks a DC as live for a given month, it typically stays live in
    all subsequent months. This function scans across each row from the earliest
    month to the latest and auto-fills later months once the first ``True`` is
    encountered.
    """

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
    """Match the month data type to the template data for reliable merges.

    The template data may store months as numbers or strings. To ensure we can
    merge edited data back into the original frame, we coerce the edited values
    to the same type where possible.
    """

    template_numeric = pd.to_numeric(template_months, errors="coerce")
    if not template_numeric.isna().any():
        return pd.to_numeric(month_values, errors="coerce")
    return month_values.astype(str)


def main() -> None:
    """Render the Streamlit UI and orchestrate user interactions."""

    st.set_page_config(layout="wide")
    st.title("P4P Savings Simulator")
    st.write(
        "Interactively explore manual and optimized activation schedules to reach your target FY26 savings."
    )

    # Load the base dataset that describes each DC, its region, the month value,
    # and whether it is currently live. The dataset is cached by Streamlit to
    # avoid reloading on every interaction.
    df = load_template_df()

    # Users enter the target savings that optimization routines will try to
    # reach. Keeping ``step`` at a large increment makes entry easier for big
    # numbers.
    target_savings = st.number_input(
        "Target FY26 savings ($)", min_value=0.0, value=0.0, step=100000.0, format="%.0f"
    )

    # Build a pivoted version of the dataset that Streamlit can display as a
    # calendar grid. ``base_pivot_reset`` holds the table, and ``month_columns``
    # lists the month headers for later use when configuring widgets.
    base_df = df.copy()

    base_pivot_reset, month_columns = build_calendar_pivot(base_df)

    # Configure how each column appears inside the data editor. Region and DC
    # identifiers are read-only text, while the month columns become checkbox
    # columns to let users toggle live months.
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
        # Initialize session state for the manual calendar the first time the
        # tab is opened. Streamlit persists this across reruns to keep user
        # edits.
        if "manual_pivot_data" not in st.session_state:
            st.session_state["manual_pivot_data"] = base_pivot_reset

        st.subheader("DC Go-Live Calendar")
        edited_pivot = st.data_editor(
            st.session_state["manual_pivot_data"],
            use_container_width=True,
            num_rows="dynamic",
            column_config=column_config,
            column_order=column_order,
            height=calendar_height(len(st.session_state["manual_pivot_data"])),
            key="manual_pivot_editor",
        )

        # Apply the forward-fill rule so that once a DC goes live it stays live
        # in subsequent months. If the enforcement changes any values, store the
        # corrected data and trigger a rerun to refresh the UI with the new
        # state.
        enforced_pivot = enforce_forward_month_selection(edited_pivot, month_columns)
        if not enforced_pivot.equals(st.session_state["manual_pivot_data"]):
            st.session_state["manual_pivot_data"] = enforced_pivot
            rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun")
            rerun()

        # Convert the pivoted table back into a long format where each row is a
        # single (DC, month) combination. This makes it straightforward to join
        # with the original dataset and reuse existing calculation functions.
        edited_long = enforced_pivot.melt(
            id_vars=[REGION, DC_NUMBER_NAME],
            value_vars=month_columns,
            var_name="MonthStr",
            value_name="IsLiveBool",
        )

        # The melted data uses string month labels; convert them to match the
        # template's dtype (numeric or string) so we can merge back accurately.
        edited_long[MONTH] = align_month_dtype(edited_long["MonthStr"], df[MONTH])
        edited_long = edited_long.drop(columns=["MonthStr"])

        # Start from a clean copy of the base data and set every row to "No".
        # We'll fill in "Yes" only for entries toggled on in the edited pivot.
        updated_df = df.copy()
        updated_df[LIVE] = "No"

        # Merge the boolean edits back into the template to produce a full
        # dataset with consistent columns.
        merged = updated_df.merge(
            edited_long,
            how="left",
            on=[REGION, DC_NUMBER_NAME, MONTH],
        )

        # Map True/False back to the original "Yes"/"No" string values that the
        # calculation functions expect.
        updated_df[LIVE] = merged["IsLiveBool"].fillna(False).map({True: "Yes", False: "No"})

        # Calculate savings based on the user's manual selections. The helper
        # returns both the detailed result dataframe and the total dollar value.
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

        st.subheader("UPH Plan Output Format")
        st.dataframe(manual_result_df, use_container_width=True)

    with optimizations_tab:
        if target_savings <= 0:
            st.warning("Enter a positive target savings above to run optimizations.")
            return

        # Keep a separate calendar in session state for optimization results so
        # manual edits are not overwritten when running algorithms.
        if "optimization_calendar" not in st.session_state:
            st.session_state["optimization_calendar"] = base_pivot_reset

        st.subheader("Run Optimizations")
        col1, col2 = st.columns(2)
        run_greedy = col1.button("Run greedy by Dollar Impact", use_container_width=True)
        run_region_grouped = col2.button("Run region-grouped schedule", use_container_width=True)

        st.subheader("DC Go-Live Calendar")
        st.data_editor(
            st.session_state["optimization_calendar"],
            use_container_width=True,
            column_config=column_config,
            column_order=column_order,
            disabled=True,
            hide_index=True,
            height=calendar_height(len(st.session_state["optimization_calendar"])),
            key="optimization_dc_month_grid",
        )

        if run_greedy:
            # Greedy algorithm toggles on the largest dollar impacts first until
            # the target savings is met or exceeded.
            greedy_df, greedy_total = build_greedy_schedule(df, target_savings)
            st.success(f"Greedy schedule total savings: ${greedy_total:,.0f}")
            st.session_state["optimization_calendar"], _ = build_calendar_pivot(greedy_df)

            st.subheader("UPH Plan Output Format")
            st.dataframe(greedy_df, use_container_width=True)

        if run_region_grouped:
            # Region-grouped algorithm attempts to cluster go-lives within the
            # same region to limit operational churn.
            region_df, region_total = build_region_grouped_schedule(df, target_savings)
            st.success(f"Region-grouped schedule total savings: ${region_total:,.0f}")
            st.session_state["optimization_calendar"], _ = build_calendar_pivot(region_df)

            st.subheader("UPH Plan Output Format")
            st.dataframe(region_df, use_container_width=True)


if __name__ == "__main__":
    main()
