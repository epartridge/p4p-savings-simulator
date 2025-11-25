"""Streamlit application for exploring manual and optimized go-live schedules.

The app exposes two tabs:
1. **Manual selection** lets the user toggle live months for each DC.
2. **Optimizations** runs helper algorithms to propose schedules that hit a
   target savings goal.

Extensive comments are included throughout so that developers new to Python or
Streamlit can understand the intent behind each block of code.
"""

import io

import altair as alt
import pandas as pd
import streamlit as st

import model
from model import (
    REGION,
    DC_NUMBER_NAME,
    MONTH,
    MONTH_ORDER,
    LIVE,
    load_inputs,
    REQUIRED_COLUMNS,
    calculate_scenario_savings,
    ensure_final_month_live,
    apply_dc_live_locks,
    build_greedy_schedule,
    build_region_grouped_schedule,
)


TEMPLATE_COLUMNS = [
    REGION,
    DC_ID,
    DC_NAME,
    MONTH,
    "Ramp",
    "Baseline Hrs",
    "Hours Saved",
    DOLLAR_IMPACT,
    DOLLAR_IMPACT_WITH_FRINGE,
    LIVE,
    DC_NUMBER_NAME,
    PERCENT_COMMITMENT,
    CPH_IMPACT,
]
LATE_MONTH_BIAS_STRENGTH = 0.6


def inject_compact_uploader_css():
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploadDropzone"] {
            padding: 4px 8px !important;
        }
        div[data-testid="stFileUploadDropzone"] > div:nth-child(1) {
            min-height: 0 !important;
            padding: 4px 8px !important;
        }
        div[data-testid="stFileUploadDropzone"] svg {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_data_from_source(file_bytes: bytes | None) -> pd.DataFrame:
    """
    If file_bytes is None, load the default template from disk.
    Otherwise, load from the uploaded Excel bytes and validate columns.
    """

    if file_bytes is None:
        # default path
        return load_inputs("data/p4p_template.xlsx")

    return load_inputs(io.BytesIO(file_bytes))


def make_download_excel(df: pd.DataFrame) -> bytes:
    """
    Return Excel bytes for df, restricted to the template columns that actually exist.
    """

    output_df = df.copy()
    cols = [c for c in TEMPLATE_COLUMNS if c in output_df.columns]
    output_df = output_df[cols]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="P4P")
    buffer.seek(0)
    return buffer.getvalue()
def load_template_df() -> pd.DataFrame:
    base = load_inputs()
    base = ensure_final_month_live(base)
    base = apply_dc_live_locks(base)
    return base


def build_reset_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleared calendar pivot that respects locked DCs and final-month rules."""

    reset_df = df.copy()
    reset_df[LIVE] = "No"
    reset_df = ensure_final_month_live(reset_df)
    reset_df = apply_dc_live_locks(reset_df)
    reset_pivot, _ = build_calendar_pivot(reset_df)
    return reset_pivot


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


def apply_schedule_rules(
    pivot_df: pd.DataFrame, month_columns: list[str], template_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply forward-fill, final-month, and DC lock rules to pivoted data.

    Returns both the corrected pivot table (with the same column ordering as the
    incoming pivot) and the reconstructed long-form DataFrame for savings
    calculations.
    """

    enforced_pivot = enforce_forward_month_selection(pivot_df, month_columns)
    updated_df = rebuild_dataset_from_pivot(enforced_pivot, month_columns, template_df)
    constrained_df = ensure_final_month_live(
        updated_df, only_if_currently_live=True
    )
    constrained_df = apply_dc_live_locks(constrained_df)

    constrained_pivot, _ = build_calendar_pivot(constrained_df)
    constrained_pivot = constrained_pivot.reindex(
        columns=[REGION, DC_NUMBER_NAME] + month_columns, fill_value=False
    )

    return constrained_pivot, constrained_df


def rebuild_dataset_from_pivot(
    pivot_df: pd.DataFrame, month_columns: list[str], template_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge the pivoted calendar back into the base dataset."""

    edited_long = pivot_df.melt(
        id_vars=[REGION, DC_NUMBER_NAME],
        value_vars=month_columns,
        var_name="MonthStr",
        value_name="IsLiveBool",
    )

    edited_long[MONTH] = align_month_dtype(edited_long["MonthStr"], template_df[MONTH])
    edited_long = edited_long.drop(columns=["MonthStr"])

    updated_df = template_df.copy()
    updated_df[LIVE] = "No"

    merged = updated_df.merge(
        edited_long,
        how="left",
        on=[REGION, DC_NUMBER_NAME, MONTH],
    )

    updated_df[LIVE] = merged["IsLiveBool"].fillna(False).map({True: "Yes", False: "No"})
    return updated_df


def calculate_manual_results(
    pivot_df: pd.DataFrame, month_columns: list[str], template_df: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Calculate savings from a pivot table of manual selections."""

    _, constrained_df = apply_schedule_rules(pivot_df, month_columns, template_df)
    return calculate_scenario_savings(constrained_df)


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
    inject_compact_uploader_css()

    if "template_filename" not in st.session_state:
        st.session_state["template_filename"] = "DEFAULT"

    st.title("P4P Savings Simulator")

    st.markdown(
        "Interactively explore manual and optimized activation schedules to reach your target FY26 savings.",
        help=None,
    )

    # Load the base dataset that describes each DC, its region, the month value,
    # and whether it is currently live. The dataset is cached by Streamlit to
    # avoid reloading on every interaction.
    active_upload = st.session_state.get("uploaded_file")
    if active_upload is not None:
        df = _load_data_from_source(active_upload.getvalue())
    else:
        st.session_state["template_filename"] = "DEFAULT"
        df = _load_data_from_source(None)

    col_target, col_max_initial = st.columns([2, 2])

    with col_target:
        target_savings = st.number_input(
            "Target FY26 savings ($):",
            min_value=0,
            step=100000,
            value=5_000_000,
            format="%i",
        )

    with col_max_initial:
        max_initial_golives_per_month = st.number_input(
            "Max DCs going live in their first month:",
            min_value=1,
            max_value=50,
            step=1,
            value=4,
        )

    # Build a pivoted version of the dataset that Streamlit can display as a
    # calendar grid. ``base_pivot_reset`` holds the table, and ``month_columns``
    # lists the month headers for later use when configuring widgets.
    base_df = df.copy()

    base_pivot_reset, month_columns = build_calendar_pivot(base_df)

    if "manual_pivot_data" not in st.session_state:
        st.session_state["manual_pivot_data"] = base_pivot_reset
    if "optimization_calendar" not in st.session_state:
        st.session_state["optimization_calendar"] = base_pivot_reset
    if "latest_manual_result_df" not in st.session_state:
        st.session_state["latest_manual_result_df"] = None
    if "latest_template_df" not in st.session_state:
        st.session_state["latest_template_df"] = df

    # Configure how each column appears inside the data editor. Region and DC
    # identifiers are read-only text, while the month columns become checkbox
    # columns to let users toggle live months.
    column_config = {
        REGION: st.column_config.TextColumn(label=REGION, disabled=True, width="small"),
        DC_NUMBER_NAME: st.column_config.TextColumn(
            label=DC_NUMBER_NAME, disabled=True, width="medium"
        ),
    }
    column_config.update(
        {month: st.column_config.CheckboxColumn(label=month, width="small") for month in month_columns}
    )

    column_order = [REGION, DC_NUMBER_NAME] + month_columns

    params = {
        "target_savings": target_savings,
        "max_initial_golives_per_month": max_initial_golives_per_month,
    }

    manual_tab, optimizations_tab, tab_heatmap, data_tab = st.tabs(
        ["Manual selection", "Optimizations", "Value Heatmap", "Data Import/Export"]
    )

    with manual_tab:
        rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun")

        constrained_pivot, constrained_df = apply_schedule_rules(
            st.session_state["manual_pivot_data"], month_columns, df
        )
        if not constrained_pivot.equals(st.session_state["manual_pivot_data"]):
            st.session_state["manual_pivot_data"] = constrained_pivot
            rerun()

        manual_result_df, top_manual_total = calculate_scenario_savings(constrained_df)
        # Show current manual savings near the top of the tab for quick reference.
        st.markdown(
            f"**Current manual selection total savings: ${top_manual_total:,.0f}**"
        )
        if target_savings > 0:
            if top_manual_total >= target_savings:
                st.info(
                    f"Target reached! You exceed the target by ${top_manual_total - target_savings:,.0f}."
                )
            else:
                st.warning(
                    f"Still short of target by ${target_savings - top_manual_total:,.0f}. "
                    "Consider adjusting selections or running optimizations."
                )

        header_col, reset_col = st.columns([6, 1])
        header_col.subheader("DC Go-Live Calendar")
        if reset_col.button("Reset", key="manual_calendar_reset"):
            st.session_state["manual_pivot_data"] = build_reset_calendar(df)
            rerun()

        edited_pivot = st.data_editor(
            constrained_pivot,
            use_container_width=True,
            num_rows="fixed",
            column_config=column_config,
            column_order=column_order,
            height=calendar_height(len(st.session_state["manual_pivot_data"])),
            key="manual_pivot_editor",
        )

        corrected_pivot, corrected_df = apply_schedule_rules(
            edited_pivot, month_columns, df
        )
        if not corrected_pivot.equals(st.session_state["manual_pivot_data"]):
            st.session_state["manual_pivot_data"] = corrected_pivot
            rerun()

        manual_result_df, _ = calculate_scenario_savings(corrected_df)

        st.session_state["latest_manual_result_df"] = manual_result_df
        st.session_state["latest_template_df"] = corrected_df

        st.subheader("UPH Plan Output Format")
        st.dataframe(manual_result_df, use_container_width=True)

        excel_bytes = make_download_excel(manual_result_df)
        st.download_button(
            label="Download updated template (manual scenario)",
            data=excel_bytes,
            file_name="p4p_manual_scenario.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    latest_manual_result_df = st.session_state.get("latest_manual_result_df")
    template_export_df = st.session_state.get("latest_template_df", df)

    download_df = (
        latest_manual_result_df if latest_manual_result_df is not None else template_export_df
    )
    scenario_bytes = make_download_excel(download_df)
    st.session_state["scenario_bytes"] = scenario_bytes

    with tab_heatmap:
        value_df = model.calculate_incremental_value_matrix(df, params)
        index_name = value_df.index.name or "building"

        value_long = (
            value_df.reset_index()
            .rename(columns={index_name: "building"})
            .melt(id_vars="building", var_name="month", value_name="value")
        )

        building_order = value_df.index.tolist()
        month_order = value_df.columns.tolist()
        value_long["building"] = pd.Categorical(
            value_long["building"], categories=building_order, ordered=True
        )
        value_long["month"] = pd.Categorical(
            value_long["month"], categories=month_order, ordered=True
        )

        heatmap = (
            alt.Chart(value_long)
            .mark_rect()
            .encode(
                x=alt.X("month:N", title="Month", sort=month_order),
                y=alt.Y("building:N", title="Building", sort=building_order),
                color=alt.Color("value:Q", title="Incremental FY26 savings"),
                tooltip=[
                    alt.Tooltip("building:N", title="Building"),
                    alt.Tooltip("month:N", title="Month"),
                    alt.Tooltip("value:Q", title="Value ($)", format=",.0f"),
                ],
            )
            .properties(height=500)
        )

        st.altair_chart(heatmap, use_container_width=True)

    with data_tab:
        st.subheader("Data import and export")

        status_col, download_col = st.columns([3, 2])

        with status_col:
            filename = st.session_state["template_filename"]
            if filename == "DEFAULT":
                label = "Using default template"
            else:
                label = f"Using custom template: {filename}"

            st.markdown(
                f"""
                <div style="
                    display:inline-block;
                    padding:4px 10px;
                    border-radius:12px;
                    background:#e8f0fe;
                    color:#1a3e8a;
                    font-size:13px;
                    font-weight:600;
                ">{label}</div>
                """,
                unsafe_allow_html=True,
            )

        with download_col:
            st.download_button(
                "Download scenario",
                data=scenario_bytes,
                file_name="p4p_scenario.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        uploader = st.file_uploader(
            "Upload custom template (.xlsx)",
            type=["xlsx"],
            label_visibility="collapsed",
            key="uploader",
        )

        if uploader is not None:
            st.session_state["uploaded_file"] = uploader
            st.session_state["template_filename"] = uploader.name

        if st.button("Reset to default template"):
            st.session_state.pop("uploaded_file", None)
            st.session_state["template_filename"] = "DEFAULT"

        st.divider()

        st.download_button(
            "Download current scenario",
            data=scenario_bytes,
            file_name="p4p_scenario.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        default_template_bytes = make_download_excel(load_template_df())
        st.download_button(
            "Download default template",
            data=default_template_bytes,
            file_name="p4p_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with optimizations_tab:
        rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun")
        if target_savings <= 0:
            st.warning("Enter a positive target savings above to run optimizations.")
            return

            st.subheader("Run Optimizations")
        col1, col2, col3 = st.columns([2, 2, 2])
        run_greedy = col1.button("Run greedy by Dollar Impact", use_container_width=True)
        run_region_grouped = col2.button("Run region-grouped schedule", use_container_width=True)
        prefer_late_months = col3.toggle(
            "Prefer later go-lives",
            value=False,
            help="Prioritize later go-lives when selecting schedule candidates.",
        )

        if run_greedy:
            # Greedy algorithm prioritizes later months before turning on the
            # largest dollar impacts until the target savings is met or exceeded.
            greedy_df, _ = build_greedy_schedule(
                df,
                target_savings,
                max_initial_golives_per_month=int(max_initial_golives_per_month),
                use_late_month_bias=prefer_late_months,
                prefer_late_golives=prefer_late_months,
                late_month_bias=LATE_MONTH_BIAS_STRENGTH,
            )
            _, greedy_total = calculate_scenario_savings(greedy_df)
            st.session_state["optimization_calendar"], _ = build_calendar_pivot(greedy_df)
            st.session_state["optimization_result_df"] = greedy_df
            st.session_state["optimization_result_label"] = "Greedy schedule"
            st.session_state["optimization_result_total"] = greedy_total
            st.session_state["greedy_result_df"] = greedy_df

        if run_region_grouped:
            # Region-grouped algorithm attempts to cluster go-lives within the
            # same region to limit operational churn while prioritizing later
            # months first.
            region_df, _ = build_region_grouped_schedule(
                df,
                target_savings,
                max_initial_golives_per_month=int(max_initial_golives_per_month),
                use_late_month_bias=prefer_late_months,
                prefer_late_golives=prefer_late_months,
                late_month_bias=LATE_MONTH_BIAS_STRENGTH,
            )
            _, region_total = calculate_scenario_savings(region_df)
            st.session_state["optimization_calendar"], _ = build_calendar_pivot(region_df)
            st.session_state["optimization_result_df"] = region_df
            st.session_state["optimization_result_label"] = "Region-grouped schedule"
            st.session_state["optimization_result_total"] = region_total
            st.session_state["region_grouped_result_df"] = region_df

        optimization_total_placeholder = st.empty()

        if "optimization_calendar" in st.session_state:
            optimization_calendar, optimization_df = apply_schedule_rules(
                st.session_state["optimization_calendar"], month_columns, df
            )
            if not optimization_calendar.equals(st.session_state["optimization_calendar"]):
                st.session_state["optimization_calendar"] = optimization_calendar
            st.session_state["optimization_result_df"] = optimization_df
            _, optimization_calendar_total = calculate_scenario_savings(optimization_df)
            st.session_state["optimization_result_total"] = optimization_calendar_total

        header_col, reset_col = st.columns([6, 1])
        header_col.subheader("DC Go-Live Calendar")
        if reset_col.button("Reset", key="optimization_calendar_reset"):
            st.session_state["optimization_calendar"] = build_reset_calendar(df)
            st.session_state.pop("optimization_result_df", None)
            st.session_state.pop("optimization_result_label", None)
            st.session_state.pop("optimization_result_total", None)
            st.session_state.pop("greedy_result_df", None)
            st.session_state.pop("region_grouped_result_df", None)
            rerun()

        edited_optimization_pivot = st.data_editor(
            st.session_state["optimization_calendar"],
            use_container_width=True,
            column_config=column_config,
            column_order=column_order,
            disabled=False,
            hide_index=True,
            height=calendar_height(len(st.session_state["optimization_calendar"])),
            key="optimization_dc_month_grid",
        )

        updated_opt_pivot, updated_opt_df = apply_schedule_rules(
            edited_optimization_pivot, month_columns, df
        )
        st.session_state["optimization_result_df"] = updated_opt_df
        _, adjusted_total = calculate_scenario_savings(updated_opt_df)
        st.session_state["optimization_result_total"] = adjusted_total
        optimization_total_placeholder.markdown(
            f"**Current optimization calendar total savings: ${adjusted_total:,.0f}**"
        )

        if not updated_opt_pivot.equals(st.session_state["optimization_calendar"]):
            st.session_state["optimization_calendar"] = updated_opt_pivot
            st.session_state["optimization_result_label"] = "Adjusted optimization"
            rerun()

        if "optimization_result_df" in st.session_state:
            result_label = st.session_state.get("optimization_result_label", "Latest optimization")
            result_total = st.session_state.get("optimization_result_total")
            if result_total is not None:
                st.success(f"{result_label} total savings: ${result_total:,.0f}")

            st.subheader("UPH Plan Output Format")
            st.dataframe(st.session_state["optimization_result_df"], use_container_width=True)

            greedy_df = st.session_state.get("greedy_result_df")
            if greedy_df is not None:
                greedy_bytes = make_download_excel(greedy_df)
                st.download_button(
                    label="Download greedy scenario template",
                    data=greedy_bytes,
                    file_name="p4p_greedy_scenario.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            grouped_df = st.session_state.get("region_grouped_result_df")
            if grouped_df is not None:
                grouped_bytes = make_download_excel(grouped_df)
                st.download_button(
                    label="Download region-grouped scenario template",
                    data=grouped_bytes,
                    file_name="p4p_region_grouped_scenario.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

if __name__ == "__main__":
    main()
