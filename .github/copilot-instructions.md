# Copilot Instructions for p4p-savings-simulator

## Project Overview
- **Purpose:** Simulate and optimize go-live schedules for Distribution Centers (DCs) to model P4P (Pay for Performance) savings scenarios for FY26.
- **UI:** Streamlit app (`app.py`) with two main tabs:
  - **Manual selection:** Toggle live months for each DC.
  - **Optimizations:** Run algorithms to propose schedules that meet target savings.
- **Core logic:** All business/data logic is in `model.py`.

## Key Files & Structure
- `app.py`: Streamlit UI, user interaction, and scenario orchestration.
- `model.py`: Data model, constants, schedule/savings logic, and optimization algorithms.
- `data/p4p_template.xlsx`: Input template for DC/month/savings data.
- `requirements.txt`: Python dependencies (pandas, openpyxl, streamlit).

## Data Flow
- User uploads or uses default Excel template (`data/p4p_template.xlsx`).
- Data is loaded and validated via `load_inputs` in `model.py`.
- UI allows manual or optimized schedule selection.
- Savings calculations and schedule logic are handled by functions in `model.py`.

## Developer Workflows
- **Run app locally:**
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
- **Edit model logic:** Update or add functions in `model.py`. All business rules and optimizations live here.
- **Add new UI features:** Modify `app.py` using Streamlit patterns. Keep UI logic separate from model logic.

## Project Conventions
- **Constants:** All column names and key business constants are defined at the top of `model.py`.
- **Month ordering:** Custom `MONTH_ORDER` treats January as the last month (after December).
- **DC live locks:** Some DCs have fixed go-live windows (`DC_LIVE_LOCKS` in `model.py`).
- **Data validation:** Use `REQUIRED_COLUMNS` and `REQUIRED_INPUT_COLUMNS` for input checks.
- **Display:** DCs are shown as `DC Number Name` (concatenated ID and name).
- **Extensive comments:** Both `app.py` and `model.py` are heavily commented for clarity.

## Integration & Extensibility
- **No external APIs:** All logic is local; no network calls.
- **Add new optimization:** Implement in `model.py` and expose via a new tab or button in `app.py`.
- **Template changes:** Update `data/p4p_template.xlsx` and adjust `REQUIRED_COLUMNS` as needed.

## Examples
- To add a new optimization algorithm, define it in `model.py` and call it from the "Optimizations" tab in `app.py`.
- To change the DC/month structure, update the template and constants in `model.py`.

---

For questions, review comments in `app.py` and `model.py` for intent and usage patterns.
