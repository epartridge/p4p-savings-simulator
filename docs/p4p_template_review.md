# P4P Template Review (2025-02 Upload)

## Workbook structure
- **File:** `data/p4p_template.xlsx`
- **Sheets:** `DC Rollout Template`, `DC Rollout Template OLD`.
- The new sheet has **229 rows x 17 columns**, while the archived sheet has **228 rows x 12 columns**.

## New `DC Rollout Template` columns
- Region, DC ID, DC Name, Month, Ramp, Baseline Hrs, Hours Saved, Dollar Impact, Dollar Impact w/ 32.73% Fringe, Live?, column 10 (blank header), **Plan A - High Certainty (Houston /Winchester 1H)**, columns 12–16 (blank headers). The simulator now treats **Dollar Impact w/ 32.73% Fringe** as the canonical value while retaining the "Dollar Impact" label in the UI for continuity.
- Months are provided as six-digit values such as `202601`–`202612`.
- The `Ramp` field contains small decimal multipliers (e.g., `0.016336`, `0.022466`).
- Several summary/annotation cells appear in the trailing blank columns, including **"Total Dollar Impact (Go Live, w/ Payout)"** and corresponding numbers (e.g., 256,997.469734; 179,898.228814).

## Archived `DC Rollout Template OLD` columns
- Region, DC ID, DC Name, Month, % Commitment, Dollar Impact, CPH Impact, Live?, column 8 (blank header), Target Dollar Amount, column labeled `1000000`, column 11 (blank header).

## Clarifying questions
1. **Month format:** Should the simulator interpret `Month` as a YYYYMM integer or convert it to calendar dates? The new sheet uses `202601`-style values instead of month names.
2. **Ramp interpretation:** How should the decimal `Ramp` values be applied (e.g., multiplier on Dollar Impact, Hours Saved, or deployment pacing)?
3. ~~**Fringe calculation:** Should `Dollar Impact w/ 32.73% Fringe` be derived from `Dollar Impact` using a fixed 32.73% uplift, or does it come from a different assumption source?~~ **Resolved:** All calculations now consume `Dollar Impact w/ 32.73% Fringe`, exposing it to users under the legacy "Dollar Impact" label.
4. **Blank/auxiliary columns:** Columns 10 and 12–16 are unlabeled but contain summary text and totals in some rows. Should these be ignored, or do they drive payout/rollout logic?
5. **Plan A flag:** How should `Plan A - High Certainty (Houston /Winchester 1H)` be used in the rollout logic, and does it apply only to specific DCs (Houston/Winchester) or more generally?
6. **Total Dollar Impact rows:** The new sheet embeds total rows within the dataset. Should the simulator treat them as metadata (skip when loading) or display/report them alongside monthly rows?
7. **Fringe/totals vs. old target:** The old sheet included a `Target Dollar Amount` column and a `1000000` header column for comparison. Are these targets still required somewhere else, or replaced entirely by the new payout totals?
