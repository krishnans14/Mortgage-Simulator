import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Mortgage Simulator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
.stApp { background-color: #0e1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b27 100%);
    border: 1px solid #2a3045;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #7a8399; margin-bottom: 8px;
}
.metric-value {
    font-size: 26px; font-weight: 500; color: #e8eaf0;
    font-family: 'DM Serif Display', serif;
}
.metric-value.accent { color: #4ecca3; }
.metric-value.warn   { color: #f0a05a; }

section[data-testid="stSidebar"] {
    background-color: #0a0d14;
    border-right: 1px solid #1e2435;
}
.stSlider > div > div > div { background: #4ecca3 !important; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #4ecca3, #38b68e) !important;
    color: #0a0d14 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6em 2.5em !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE — initialise defaults once
# ═══════════════════════════════════════════════════════════════════════════
def _default_var_rates(loan_months: int) -> pd.DataFrame:
    """5 equal-ish blocks across the full loan in months."""
    segments = np.array_split(np.arange(1, loan_months + 1), 5)
    rows = []
    for i, seg in enumerate(segments):
        if len(seg):
            rows.append({
                "Month From": int(seg[0]),
                "Month To":   int(seg[-1]),
                "Variable Rate (%)": round(3.5 + i * 0.3, 2),
            })
    return pd.DataFrame(rows)

def _default_inflation(loan_months: int) -> pd.DataFrame:
    """5 equal-ish blocks across the full loan in months."""
    block = max(1, loan_months // 5)
    rows = []
    for i in range(5):
        m_from = i * block + 1
        m_to   = min((i + 1) * block, loan_months)
        if m_from > loan_months:
            break
        rows.append({
            "Month From":  m_from,
            "Month To":    m_to,
            "Inflation (%)": round(2.0 + i * 0.2, 1),
        })
    return pd.DataFrame(rows)

_LOAN_MONTHS_DEFAULT = 25 * 12
for key, val in [
    ("results",      None),
    ("run_params",   None),
    ("var_rates_df", _default_var_rates(_LOAN_MONTHS_DEFAULT)),
    ("inflation_df", _default_inflation(_LOAN_MONTHS_DEFAULT)),
    ("extra_df",     pd.DataFrame({
        "Month From":        [1,  24,  60,  120, 180],
        "Extra Payment (€)": [0.0, 0.0, 0.0, 0.0, 0.0],
    })),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs + Run button
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 Mortgage Simulator")
    st.markdown("---")

    st.markdown("### Loan Basics")
    total_loan = st.number_input(
        "Total Loan Amount (€)", min_value=10_000, max_value=10_000_000,
        value=410_000, step=5_000, format="%d")
    loan_years  = st.slider("Loan Duration (years)", min_value=1, max_value=30, value=25)
    loan_months = loan_years * 12

    st.markdown("---")
    st.markdown("### Tranche 1 — Fixed then Variable")
    t1_amount = st.number_input(
        "Amount (€) — T1", min_value=0, max_value=total_loan,
        value=min(160_000, total_loan), step=5_000, format="%d", key="t1a")
    t1_rate = st.number_input(
        "Fixed Interest Rate (%) — T1", min_value=0.0, max_value=20.0,
        value=3.75, step=0.1, format="%.2f", key="t1r")
    t1_fixed_years = st.slider(
        "Fixed Period (years) — T1", min_value=1, max_value=loan_years,
        value=min(10, loan_years), key="t1fy")
    st.caption(f"T1 switches to variable rate after month {t1_fixed_years * 12}.")

    st.markdown("---")
    st.markdown("### Tranche 2 — Fixed then Variable")
    t2_max = max(0, total_loan - t1_amount)
    t2_amount = st.number_input(
        "Amount (€) — T2", min_value=0, max_value=t2_max,
        value=min(90_000, t2_max), step=5_000, format="%d", key="t2a")
    t2_rate = st.number_input(
        "Fixed Interest Rate (%) — T2", min_value=0.0, max_value=20.0,
        value=3.27, step=0.1, format="%.2f", key="t2r")
    t2_fixed_years = st.slider(
        "Fixed Period (years) — T2", min_value=1, max_value=loan_years,
        value=min(5, loan_years), key="t2fy")
    st.caption(f"T2 switches to variable rate after month {t2_fixed_years * 12}.")

    t3_amount = max(0, total_loan - t1_amount - t2_amount)
    st.markdown("---")
    st.markdown(f"### Tranche 3 — Variable")
    st.markdown(f"**Amount: €{t3_amount:,.0f}** *(remainder)*")

    st.markdown("---")

    # Live cascade preview — updates immediately as sliders change, no Run needed
    if t1_fixed_years <= t2_fixed_years:
        _fa, _fa_m = "T1", t1_fixed_years * 12
        _fb, _fb_m = "T2", t2_fixed_years * 12
    else:
        _fa, _fa_m = "T2", t2_fixed_years * 12
        _fb, _fb_m = "T1", t1_fixed_years * 12
    _tie = " *(equal fixed periods — T1 prioritised)*" if t1_fixed_years == t2_fixed_years else ""

    st.info(
        f"**Extra payment cascade:**\n\n"
        f"T3 *(mo. 1)* → **{_fa}** *(variable from mo. {_fa_m})* → **{_fb}** *(variable from mo. {_fb_m})*{_tie}\n\n"
        f"Order is determined by which fixed tranche becomes variable first. "
        f"Updates live as you adjust the sliders above."
    )

    run_clicked = st.button("▶  RUN SIMULATION", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA — Header + Editable tables
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("# Mortgage Simulator")
st.markdown("---")

# ── Variable Rate Schedule ────────────────────────────────────────────────
st.markdown("### Variable Rate Schedule")
st.caption(
    "Defines the variable rate applied to T3 from month 1, "
    "and to T1/T2 after their respective fixed periods end. "
    "Add or delete rows freely — applied on Run."
)
edited_var = st.data_editor(
    st.session_state["var_rates_df"],
    use_container_width=True,
    num_rows="dynamic",
    key="var_rates_editor",
    hide_index=True,
    column_config={
        "Month From":         st.column_config.NumberColumn("Month From", min_value=1, step=1, format="%d"),
        "Month To":           st.column_config.NumberColumn("Month To",   min_value=1, step=1, format="%d"),
        "Variable Rate (%)":  st.column_config.NumberColumn("Rate (%)",   min_value=0.0, format="%.2f"),
    }
)

st.markdown("### Inflation Schedule")
st.caption("Cumulative inflation used to compute inflation-adjusted payments. Add or delete rows freely — applied on Run.")
edited_inf = st.data_editor(
    st.session_state["inflation_df"],
    use_container_width=True,
    num_rows="dynamic",
    key="inflation_editor",
    hide_index=True,
    column_config={
        "Month From":    st.column_config.NumberColumn("Month From",   min_value=1, step=1, format="%d"),
        "Month To":      st.column_config.NumberColumn("Month To",     min_value=1, step=1, format="%d"),
        "Inflation (%)": st.column_config.NumberColumn("Inflation (%)", min_value=0.0, format="%.2f"),
    }
)

st.markdown("### Extra Payments")
st.caption(
    "Each row sets a recurring monthly extra principal payment from 'Month From' until "
    "the next row's month (or end of loan). "
    "The cascade order (T3 → shorter-fixed tranche → longer-fixed tranche) is "
    "determined automatically at run time and shown above the results."
)
edited_extra = st.data_editor(
    st.session_state["extra_df"],
    use_container_width=True,
    num_rows="dynamic",
    key="extra_editor",
    hide_index=True,
    column_config={
        "Month From":        st.column_config.NumberColumn("Month From",   min_value=1, step=1, format="%d"),
        "Extra Payment (€)": st.column_config.NumberColumn("Extra/mo (€)", min_value=0.0, format="%.0f"),
    }
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def get_rate_for_month(month: int, df_rate: pd.DataFrame, col: str) -> float:
    """Look up an annual rate (%) for a given month from a month-based schedule,
    returning it as a monthly decimal rate (annual% / 100 / 12)."""
    for _, row in df_rate.iterrows():
        try:
            if float(row["Month From"]) <= month <= float(row["Month To"]):
                return float(row[col]) / 100.0 / 12.0
        except Exception:
            pass
    # Fallback: use last row
    try:
        return float(df_rate[col].iloc[-1]) / 100.0 / 12.0
    except Exception:
        return 0.0


def build_inflation_factors(loan_months: int, df_inf: pd.DataFrame) -> list:
    """Cumulative compound inflation factor for each month (1-indexed)."""
    factors, factor = [], 1.0
    for m in range(1, loan_months + 1):
        monthly_inf = get_rate_for_month(m, df_inf, "Inflation (%)")
        factor *= (1 + monthly_inf)
        factors.append(factor)
    return factors


def annuity_payment(principal: float, monthly_rate: float, n_months: int) -> float:
    if principal <= 0 or n_months <= 0:
        return 0.0
    if monthly_rate == 0:
        return principal / n_months
    return (principal * monthly_rate * (1 + monthly_rate) ** n_months
            / ((1 + monthly_rate) ** n_months - 1))


def build_extra_schedule(df_extra: pd.DataFrame, loan_months: int) -> list:
    """Return sorted list of (start_month, amount) for recurring extra payments."""
    schedule = []
    if df_extra is not None:
        for _, row in df_extra.dropna(how="all").iterrows():
            try:
                m = int(row["Month From"])
                v = float(row["Extra Payment (€)"])
                if m > 0:
                    schedule.append((m, max(v, 0.0)))
            except Exception:
                pass
    schedule.sort(key=lambda x: x[0])
    return schedule


def get_extra_for_month(month: int, schedule: list, loan_months: int) -> float:
    active = 0.0
    for i, (start, amt) in enumerate(schedule):
        if month < start:
            break
        next_start = schedule[i + 1][0] if i + 1 < len(schedule) else loan_months + 1
        if start <= month < next_start:
            active = amt
    return active


def run_engine(
    total_loan, loan_months, loan_years,
    t1_amount, t1_rate, t1_fixed_years,
    t2_amount, t2_rate, t2_fixed_years,
    t3_amount,
    df_var, df_inf, df_extra,
) -> tuple:
    """Returns (DataFrame, cascade_order) where cascade_order is a list of label strings."""

    extra_schedule = build_extra_schedule(df_extra, loan_months)
    inf_factors    = build_inflation_factors(loan_months, df_inf)

    # ── Auto-sort fixed tranches by fixed period (shorter → variable sooner → gets extra first)
    # fa = first-to-variable, fb = second-to-variable
    if t1_fixed_years <= t2_fixed_years:
        fa_label, fb_label           = "T1", "T2"
        fa_amount, fb_amount         = float(t1_amount), float(t2_amount)
        fa_rate,   fb_rate           = t1_rate, t2_rate
        fa_fixed_years, fb_fixed_years = t1_fixed_years, t2_fixed_years
    else:
        fa_label, fb_label           = "T2", "T1"
        fa_amount, fb_amount         = float(t2_amount), float(t1_amount)
        fa_rate,   fb_rate           = t2_rate, t1_rate
        fa_fixed_years, fb_fixed_years = t2_fixed_years, t1_fixed_years

    fa_fixed_months = fa_fixed_years * 12
    fb_fixed_months = fb_fixed_years * 12
    cascade_order = ["T3", fa_label, fb_label]

    bal_fa, bal_fb, bal_t3 = fa_amount, fb_amount, float(t3_amount)

    pmt_fa = annuity_payment(bal_fa, fa_rate / 100 / 12, loan_months)
    pmt_fb = annuity_payment(bal_fb, fb_rate / 100 / 12, loan_months)

    prev_rfa_var = None
    prev_rfb_var = None
    prev_r3      = get_rate_for_month(1, df_var, "Variable Rate (%)")
    pmt_t3       = annuity_payment(bal_t3, prev_r3, loan_months)

    records = []

    for month in range(1, loan_months + 1):
        rem = loan_months - month + 1

        extra_requested = get_extra_for_month(month, extra_schedule, loan_months)
        extra_t3 = extra_fa = extra_fb = 0.0

        # ── T3 — always variable ──────────────────────────────────────────
        if bal_t3 > 1e-2:
            r3 = get_rate_for_month(month, df_var, "Variable Rate (%)")
            if abs(r3 - prev_r3) > 1e-12:
                pmt_t3  = annuity_payment(bal_t3, r3, rem)
                prev_r3 = r3
            int_t3   = bal_t3 * r3
            extra_t3 = extra_requested
            prin_t3  = min(max(pmt_t3 - int_t3, 0) + extra_t3, bal_t3)
            bal_t3   = max(bal_t3 - prin_t3, 0)
            if extra_t3 > 0 and bal_t3 > 1e-2 and (loan_months - month) > 0:
                pmt_t3  = annuity_payment(bal_t3, r3, loan_months - month)
                prev_r3 = r3
        else:
            r3 = get_rate_for_month(month, df_var, "Variable Rate (%)")
            int_t3 = prin_t3 = 0.0
            if extra_requested > 0 and month > fa_fixed_months:
                extra_fa = extra_requested

        # ── fa tranche — fixed then variable (first to become variable) ───
        if bal_fa > 1e-2:
            if month <= fa_fixed_months:
                rfa = fa_rate / 100 / 12
            else:
                rfa = get_rate_for_month(month, df_var, "Variable Rate (%)")
                if prev_rfa_var is None or abs(rfa - prev_rfa_var) > 1e-12:
                    pmt_fa       = annuity_payment(bal_fa, rfa, rem)
                    prev_rfa_var = rfa
            int_fa  = bal_fa * rfa
            prin_fa = min(max(pmt_fa - int_fa, 0) + extra_fa, bal_fa)
            bal_fa  = max(bal_fa - prin_fa, 0)
            if extra_fa > 0 and bal_fa > 1e-2 and (loan_months - month) > 0:
                pmt_fa       = annuity_payment(bal_fa, rfa, loan_months - month)
                prev_rfa_var = rfa
        else:
            rfa = fa_rate / 100 / 12 if month <= fa_fixed_months \
                  else get_rate_for_month(month, df_var, "Variable Rate (%)")
            int_fa = prin_fa = 0.0
            spill = extra_fa if extra_fa > 0 else (
                extra_requested if bal_t3 <= 1e-2 and month > fa_fixed_months else 0.0
            )
            if spill > 0 and month > fb_fixed_months:
                extra_fb = spill
            extra_fa = 0.0

        # ── fb tranche — fixed then variable (second to become variable) ──
        if bal_fb > 1e-2:
            if month <= fb_fixed_months:
                rfb = fb_rate / 100 / 12
            else:
                rfb = get_rate_for_month(month, df_var, "Variable Rate (%)")
                if prev_rfb_var is None or abs(rfb - prev_rfb_var) > 1e-12:
                    pmt_fb       = annuity_payment(bal_fb, rfb, rem)
                    prev_rfb_var = rfb
            int_fb  = bal_fb * rfb
            prin_fb = min(max(pmt_fb - int_fb, 0) + extra_fb, bal_fb)
            bal_fb  = max(bal_fb - prin_fb, 0)
            if extra_fb > 0 and bal_fb > 1e-2 and (loan_months - month) > 0:
                pmt_fb       = annuity_payment(bal_fb, rfb, loan_months - month)
                prev_rfb_var = rfb
        else:
            rfb = fb_rate / 100 / 12 if month <= fb_fixed_months \
                  else get_rate_for_month(month, df_var, "Variable Rate (%)")
            int_fb = prin_fb = extra_fb = 0.0

        # ── Map fa/fb back to T1/T2 labels ───────────────────────────────
        if fa_label == "T1":
            r1, int_t1, prin_t1, extra_t1, bal_t1 = rfa, int_fa, prin_fa, extra_fa, bal_fa
            r2, int_t2, prin_t2, extra_t2, bal_t2 = rfb, int_fb, prin_fb, extra_fb, bal_fb
        else:
            r1, int_t1, prin_t1, extra_t1, bal_t1 = rfb, int_fb, prin_fb, extra_fb, bal_fb
            r2, int_t2, prin_t2, extra_t2, bal_t2 = rfa, int_fa, prin_fa, extra_fa, bal_fa

        # ── Totals ────────────────────────────────────────────────────────
        reg_prin_t1 = prin_t1 - extra_t1
        reg_prin_t2 = prin_t2 - extra_t2
        reg_prin_t3 = prin_t3 - extra_t3
        total_extra = extra_t1 + extra_t2 + extra_t3
        total_pay   = (prin_t1 + int_t1) + (prin_t2 + int_t2) + (prin_t3 + int_t3)
        real_pay    = total_pay / inf_factors[month - 1]

        records.append({
            "Month": month,
            "Year":  (month - 1) // 12 + 1,
            "T1 Rate (%)":          round(r1 * 12 * 100, 4) if bal_t1 > 1e-2 or prin_t1 > 0 else 0.0,
            "T1 Regular Principal": reg_prin_t1,
            "T1 Extra":             extra_t1,
            "T1 Principal":         prin_t1,
            "T1 Interest":          int_t1,
            "T1 Balance":           bal_t1,
            "T2 Rate (%)":          round(r2 * 12 * 100, 4) if bal_t2 > 1e-2 or prin_t2 > 0 else 0.0,
            "T2 Regular Principal": reg_prin_t2,
            "T2 Extra":             extra_t2,
            "T2 Principal":         prin_t2,
            "T2 Interest":          int_t2,
            "T2 Balance":           bal_t2,
            "T3 Rate (%)":          round(r3 * 12 * 100, 4),
            "T3 Regular Principal": reg_prin_t3,
            "T3 Extra":             extra_t3,
            "T3 Principal":         prin_t3,
            "T3 Interest":          int_t3,
            "T3 Balance":           bal_t3,
            "Total Extra":          total_extra,
            "Total Payment":        total_pay,
            "Total Principal":      prin_t1 + prin_t2 + prin_t3,
            "Total Interest":       int_t1  + int_t2  + int_t3,
            "Real Payment":         real_pay,
            "Inflation Factor":     inf_factors[month - 1],
        })

    return pd.DataFrame(records), cascade_order


# ═══════════════════════════════════════════════════════════════════════════
# RUN BUTTON HANDLER
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    st.session_state["var_rates_df"] = edited_var.dropna(how="all").reset_index(drop=True)
    st.session_state["inflation_df"] = edited_inf.dropna(how="all").reset_index(drop=True)
    st.session_state["extra_df"]     = edited_extra.dropna(how="all").reset_index(drop=True)

    st.session_state["run_params"] = dict(
        loan_months=loan_months,
        loan_years=loan_years,
        t1_fixed_months=t1_fixed_years * 12,
        t2_fixed_months=t2_fixed_years * 12,
    )

    with st.spinner("Running simulation…"):
        df_result, cascade_order = run_engine(
            total_loan, loan_months, loan_years,
            t1_amount, t1_rate, t1_fixed_years,
            t2_amount, t2_rate, t2_fixed_years,
            t3_amount,
            st.session_state["var_rates_df"],
            st.session_state["inflation_df"],
            st.session_state["extra_df"],
        )
        st.session_state["results"]       = df_result
        st.session_state["run_params"]["cascade_order"] = cascade_order

# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state["results"] is None:
    st.info("👈  Configure your loan in the sidebar, then press **▶ RUN SIMULATION** to see results.")
    st.stop()

df = st.session_state["results"]
p  = st.session_state["run_params"]

# ── Summary metrics ──────────────────────────────────────────────────────
total_nominal  = df["Total Payment"].sum()   # already includes extra principal
total_interest = df["Total Interest"].sum()
total_real     = df["Real Payment"].sum()
inf_saving     = total_nominal - total_real
inf_adj_interest_paid     = total_real - total_loan


mc1, mc2, mc3, mc4 = st.columns(4)
for col, label, value, cls in [
    (mc1, "Total Nominal Cost",        f"€{total_nominal:,.0f}",  ""),
    (mc2, "Nominal Interest Paid",     f"€{total_interest:,.0f}", "warn"),
    (mc3, "Inflation-Adj. Total Cost", f"€{total_real:,.0f}",     "accent"),
    # (mc4, 'Inflation "Savings"',       f"€{inf_saving:,.0f}",     "accent"),
    (mc4, 'Inflation-Adj. Interest Paid',       f"€{inf_adj_interest_paid:,.0f}",     "accent"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



# ── Chart theme ──────────────────────────────────────────────────────────
C = dict(
    t1="#4ecca3", t2="#f0a05a", t3="#a78bfa",
    extra_t3="#f0e05a", extra_t1="#ff7eb6", extra_t2="#60c9f8",
    real="#f05ab4",
    bg="#0e1117", grid="#1e2435", text="#8892a4",
)
LB = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(family="DM Mono, monospace", color=C["text"], size=11),
    xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
    yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=50, r=20, t=50, b=40),
)

# ── Chart 1 — Tranche Balances ──────────────────────────────────────────
fig_bal = go.Figure()
for col_name, color, name, fill in [
    ("T1 Balance", C["t1"], "T1 (Fixed → Variable)", "rgba(78,204,163,0.08)"),
    ("T2 Balance", C["t2"], "T2 (Fixed → Variable)", "rgba(240,160,90,0.08)"),
    ("T3 Balance", C["t3"], "T3 (Variable)",          "rgba(167,139,250,0.08)"),
]:
    fig_bal.add_trace(go.Scatter(
        x=df["Month"], y=df[col_name], name=name,
        line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill,
    ))
for x, color, label in [
    (p["t1_fixed_months"], C["t1"], "T1 → Variable"),
    (p["t2_fixed_months"], C["t2"], "T2 → Variable"),
]:
    if 0 < x < p["loan_months"]:
        fig_bal.add_vline(x=x, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_font_color=color,
                           annotation_position="top right")
fig_bal.update_layout(**LB, title="Tranche Balances Over Time",
                       xaxis_title="Month", yaxis_title="Outstanding Balance (€)")

# ── Chart 2 — Monthly Payment Breakdown ─────────────────────────────────
fig_pay = go.Figure()
# bar_layers = [
#     ("T1 Interest",          "rgba(78,204,163,0.95)"),
#     ("T1 Regular Principal", "rgba(78,204,163,0.40)"),
#     ("T1 Extra",             C["extra_t1"]),
#     ("T2 Interest",          "rgba(240,160,90,0.95)"),
#     ("T2 Regular Principal", "rgba(240,160,90,0.40)"),
#     ("T2 Extra",             C["extra_t2"]),
#     ("T3 Interest",          "rgba(167,139,250,0.95)"),
#     ("T3 Regular Principal", "rgba(167,139,250,0.40)"),
#     ("T3 Extra",             C["extra_t3"]),
# ]
bar_layers = [
    ("T1 Interest",          "rgba(78,204,163,0.95)"),
    ("T1 Regular Principal", "rgba(78,204,163,0.70)"),
    ("T1 Extra",             "rgba(78,204,163,0.35)"),
    ("T2 Interest",          "rgba(240,160,90,0.95)"),
    ("T2 Regular Principal", "rgba(240,160,90,0.70)"),
    ("T2 Extra",             "rgba(240,160,90,0.35)"),
    ("T3 Interest",          "rgba(167,139,250,0.95)"),
    ("T3 Regular Principal", "rgba(167,139,250,0.70)"),
    ("T3 Extra",             "rgba(167,139,250,0.35)"),
]
for col_name, color in bar_layers:
    fig_pay.add_trace(go.Bar(
        x=df["Month"], y=df[col_name], name=col_name,
        marker_color=color, offsetgroup=0,
    ))
fig_pay.add_trace(go.Scatter(
    x=df["Month"], y=df["Real Payment"],
    name="Real (Inflation-Adj.) Total",
    line=dict(color=C["real"], width=1.5, dash="dot"), mode="lines",
))
fig_pay.update_layout(**LB, title="Monthly Payment Breakdown", barmode="stack",
                       xaxis_title="Month", yaxis_title="Payment (€)")

tab1, tab2 = st.tabs(["📊 Tranche Balances", "💳 Monthly Payments"])
with tab1:
    st.plotly_chart(fig_bal, use_container_width=True)
with tab2:
    st.plotly_chart(fig_pay, use_container_width=True)

# ── Full Amortisation Table ──────────────────────────────────────────────
with st.expander("📋 Full Amortisation Schedule", expanded=False):
    display_cols = [
        "Month", "Year",
        "T1 Rate (%)", "T1 Regular Principal", "T1 Extra", "T1 Principal", "T1 Interest", "T1 Balance",
        "T2 Rate (%)", "T2 Regular Principal", "T2 Extra", "T2 Principal", "T2 Interest", "T2 Balance",
        "T3 Rate (%)", "T3 Regular Principal", "T3 Extra", "T3 Principal", "T3 Interest", "T3 Balance",
        "Total Extra", "Total Payment", "Total Principal", "Total Interest",
        "Real Payment", "Inflation Factor",
    ]
    df_disp = df[display_cols].copy()
    int_cols   = {"Month", "Year"}
    rate_cols  = {"T1 Rate (%)", "T2 Rate (%)", "T3 Rate (%)"}
    for c in display_cols:
        if c in int_cols:
            continue
        elif c in rate_cols:
            df_disp[c] = df_disp[c].map(lambda x: f"{x:.4f}")
        elif c == "Inflation Factor":
            df_disp[c] = df_disp[c].map(lambda x: f"{x:.6f}")
        else:
            df_disp[c] = df_disp[c].map(lambda x: f"{x:,.2f}")
    st.dataframe(df_disp, use_container_width=True, height=420)
    csv = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv,
                        file_name="mortgage_schedule.csv", mime="text/csv")
