import sqlite3
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Health Claims BI", page_icon="📊", layout="wide")

TEAL = "#0fa3a3"
INK = "#0d1b2a"
OCHRE = "#e0902b"


# ---------------------------------------------------------------- data
@st.cache_resource
def build_db() -> sqlite3.Connection:
    """Synthetic health-insurance claims (fictional payers and providers)."""
    rng = np.random.default_rng(7)
    n = 60_000
    start = date(2025, 1, 1)
    days = (date(2026, 10, 31) - start).days

    payers = ["Andes Salud", "Plata Health", "Delta Prepaga", "Norte Medical", "Sur Seguros"]
    payer_w = [0.30, 0.25, 0.20, 0.15, 0.10]
    categories = ["Consultation", "Lab test", "Imaging", "Surgery", "Physical therapy", "Pharmacy"]
    cat_w = [0.32, 0.24, 0.14, 0.05, 0.10, 0.15]
    base_amount = {"Consultation": 45, "Lab test": 30, "Imaging": 160, "Surgery": 2400,
                   "Physical therapy": 38, "Pharmacy": 70}
    providers = [f"PRV-{i:03d}" for i in range(1, 61)]
    prov_risk = rng.beta(2, 30, len(providers))  # baseline rejection propensity
    prov_risk[[7, 23, 41]] += 0.12               # three providers with a real problem
    reasons = ["Missing authorization", "Member not eligible", "Duplicate claim",
               "Invalid procedure code", "Amount above contract"]
    reason_w = [0.34, 0.22, 0.16, 0.18, 0.10]

    d = rng.integers(0, days, n)
    # seasonal volume: more claims mid-year, fewer in January
    keep = rng.random(n) < (0.75 + 0.25 * np.sin((d % 365) / 365 * 2 * np.pi - 1.2))
    d = d[keep]
    m = len(d)
    prov_idx = rng.integers(0, len(providers), m)
    cat = rng.choice(categories, m, p=cat_w)
    payer = rng.choice(payers, m, p=payer_w)
    amount = np.round(np.array([base_amount[c] for c in cat]) * rng.lognormal(0, 0.35, m), 2)

    # PRV-024 starts failing authorizations in Aug 2026 (the anomaly to find)
    claim_date = np.array([start + timedelta(days=int(x)) for x in d])
    risk = prov_risk[prov_idx].copy()
    spike = (prov_idx == 23) & (claim_date >= date(2026, 8, 1))
    risk[spike] += 0.25
    rejected = rng.random(m) < risk
    reason = np.where(rejected, rng.choice(reasons, m, p=reason_w), None)
    reason[spike & rejected] = "Missing authorization"
    proc_days = np.clip(rng.gamma(2.2, 2.0, m) + rejected * 3, 0.5, None).round(1)

    df = pd.DataFrame({
        "claim_id": np.arange(1, m + 1),
        "claim_date": pd.to_datetime(claim_date).strftime("%Y-%m-%d"),
        "payer": payer,
        "provider": np.array(providers)[prov_idx],
        "category": cat,
        "amount": amount,
        "status": np.where(rejected, "Rejected", "Approved"),
        "reason": reason,
        "processing_days": proc_days,
        "member_id": rng.integers(100000, 999999, m).astype(str),
    })
    # a few data-quality problems, as in any real feed
    bad = rng.choice(m, 40, replace=False)
    df.loc[bad[:15], "member_id"] = None
    df.loc[bad[15:25], "amount"] = -df.loc[bad[15:25], "amount"]
    dup = df.sample(25, random_state=3).copy()
    df = pd.concat([df, dup], ignore_index=True)

    con = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("claims", con, index=False)
    con.execute("CREATE INDEX ix_date ON claims(claim_date)")
    return con


con = build_db()


def q(sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


# ---------------------------------------------------------------- filters
bounds = q("SELECT MIN(claim_date) a, MAX(claim_date) b FROM claims").iloc[0]
min_d, max_d = pd.to_datetime(bounds.a).date(), pd.to_datetime(bounds.b).date()
payers = q("SELECT DISTINCT payer FROM claims ORDER BY payer").payer.tolist()
cats = q("SELECT DISTINCT category FROM claims ORDER BY category").category.tolist()

with st.sidebar:
    st.header("Filters")
    rng_sel = st.date_input("Period", (date(2026, 5, 1), max_d), min_value=min_d, max_value=max_d)
    sel_payers = st.multiselect("Payer", payers, default=payers)
    sel_cats = st.multiselect("Category", cats, default=cats)
    st.caption("Synthetic data: fictional payers and providers, built to look like a real claims feed.")

if not isinstance(rng_sel, tuple) or len(rng_sel) != 2 or not sel_payers or not sel_cats:
    st.info("Pick a full period, at least one payer and one category.")
    st.stop()

d0, d1 = rng_sel
span = (d1 - d0).days + 1
p0, p1 = d0 - timedelta(days=span), d0 - timedelta(days=1)

ph_p = ",".join("?" * len(sel_payers))
ph_c = ",".join("?" * len(sel_cats))
FILTER = f"payer IN ({ph_p}) AND category IN ({ph_c}) AND amount > 0"
fparams = tuple(sel_payers) + tuple(sel_cats)

KPI_SQL = f"""
SELECT COUNT(*)                                             AS claims,
       ROUND(SUM(amount), 0)                                AS billed,
       ROUND(100.0 * AVG(status = 'Rejected'), 2)           AS rejection_rate,
       ROUND(AVG(processing_days), 1)                       AS avg_days
FROM claims
WHERE claim_date BETWEEN ? AND ? AND {FILTER}
"""
cur = q(KPI_SQL, (str(d0), str(d1)) + fparams).iloc[0]
prev = q(KPI_SQL, (str(p0), str(p1)) + fparams).iloc[0]

# ---------------------------------------------------------------- header + KPIs
st.title("Health Claims Dashboard")
st.caption(f"{d0:%b %d, %Y} – {d1:%b %d, %Y} · compared with the previous {span} days")

def delta(a, b, pct=True):
    if b in (None, 0) or pd.isna(b) or abs(a - b) < 1e-9:
        return None
    return f"{(a - b) / b * 100:+.1f}%" if pct else f"{a - b:+.2f} pp"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Claims", f"{int(cur.claims):,}", delta(cur.claims, prev.claims))
k2.metric("Billed amount", f"${cur.billed:,.0f}", delta(cur.billed, prev.billed))
k3.metric("Rejection rate", f"{cur.rejection_rate:.2f}%", delta(cur.rejection_rate, prev.rejection_rate, False),
          delta_color="inverse")
k4.metric("Avg. processing days", f"{cur.avg_days:.1f}", delta(cur.avg_days, prev.avg_days), delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Providers", "Data quality", "SQL"])

# ---------------------------------------------------------------- trends
TREND_SQL = f"""
SELECT substr(claim_date, 1, 7)                             AS month,
       COUNT(*)                                             AS claims,
       ROUND(100.0 * AVG(status = 'Rejected'), 2)           AS rejection_rate
FROM claims
WHERE {FILTER}
GROUP BY month ORDER BY month
"""
REASON_SQL = f"""
SELECT reason, COUNT(*) AS rejected
FROM claims
WHERE status = 'Rejected' AND claim_date BETWEEN ? AND ? AND {FILTER}
GROUP BY reason ORDER BY rejected DESC
"""
with tab1:
    tr = q(TREND_SQL, fparams)
    tr["month"] = pd.to_datetime(tr["month"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Claims and rejection rate by month")
        base = alt.Chart(tr).encode(x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")))
        bars = base.mark_bar(color="#cfe9e9").encode(y=alt.Y("claims:Q", title="Claims"))
        line = base.mark_line(color=OCHRE, strokeWidth=3, point=alt.OverlayMarkDef(color=OCHRE)).encode(
            y=alt.Y("rejection_rate:Q", title="Rejection rate (%)"))
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=320),
                        width="stretch")
    with c2:
        st.subheader("Why claims are rejected")
        rs = q(REASON_SQL, (str(d0), str(d1)) + fparams)
        st.altair_chart(
            alt.Chart(rs).mark_bar(color=TEAL).encode(
                x=alt.X("rejected:Q", title="Rejected claims"),
                y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=["reason", "rejected"]).properties(height=320),
            width="stretch")

# ---------------------------------------------------------------- providers + anomaly
PROV_SQL = f"""
WITH cur AS (
  SELECT provider, COUNT(*) AS claims, AVG(status = 'Rejected') AS rate
  FROM claims WHERE claim_date BETWEEN ? AND ? AND {FILTER}
  GROUP BY provider
),
hist AS (
  SELECT provider, AVG(status = 'Rejected') AS rate
  FROM claims WHERE claim_date < ? AND {FILTER}
  GROUP BY provider
)
SELECT cur.provider,
       cur.claims,
       ROUND(100 * cur.rate, 1)              AS rejection_rate,
       ROUND(100 * hist.rate, 1)             AS historical_rate,
       ROUND(100 * (cur.rate - hist.rate), 1) AS change_pp
FROM cur JOIN hist USING (provider)
WHERE cur.claims >= 30
ORDER BY change_pp DESC
"""
with tab2:
    pv = q(PROV_SQL, (str(d0), str(d1)) + fparams + (str(d0),) + fparams)
    if pv.empty:
        st.info("Not enough history for this period. Pick a later start date.")
    else:
        mu, sd = pv.change_pp.mean(), pv.change_pp.std()
        pv["flag"] = np.where(pv.change_pp > mu + 2 * sd, "Investigate", "")
        flagged = pv[pv.flag != ""]
        st.subheader("Providers whose rejection rate jumped")
        if len(flagged):
            st.warning(
                f"{len(flagged)} provider(s) are more than 2 standard deviations above the usual change: "
                + ", ".join(f"**{r.provider}** ({r.historical_rate}% → {r.rejection_rate}%)" for r in flagged.itertuples()))
        else:
            st.success("No provider stands out in this period.")
        st.altair_chart(
            alt.Chart(pv).mark_circle(size=90).encode(
                x=alt.X("historical_rate:Q", title="Historical rejection rate (%)"),
                y=alt.Y("rejection_rate:Q", title="Rejection rate in period (%)"),
                color=alt.condition(alt.datum.flag == "Investigate", alt.value(OCHRE), alt.value(TEAL)),
                tooltip=["provider", "claims", "historical_rate", "rejection_rate", "change_pp"]
            ).properties(height=340),
            width="stretch")
        st.dataframe(pv.head(15), width="stretch", hide_index=True)

# ---------------------------------------------------------------- data quality
DQ_SQL = {
    "Duplicate claims (same member, provider, date, amount)": """
        SELECT COUNT(*) - COUNT(DISTINCT COALESCE(member_id, '?') || provider || claim_date || amount) AS n FROM claims""",
    "Missing member ID": "SELECT COUNT(*) AS n FROM claims WHERE member_id IS NULL",
    "Negative amounts": "SELECT COUNT(*) AS n FROM claims WHERE amount <= 0",
    "Rejected without a reason": "SELECT COUNT(*) AS n FROM claims WHERE status = 'Rejected' AND reason IS NULL",
}
with tab3:
    st.subheader("Validation checks on the raw feed")
    st.caption("Run before any number reaches a report. Negative amounts are already excluded from the KPIs.")
    rows = [{"Check": k, "Rows": int(q(v).n.iloc[0])} for k, v in DQ_SQL.items()]
    dq = pd.DataFrame(rows)
    dq["Status"] = np.where(dq.Rows == 0, "OK", "Review")
    st.dataframe(dq, width="stretch", hide_index=True)
    st.caption(f"Total rows in the feed: {int(q('SELECT COUNT(*) n FROM claims').n.iloc[0]):,}")

# ---------------------------------------------------------------- SQL
with tab4:
    st.subheader("Queries behind this dashboard")
    st.caption("SQLite in the browser. Same SQL runs on Snowflake or SQL Server with minor changes.")
    for title, sql in [("KPIs", KPI_SQL), ("Monthly trend", TREND_SQL),
                       ("Rejection reasons", REASON_SQL), ("Provider anomalies", PROV_SQL)]:
        with st.expander(title):
            st.code(sql.strip(), language="sql")
