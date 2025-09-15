# app.py
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

MODELDIR = Path("models")
DATADIR  = Path("data")

# ---------- Cached data loader ----------
@st.cache_data(show_spinner=False)
def load_raw_df():
    """Load the wide-format dataset once and cache it."""
    df = pd.read_csv(DATADIR / "compiled_clean_data.csv")
    # normalize STATE just in case
    if "STATE" in df.columns:
        df["STATE"] = df["STATE"].astype(str).str.strip().str.upper()
    return df

raw_df = load_raw_df()

# ---------- GeoJSON (cached): build an exact-case mapper ----------
@st.cache_data(show_spinner=False)
def load_india_geojson_and_mapper():
    with open(DATADIR / "india_state.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    # NAME_1 values in the geojson are Title/Proper case (e.g., "Andhra Pradesh")
    # Build a mapping: UPPER -> exact case from geojson
    name_case_map = {}
    for feat in gj.get("features", []):
        name_exact = str(feat.get("properties", {}).get("NAME_1", "")).strip()
        if name_exact:
            name_case_map[name_exact.upper()] = name_exact
    return gj, name_case_map

gj, NAME_CASE_MAP = load_india_geojson_and_mapper()

# handle legacy / alternate spellings in the CSV/model vs geojson
ALIASES = {
    "ORISSA": "ODISHA",
    "UTTARANCHAL": "UTTARAKHAND",
    "PONDICHERRY": "PUDUCHERRY",
    "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
    "JAMMU AND KASHMIR (UT)": "JAMMU AND KASHMIR",
}

def to_gj_exact_case(state_upper: str) -> str | None:
    """Return the NAME_1 string exactly as in the GeoJSON (case-sensitive)."""
    s = state_upper.strip().upper().replace("&", "AND")
    s = ALIASES.get(s, s)
    s = " ".join(s.split())
    return NAME_CASE_MAP.get(s)  # None if not found

# ---------- Cached helpers ----------
@st.cache_resource(show_spinner=False)
def load_bundle(which: str):
    """Load model/encoder/feature order for the chosen algorithm (cached)."""
    if which == "XGBoost":
        model   = pickle.load(open(MODELDIR / "xgb_full_model.pkl", "rb"))
        encoder = pickle.load(open(MODELDIR / "xgb_state_ohe.pkl",  "rb"))
        featord = pickle.load(open(MODELDIR / "xgb_feature_order.pkl", "rb"))
    else:  # RandomForest
        model   = pickle.load(open(MODELDIR / "rf_full_model.pkl",  "rb"))
        encoder = pickle.load(open(MODELDIR / "rf_state_ohe.pkl",   "rb"))
        featord = pickle.load(open(MODELDIR / "rf_feature_order.pkl", "rb"))
    return model, encoder, featord

def make_X(state: str, year: int, encoder, feat_ord):
    row = np.hstack([encoder.transform([[state]]), [[year]]])
    return pd.DataFrame(row, columns=feat_ord)

@st.cache_data(show_spinner=False)
def hist(df: pd.DataFrame, series_prefix: str, state: str):
    """Cached history extractor for 2005–2018."""
    return [
        df.loc[df.STATE.eq(state), f"{series_prefix}_{y}"].values[0]
        if f"{series_prefix}_{y}" in df.columns else np.nan
        for y in range(2005, 2019)
    ]

# ---- Trends for fallback (cached) ----
YEARS = list(range(2005, 2019))
states = sorted(raw_df["STATE"].unique().tolist())

@st.cache_data(show_spinner=False)
def compute_trends(df: pd.DataFrame, states_list, years_list):
    """Build per-state linear trends for IIP/AFOLU/WASTE/VEHICLE."""
    def build_trend(prefix):
        trends = {}
        for s in states_list:
            xs, ys = [], []
            for y in years_list:
                col = f"{prefix}_{y}"
                if col in df.columns:
                    v = df.loc[df.STATE.eq(s), col].squeeze()
                    if pd.notna(v):
                        xs.append(y); ys.append(float(v))
            if len(xs) >= 2:
                m, b = np.polyfit(xs, ys, 1)  # y = m*year + b
                trends[s] = (m, b)
        return trends

    return {
        "IIP":      build_trend("IIP"),
        "AFOLU":    build_trend("AFOLU"),
        "WASTE":    build_trend("WASTE"),
        "VEHICLE":  build_trend("VEHICLE"),
    }

TREND = compute_trends(raw_df, states, YEARS)

def trend_predict(state, year, name):
    m_b = TREND[name].get(state)
    if m_b is None:
        return None
    m, b = m_b
    return m * year + b

def predict_with_fallback(state: str, year: int, model, encoder, feat_ord):
    """Use chosen model inside 2005–2018; linear-trend fallback outside."""
    in_range = 2005 <= year <= 2018
    iip, afolu, waste, veh, pop, nsdp = model.predict(make_X(state, year, encoder, feat_ord))[0]
    if not in_range:
        iip_t   = trend_predict(state, year, "IIP")
        afolu_t = trend_predict(state, year, "AFOLU")
        waste_t = trend_predict(state, year, "WASTE")
        veh_t   = trend_predict(state, year, "VEHICLE")
        iip   = iip_t   if iip_t   is not None else iip
        afolu = afolu_t if afolu_t is not None else afolu
        waste = waste_t if waste_t is not None else waste
        veh   = veh_t   if veh_t   is not None else veh
    return iip, afolu, waste, veh, pop, nsdp

def fmt(x, mt=False, decimals=2):
    scale = 1e6 if mt else 1.0
    return f"{(x/scale):,.{decimals}f}"

# ---------- UI ----------
st.set_page_config(page_title="GHG Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align:center;margin-bottom:0.25rem'>State-Wise CO₂ Emission Dashboard</h1>"
    "<div style='text-align:center;color:#8c8c8c'>RF / XGBoost multi-output model • Units in tCO₂e / MtCO₂e</div>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1, 3])

with left:
    model_choice = st.radio("Model", ["RandomForest", "XGBoost"], horizontal=True, index=1)
    model, encoder, feat_ord = load_bundle(model_choice)

    st_state = st.selectbox("State", states,
                            index=states.index("ANDHRA PRADESH") if "ANDHRA PRADESH" in states else 0)
    st_year  = int(st.number_input("Year", min_value=1900, max_value=2100, value=2015, step=1))
    unit     = st.radio("Display unit", ["tCO₂e", "MtCO₂e"], index=0, horizontal=True)
    mt       = (unit == "MtCO₂e")

    iip, afolu, waste, veh, pop, nsdp = predict_with_fallback(st_state, st_year, model, encoder, feat_ord)
    net_emission = iip + afolu + waste
    afolu_source = max(afolu, 0.0); afolu_sink = -min(afolu, 0.0)

    # YoY box
    prev_iip, prev_afolu, prev_waste, _, _, _ = predict_with_fallback(
        st_state, max(st_year - 1, 1900), model, encoder, feat_ord
    )
    prev_net = prev_iip + prev_afolu + prev_waste
    yoy_abs  = net_emission - prev_net
    yoy_pct  = (yoy_abs / abs(prev_net) * 100.0) if prev_net != 0 else np.nan

    per_capita = (net_emission / pop) if pop and pop > 0 else np.nan
    nsdp_crore = nsdp / 1e7 if nsdp and nsdp > 0 else np.nan
    intensity  = (net_emission / nsdp_crore) if nsdp_crore and nsdp_crore > 0 else np.nan

    pos_sources = [v for v in [max(iip,0), max(afolu,0), max(waste,0)] if v > 0]
    pos_total   = sum(pos_sources) if len(pos_sources) else 0.0
    share = lambda v: (v / pos_total * 100.0) if pos_total > 0 else 0.0
    iip_share, afolu_share, waste_share = share(max(iip,0)), share(max(afolu,0)), share(max(waste,0))
    sink_offset_pct = (afolu_sink / pos_total * 100.0) if pos_total > 0 else 0.0

    st.subheader("Insights")
    st.markdown(
        f"""
        <div style="background:#151515;border-radius:12px;padding:14px;line-height:1.5">
          <div><b>Net emission:</b> {fmt(net_emission, mt)} {unit}</div>
          <div><b>YoY change vs {st_year-1}:</b> {fmt(yoy_abs, mt)} {unit}
               {"(" + ("+" if yoy_pct>=0 else "") + f"{yoy_pct:.2f}%" + ")" if not np.isnan(yoy_pct) else ""}</div>
          <div><b>Per-capita:</b> {per_capita:,.3f} tCO₂e / person</div>
          <div><b>Intensity:</b> {intensity:,.2f} tCO₂e per ₹ crore NSDP</div>
          <hr style="border-color:#2a2a2a;margin:10px 0">
          <div><b>Sector shares (positive sources only):</b></div>
          <ul style="margin-top:6px">
            <li>IIP: {iip_share:.1f}%</li>
            <li>AFOLU: {afolu_share:.1f}%</li>
            <li>WASTE: {waste_share:.1f}%</li>
          </ul>
          <div><b>AFOLU sink offset:</b> {fmt(afolu_sink, mt)} {unit} ({sink_offset_pct:.1f}% of positive sources)</div>
          <div style="margin-top:6px"><b>Drivers:</b> Vehicles {veh:,.0f} | Population {int(pop):,} | NSDP ₹ {nsdp:,.0f}</div>
        </div>
        """, unsafe_allow_html=True
    )

with right:
    tabs = st.tabs(["India Map", "Sector Share", "History 2005–2018", "Details"])

    # ===== India Map (fixed matching using exact-case NAME_1) =====
    with tabs[0]:
        st.markdown(f"**State-wise predictions for {st_year}**")
        with st.spinner("Computing predictions across states..."):
            rows = []
            for s in states:
                i, a, w, v, p, n = predict_with_fallback(s, st_year, model, encoder, feat_ord)
                net = i + a + w
                gj_name_exact = to_gj_exact_case(s)  # <-- exact case from geojson
                rows.append({
                    "STATE": s,
                    "GJ_NAME_EXACT": gj_name_exact,
                    "IIP": i, "AFOLU": a, "WASTE": w,
                    "VEHICLE_COUNT": v, "TOTAL_POPULATION": p, "NSDP_2024_25": n,
                    "NET": net
                })

        df_pred = pd.DataFrame(rows).dropna(subset=["GJ_NAME_EXACT"]).copy()
        df_pred["NET_DISPLAY"] = df_pred["NET"] / (1e6 if mt else 1.0)

        if df_pred.empty:
            st.warning("No states matched the GeoJSON names. Check alias mapping.")
        else:
            fig_map = px.choropleth(
                df_pred,
                geojson=gj,
                locations="GJ_NAME_EXACT",                  # exact-case strings
                featureidkey="properties.NAME_1",           # exact-key in geojson
                color="NET_DISPLAY",
                color_continuous_scale="YlOrRd",
                hover_name="STATE",
                hover_data={
                    "NET_DISPLAY": (":,.2f" if mt else ":,.0f"),
                    "IIP": ":,.0f",
                    "AFOLU": ":,.0f",
                    "WASTE": ":,.0f",
                    "VEHICLE_COUNT": ":,.0f",
                    "TOTAL_POPULATION": ":,.0f",
                    "NSDP_2024_25": ":,.0f",
                    "GJ_NAME_EXACT": False
                },
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                height=560,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(title=f"Net ({'MtCO₂e' if mt else 'tCO₂e'})")
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # --- Sector Share ---
    with tabs[1]:
        pie_vals, pie_labels = [], []
        if iip > 0:           pie_vals.append(iip);          pie_labels.append("IIP")
        if afolu_source > 0:  pie_vals.append(afolu_source); pie_labels.append("AFOLU")
        if waste > 0:         pie_vals.append(waste);        pie_labels.append("WASTE")

        if len(pie_vals) == 0:
            st.info("No positive emission sources for this selection.")
            if afolu_sink > 0:
                st.success(f"AFOLU sink: {fmt(afolu_sink, mt)} {unit}")
        else:
            vals = [v/1e6 if mt else v for v in pie_vals]
            fig = go.Figure(data=[go.Pie(labels=pie_labels, values=vals, hole=0.35)])
            title = f"{st_state} {st_year} – Share of Positive Sources ({unit})"
            if afolu_sink > 0:
                title += f"  |  AFOLU sink: {fmt(afolu_sink, mt)} {unit}"
            fig.update_layout(
                title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18}},
                height=430, margin=dict(l=10, r=10, t=70, b=10),
                legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- History 2005–2018 ---
    with tabs[2]:
        yrs = YEARS
        hist_iip   = hist(raw_df, "IIP",   st_state)
        hist_afolu = hist(raw_df, "AFOLU", st_state)
        hist_waste = hist(raw_df, "WASTE", st_state)

        y_iip   = [v/1e6 if mt and pd.notna(v) else (v if pd.notna(v) else None) for v in hist_iip]
        y_afolu = [v/1e6 if mt and pd.notna(v) else (v if pd.notna(v) else None) for v in hist_afolu]
        y_waste = [v/1e6 if mt and pd.notna(v) else (v if pd.notna(v) else None) for v in hist_waste]

        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_trace(go.Scatter(x=yrs, y=y_iip,   mode="lines+markers", name="IIP"))
        fig.add_trace(go.Scatter(x=yrs, y=y_afolu, mode="lines+markers", name="AFOLU"))
        fig.add_trace(go.Scatter(x=yrs, y=y_waste, mode="lines+markers", name="WASTE"))
        fig.add_trace(go.Scatter(x=[st_year], y=[(iip/1e6 if mt else iip)],
                                 mode="markers", name="IIP (pred)", marker=dict(size=12)))
        fig.add_trace(go.Scatter(x=[st_year], y=[(afolu/1e6 if mt else afolu)],
                                 mode="markers", name="AFOLU (pred)", marker=dict(size=12)))
        fig.add_trace(go.Scatter(x=[st_year], y=[(waste/1e6 if mt else waste)],
                                 mode="markers", name="WASTE (pred)", marker=dict(size=12)))
        fig.update_layout(
            title={"text": f"{st_state} – History (2005–2018) + Predicted Point ({unit})",
                   "x": 0.5, "xanchor": "center", "font": {"size": 18}},
            height=430, margin=dict(l=10, r=10, t=70, b=10),
            xaxis_title="Year", yaxis_title=f"Emission ({unit})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Details ---
    with tabs[3]:
        df_show = pd.DataFrame({
            "Metric": ["IIP", "AFOLU (negative = sink)", "WASTE",
                       "Vehicle Count", "Population", "NSDP 2024–25", "Net Emission"],
            "Value": [
                f"{fmt(iip, mt)} {unit}",
                f"{fmt(afolu, mt)} {unit}",
                f"{fmt(waste, mt)} {unit}",
                f"{int(veh):,}",
                f"{int(pop):,}",
                f"₹ {nsdp:,.0f}",
                f"{fmt(net_emission, mt)} {unit}",
            ]
        })
        st.dataframe(df_show, use_container_width=True)

st.caption("• AFOLU can be negative (sink). • RF/XGB predict within 2005–2018; linear-trend fallback extrapolates outside. • Map uses india_state.geojson and shows state-wise net emissions for the selected year.")
