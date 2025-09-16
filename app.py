# app.py
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # for charts

import folium
from folium.features import GeoJsonTooltip, GeoJsonPopup
import branca.colormap as cm
from streamlit.components.v1 import html as st_html  # embed HTML so map interactions don't trigger reruns

MODELDIR = Path("models")
DATADIR  = Path("data")
GEOJSON_FILENAME = "india_state.geojson"  # your GADM-like file (NAME_1)

# ---------- Cached data loader ----------
@st.cache_data(show_spinner=False)
def load_raw_df():
    df = pd.read_csv(DATADIR / "compiled_clean_data.csv")
    if "STATE" in df.columns:
        df["STATE"] = (
            df["STATE"].astype(str).str.strip().str.upper()
              .str.replace("&", "AND", regex=False)
              .str.replace(r"\s+", " ", regex=True)
        )
    return df

raw_df = load_raw_df()

# ---------- GeoJSON (auto-detect name field; build synonym map) ----------
@st.cache_data(show_spinner=False)
def load_india_geojson_and_mapper():
    with open(DATADIR / GEOJSON_FILENAME, "r", encoding="utf-8") as f:
        gj = json.load(f)

    cand_fields = ["NAME_1", "st_nm", "STATE", "STATE_NAME", "name"]
    sample_props = (gj.get("features") or [{}])[0].get("properties", {})
    name_field = next((f for f in cand_fields if f in sample_props), None)
    if not name_field:
        raise ValueError("Could not find a state-name property in GeoJSON (tried NAME_1, st_nm, STATE, STATE_NAME, name).")

    # UPPER -> exact case map; include VARNAME_1 synonyms if available
    name_case_map = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        base = str(props.get(name_field, "")).strip()
        if not base:
            continue
        name_case_map[base.upper().replace("&", "AND")] = base
        varalts = str(props.get("VARNAME_1", "") or "")
        if varalts:
            for alt in varalts.split("|"):
                alt = alt.strip()
                if alt:
                    name_case_map[alt.upper().replace("&", "AND")] = base
    return gj, name_field, name_case_map

gj, NAME_FIELD, NAME_CASE_MAP = load_india_geojson_and_mapper()

# ---------- Aliases to reconcile CSV ↔ GeoJSON ----------
ALIASES = {
    # Modern → older/GADM spelling or what your file contains
    "ODISHA": "ORISSA",
    "UTTARAKHAND": "UTTARANCHAL",
    "PONDICHERRY": "PUDUCHERRY",
    "PUDUCHERRY": "PUDUCHERRY",
    "DELHI (NCT)": "DELHI",
    "NCT OF DELHI": "DELHI",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN AND NICOBAR",
    "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN AND NICOBAR",
    "JAMMU AND KASHMIR (UT)": "JAMMU AND KASHMIR",
    "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
    # UT merge variants (fallback to one present in older files)
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI",
    "DADRA & NAGAR HAVELI AND DAMAN & DIU": "DADRA AND NAGAR HAVELI",
    # Telangana missing in many old files → map to AP polygon (explicit)
    "TELANGANA": "ANDHRA PRADESH",
}

def to_gj_exact_case(state_upper: str) -> str | None:
    """Normalize CSV STATE to the GeoJSON naming and return exact-case string from GeoJSON."""
    s = state_upper.strip().upper().replace("&", "AND")
    s = " ".join(s.split())
    s = ALIASES.get(s, s)
    return NAME_CASE_MAP.get(s)  # None if not found

# ---------- Model & helpers ----------
@st.cache_resource(show_spinner=False)
def load_bundle(which: str):
    if which == "XGBoost":
        model   = pickle.load(open(MODELDIR / "xgb_full_model.pkl", "rb"))
        encoder = pickle.load(open(MODELDIR / "xgb_state_ohe.pkl",  "rb"))
        featord = pickle.load(open(MODELDIR / "xgb_feature_order.pkl", "rb"))
    else:
        model   = pickle.load(open(MODELDIR / "rf_full_model.pkl",  "rb"))
        encoder = pickle.load(open(MODELDIR / "rf_state_ohe.pkl",   "rb"))
        featord = pickle.load(open(MODELDIR / "rf_feature_order.pkl", "rb"))
    return model, encoder, featord

def make_X(state: str, year: int, encoder, feat_ord):
    row = np.hstack([encoder.transform([[state]]), [[year]]])
    return pd.DataFrame(row, columns=feat_ord)

@st.cache_data(show_spinner=False)
def hist(df: pd.DataFrame, series_prefix: str, state: str):
    return [
        df.loc[df.STATE.eq(state), f"{series_prefix}_{y}"].values[0]
        if f"{series_prefix}_{y}" in df.columns else np.nan
        for y in range(2005, 2019)
    ]

YEARS = list(range(2005, 2019))
states = sorted(raw_df["STATE"].unique().tolist())

@st.cache_data(show_spinner=False)
def compute_trends(df: pd.DataFrame, states_list, years_list):
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
                m, b = np.polyfit(xs, ys, 1)
                trends[s] = (m, b)
        return trends
    return {
        "IIP": build_trend("IIP"),
        "AFOLU": build_trend("AFOLU"),
        "WASTE": build_trend("WASTE"),
        "VEHICLE": build_trend("VEHICLE"),
    }

TREND = compute_trends(raw_df, states, YEARS)

def trend_predict(state, year, name):
    m_b = TREND[name].get(state)
    if m_b is None: return None
    m, b = m_b
    return m * year + b

def predict_with_fallback(state: str, year: int, model, encoder, feat_ord):
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

# ---------- Folium builder (AGGREGATES duplicates to ensure unique index) ----------
def build_folium_map(df_pred: pd.DataFrame, unit_label: str, mt: bool):
    """
    df_pred columns needed:
      - GJ_NAME_EXACT (matches geojson properties[NAME_FIELD])
      - STATE, IIP, AFOLU, WASTE, VEHICLE_COUNT, TOTAL_POPULATION, NSDP_2024_25, NET
    """
    if df_pred.empty:
        return None

    # Aggregate duplicates so index is unique (e.g., TELANGANA merged into AP)
    agg = {
        "IIP": "sum",
        "AFOLU": "sum",
        "WASTE": "sum",
        "VEHICLE_COUNT": "sum",
        "TOTAL_POPULATION": "sum",
        "NSDP_2024_25": "sum",
        "NET": "sum",
    }
    df_agg = (
        df_pred.groupby("GJ_NAME_EXACT", as_index=False)
               .agg({**agg, "STATE": lambda s: ", ".join(sorted(set(s)))})
    )

    df_agg["NET_DISPLAY"] = df_agg["NET"] / (1e6 if mt else 1.0)
    vmin, vmax = float(df_agg["NET_DISPLAY"].min()), float(df_agg["NET_DISPLAY"].max())
    if vmin == vmax:
        vmin, vmax = 0.0, max(1.0, vmax)

    # Tidy stepped legend
    steps = list(np.linspace(vmin, vmax, 5))
    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax).to_step(index=steps)
    colormap.caption = f"Net emissions ({'MtCO₂e' if mt else 'tCO₂e'})"

    # Unique lookup by matched geojson name
    lookup = df_agg.set_index("GJ_NAME_EXACT").to_dict(orient="index")

    # Build augmented GeoJSON
    gj_aug = {"type": "FeatureCollection", "features": []}
    for feat in gj.get("features", []):
        props = dict(feat.get("properties", {}) or {})
        name = props.get(NAME_FIELD)
        row  = lookup.get(name)
        if row is None:
            continue
        props.update({
            "STATE_CSV": row["STATE"],  # shows CSV names merged into this polygon
            "NET_DISPLAY": float(row["NET_DISPLAY"]),
            "IIP": float(row["IIP"]),
            "AFOLU": float(row["AFOLU"]),
            "WASTE": float(row["WASTE"]),
            "VEHICLE_COUNT": int(row["VEHICLE_COUNT"]),
            "TOTAL_POPULATION": int(row["TOTAL_POPULATION"]),
            "NSDP_2024_25": float(row["NSDP_2024_25"]),
        })
        gj_aug["features"].append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": props,
        })

    m = folium.Map(location=[22.8, 79], zoom_start=4.5,
                   tiles="CartoDB positron", control_scale=True)

    def style_fn(feature):
        val = feature["properties"].get("NET_DISPLAY")
        return {
            "fillColor": colormap(val) if val is not None else "#f0f0f0",
            "color": "#555555",
            "weight": 0.6,
            "opacity": 0.7,
            "fillOpacity": 0.85 if val is not None else 0.1,
        }

    tooltip = GeoJsonTooltip(
        fields=[NAME_FIELD, "STATE_CSV", "NET_DISPLAY"],
        aliases=["GeoJSON name:", "CSV name(s):", f"Net ({'MtCO₂e' if mt else 'tCO₂e'}):"],
        localize=True, sticky=False, labels=True,
        style=("background-color: white; color: #333; font-family: arial; "
               "font-size: 12px; padding: 8px; border-radius: 6px;")
    )

    popup = GeoJsonPopup(
        fields=[NAME_FIELD, "NET_DISPLAY", "IIP", "AFOLU", "WASTE",
                "VEHICLE_COUNT", "TOTAL_POPULATION", "NSDP_2024_25", "STATE_CSV"],
        aliases=["State:", f"Net ({'MtCO₂e' if mt else 'tCO₂e'}):", "IIP:", "AFOLU:",
                 "WASTE:", "Vehicles:", "Population:", "NSDP 2024–25 (₹):", "CSV name(s):"],
        localize=True, labels=True, parse_html=False, max_width=360,
    )

    folium.GeoJson(
        gj_aug, name="Net Emissions", style_function=style_fn,
        tooltip=tooltip, popup=popup,
        highlight_function=lambda f: {"weight": 2.5, "color": "#000", "fillOpacity": 0.9},
    ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m

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

    states = sorted(raw_df["STATE"].unique().tolist())
    st_state = st.selectbox("State", states,
                            index=states.index("ANDHRA PRADESH") if "ANDHRA PRADESH" in states else 0)
    st_year  = int(st.number_input("Year", min_value=1900, max_value=2100, value=2015, step=1))
    unit     = st.radio("Display unit", ["tCO₂e", "MtCO₂e"], index=0, horizontal=True)
    mt       = (unit == "MtCO₂e")

    iip, afolu, waste, veh, pop, nsdp = predict_with_fallback(st_state, st_year, model, encoder, feat_ord)
    net_emission = iip + afolu + waste
    afolu_source = max(afolu, 0.0); afolu_sink = -min(afolu, 0.0)

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

    # ===== India Map (Folium only; embed HTML to avoid reruns on zoom) =====
    with tabs[0]:
        st.markdown(f"**State-wise predictions for {st_year}**")

        with st.spinner("Computing predictions across states..."):
            rows = []
            for s in states:
                i, a, w, v, p, n = predict_with_fallback(s, st_year, model, encoder, feat_ord)
                net = i + a + w
                gj_name_exact = to_gj_exact_case(s)  # exact-case name from GeoJSON or None
                rows.append({
                    "STATE": s,
                    "GJ_NAME_EXACT": gj_name_exact,
                    "IIP": i, "AFOLU": a, "WASTE": w,
                    "VEHICLE_COUNT": v, "TOTAL_POPULATION": p, "NSDP_2024_25": n,
                    "NET": net
                })

        df_pred_all = pd.DataFrame(rows)
        df_pred     = df_pred_all.dropna(subset=["GJ_NAME_EXACT"]).copy()

        # Diagnostics: unmatched CSV states (helps patch aliases quickly)
        unmatched = sorted(df_pred_all.loc[df_pred_all.GJ_NAME_EXACT.isna(), "STATE"].unique().tolist())
        if unmatched:
            st.warning("Unmatched CSV state names (not found in GeoJSON after aliasing): " + ", ".join(unmatched))
            if "TELANGANA" in unmatched:
                st.info("Note: TELANGANA is merged into ANDHRA PRADESH (older GeoJSONs often lack a separate Telangana).")

        # Diagnostics: show which CSV states collapsed into same polygon
        dups = df_pred[df_pred.duplicated("GJ_NAME_EXACT", keep=False)]
        if not dups.empty:
            merged = (dups.groupby("GJ_NAME_EXACT")["STATE"]
                      .apply(lambda s: ", ".join(sorted(set(s)))).to_dict())
            st.info("Merged multiple CSV states into one polygon:\n" +
                    "\n".join([f"• {k}: {v}" for k, v in merged.items()]))

        if df_pred.empty:
            st.error("No states matched the GeoJSON names. Check alias mapping / choose a newer GeoJSON.")
        else:
            m = build_folium_map(df_pred, unit, mt)  # <-- uses aggregated data internally
            st_html(m.get_root().render(), height=560)

    # --- Sector Share ---
    with tabs[1]:
        afolu_source = max(afolu, 0.0)
        afolu_sink   = -min(afolu, 0.0)

        pie_vals, pie_labels = [], []
        if iip > 0:           pie_vals.append(iip);          pie_labels.append("IIP")
        if afolu_source > 0:  pie_vals.append(afolu_source); pie_labels.append("AFOLU")
        if waste > 0:         pie_vals.append(waste);        pie_labels.append("WASTE")

        if len(pie_vals) == 0:
            st.info("No positive emission sources for this selection.")
            if afolu_sink > 0:
                st.success(f"AFOLU sink: {fmt(afolu_sink, mt)} {('MtCO₂e' if mt else 'tCO₂e')}")
        else:
            vals = [v/1e6 if mt else v for v in pie_vals]
            fig = go.Figure(data=[go.Pie(labels=pie_labels, values=vals, hole=0.35)])
            title = f"{st_state} {st_year} – Share of Positive Sources ({'MtCO₂e' if mt else 'tCO₂e'})"
            if afolu_sink > 0:
                title += f"  |  AFOLU sink: {fmt(afolu_sink, mt)} {('MtCO₂e' if mt else 'tCO₂e')}"
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

        mt_unit = ('MtCO₂e' if mt else 'tCO₂e')

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
            title={"text": f"{st_state} – History (2005–2018) + Predicted Point ({mt_unit})",
                   "x": 0.5, "xanchor": "center", "font": {"size": 18}},
            height=430, margin=dict(l=10, r=10, t=70, b=10),
            xaxis_title="Year", yaxis_title=f"Emission ({mt_unit})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Details ---
    with tabs[3]:
        df_show = pd.DataFrame({
            "Metric": ["IIP", "AFOLU (negative = sink)", "WASTE",
                       "Vehicle Count", "Population", "NSDP 2024–25", "Net Emission"],
            "Value": [
                f"{fmt(iip, mt)} {('MtCO₂e' if mt else 'tCO₂e')}",
                f"{fmt(afolu, mt)} {('MtCO₂e' if mt else 'tCO₂e')}",
                f"{fmt(waste, mt)} {('MtCO₂e' if mt else 'tCO₂e')}",
                f"{int(veh):,}",
                f"{int(pop):,}",
                f"₹ {nsdp:,.0f}",
                f"{fmt(iip + afolu + waste, mt)} {('MtCO₂e' if mt else 'tCO₂e')}",
            ]
        })
        st.dataframe(df_show, use_container_width=True)

st.caption("• Aggregates duplicate matches (e.g., Telangana→AP) to avoid index errors. • Robust name matching (auto-detect NAME_1/st_nm + VARNAME_1). • Folium embedded as HTML so zoom/pan doesn’t rerun the app.")
