#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SMARTSTOCK :: ENTERPRISE DASHBOARD   (main_app.py)
================================================================================

Streamlit front-end for the SmartStock HDFS & PySpark Inventory Analytics Suite.

ARCHITECTURE (this is the slide you show first in the viva)
-----------------------------------------------------------
    +------------------------------------------------------------------+
    |  HDFS RAW LAYER (simulated)   hdfs/user/hadoop/inventory/raw/    |
    |      products.csv | stock.csv | sales_ledger.csv  + _SUCCESS     |
    +--------------------------------+---------------------------------+
                                     |
                    (1) auto-bootstrap if the layer is not committed
                                     v
    +------------------------------------------------------------------+
    |  PySparkInventoryEngine  (pyspark_analytics_engine.py)           |
    |   read w/ StructType -> groupBy -> window(ABC) -> ITR -> flags    |
    |   -> LEFT JOIN spine -> toPandas()  [ACTIONS]                     |
    +--------------------------------+---------------------------------+
                                     |
                    (2) st.cache_resource: the SparkSession is created
                        ONCE and reused across every widget interaction
                                     v
    +------------------------------------------------------------------+
    |  Streamlit UI (this file)                                        |
    |   sidebar filters -> KPI cards -> Plotly charts -> action table    |
    |   filtering happens in Pandas on the 60-row result (a driver-sized|
    |   payload), NOT in Spark -- see "WHERE THE WORK HAPPENS" below.    |
    +------------------------------------------------------------------+

WHERE THE WORK HAPPENS  (the architecture question to be ready for)
---------------------------------------------------------------------
* Spark  : all the *heavy* work -- parsing 2,500 typed rows, the shuffle
           for the groupBy, the Pareto window, 3 broadcast joins.
* Driver : all the *presentation* work -- filtering a 60-row Pandas frame
           and drawing charts.  This split is deliberate: a Pandas DataFrame
           is single-process, so shipping 2,500 raw rows to the UI on every
           slider move would be wasteful, while shipping 2,500 *aggregated*
           rows would be a design error.  We ship exactly 60.

CACHING STRATEGY (why the dashboard feels instant)
--------------------------------------------------
`st.cache_resource` caches the pipeline RESULT (and keeps the SparkSession
alive) keyed on nothing, so moving a filter widget re-runs only the Pandas
layer -- milliseconds instead of the ~20 s Spark job.  `st.cache_data` is
*not* used here on purpose: it would hash and copy the frames, and we want
the engine instance itself to be reused so its JVM is not restarted.
================================================================================
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Make the project importable no matter where Streamlit is launched from
# (`streamlit run C:\...\main_app.py` from any directory).
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.express as px
import plotly.graph_objects as go

from hdfs_storage_mock import (
    CATEGORIES,
    HDFS_LOCAL_ROOT,
    ensure_hdfs_dataset,
    hdfs_dataset_exists,
    read_manifest,
)
from pyspark_analytics_engine import PySparkInventoryEngine

# =============================================================================
# SECTION 1 :: PAGE CONFIG + BRANDING
# =============================================================================

st.set_page_config(
    page_title="SmartStock | HDFS & PySpark Inventory Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

#: Enterprise colour palette. Class A = the gold, class C = the problem.
ABC_COLORS = {"A": "#00b894", "B": "#fdcb6e", "C": "#e17055"}
SEVERITY_COLORS = {
    "CRITICAL": "#d63031", "HIGH": "#e17055",
    "MEDIUM": "#fdcb6e", "OK": "#00b894",
}
PRIMARY = "#0b3c5d"      # deep navy - headers / sidebar
ACCENT = "#1d7874"       # teal - buttons, accents

CSS = """
<style>
/* ---------- Global polish ---------- */
.stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%); }
.block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1500px; }

/* ---------- Hero header ---------- */
.smartstock-hero {
    background: linear-gradient(115deg, #0b3c5d 0%, #1d7874 55%, #14b8a6 100%);
    padding: 1.5rem 1.9rem; border-radius: 16px; margin-bottom: 1.1rem;
    box-shadow: 0 10px 28px rgba(11, 60, 93, 0.28);
}
.smartstock-hero h1 { color: #ffffff; font-size: 2.05rem; margin: 0;
                      font-weight: 800; letter-spacing: -0.4px; }
.smartstock-hero p  { color: #d8f3ef; font-size: 0.95rem; margin: 0.45rem 0 0;
                      line-height: 1.5; }
.hero-pills { margin-top: 0.85rem; }
.hero-pill {
    display: inline-block; background: rgba(255,255,255,0.16);
    color: #ffffff; border: 1px solid rgba(255,255,255,0.32);
    border-radius: 999px; padding: 0.2rem 0.75rem;
    font-size: 0.76rem; margin-right: 0.4rem; font-weight: 600;
}

/* ---------- KPI metric cards ---------- */
.kpi-card {
    background: #ffffff; border: 1px solid #e3e9f0; border-radius: 14px;
    padding: 1.05rem 1.15rem 0.85rem 1.15rem;
    /* A fixed min-height makes all five cards exactly the same size, so the
       KPI numbers sit on one horizontal line across the row. */
    min-height: 196px;
    box-shadow: 0 3px 12px rgba(16,42,67,0.07);
    border-top: 4px solid var(--kpi-accent, #1d7874);
    transition: transform .15s ease, box-shadow .15s ease;
}
.kpi-card:hover { transform: translateY(-3px);
                  box-shadow: 0 8px 20px rgba(16,42,67,0.13); }
.kpi-label { color: #5c7081; font-size: 0.63rem; font-weight: 700;
             text-transform: uppercase; letter-spacing: 0.35px;
             line-height: 1.35; word-break: keep-all; hyphens: none;
             /* Reserve 4 lines so the KPI VALUES line up across the row,
                whatever length each label happens to be. */
             min-height: 5.4em; }
.kpi-value { color: #0b3c5d; font-size: 1.45rem; font-weight: 800;
             margin: 0.28rem 0 0.15rem 0; line-height: 1.1;
             letter-spacing: -0.5px; white-space: nowrap; }
.kpi-delta { font-size: 0.7rem; font-weight: 600; line-height: 1.4; }
.kpi-delta.up   { color: #00a884; }
.kpi-delta.warn { color: #d35400; }
.kpi-delta.bad  { color: #d63031; }
.kpi-delta.ok   { color: #5c7081; }
.kpi-hint { color: #8898a6; font-size: 0.63rem; line-height: 1.45;
            margin-top: 0.4rem; border-top: 1px dashed #e3e9f0;
            padding-top: 0.4rem; }

/* ---------- Section headings ---------- */
.section-head { display: flex; align-items: center; gap: 0.55rem;
                margin: 1.5rem 0 0.2rem 0; }
.section-head h3 { color: #0b3c5d; font-size: 1.16rem; margin: 0;
                   font-weight: 750; }
.section-head .dot { width: 9px; height: 22px; border-radius: 5px;
                     background: linear-gradient(180deg,#0b3c5d,#1d7874); }
.section-sub { color: #6b7c8c; font-size: 0.83rem; margin: 0.1rem 0 0.85rem 0; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b3c5d 0%,#10496a 100%); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: #e8f4f6 !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #7fe3d8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18); }

/* ---------- Misc ---------- */
div[data-testid="stExpander"] details {
    border: 1px solid #e3e9f0; border-radius: 12px; background: #ffffff; }
.small-note { color: #6b7c8c; font-size: 0.78rem; }
.footer { text-align: center; color: #8898a6; font-size: 0.75rem;
          margin-top: 2.2rem; border-top: 1px solid #e3e9f0; padding-top: 0.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# SECTION 2 :: FORMATTING HELPERS
# =============================================================================

def fmt_money(value: float, decimals: int = 0) -> str:
    """1234567.8 -> '$1,234,568'  (defensive: NaN/None -> 'n/a')."""
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return "${:,.{}f}".format(float(value), decimals)
    except (TypeError, ValueError):
        return "n/a"


def fmt_number(value: float, decimals: int = 0) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return "{:,.{}f}".format(float(value), decimals)
    except (TypeError, ValueError):
        return "n/a"


def kpi_card(label: str, value: str, delta_html: str = "", hint: str = "",
             accent: str = ACCENT) -> None:
    """Render one custom KPI card (Streamlit's st.metric cannot be styled)."""
    st.markdown(
        """
        <div class="kpi-card" style="--kpi-accent:{accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta {cls}">{delta}</div>
            <div class="kpi-hint">{hint}</div>
        </div>
        """.format(
            accent=accent, label=label, value=value,
            delta=delta_html or "&nbsp;", cls=_delta_class(delta_html),
            hint=hint or "&nbsp;",
        ),
        unsafe_allow_html=True,
    )


def _delta_class(delta_html: str) -> str:
    lowered = (delta_html or "").lower()
    for token, css_class in (("critical", "bad"), ("risk", "bad"),
                             ("low", "bad"), ("warn", "warn"),
                             ("good", "up"), ("healthy", "up"),
                             ("dead", "warn"), ("high", "warn")):
        if token in lowered:
            return css_class
    return "ok"


def section_head(icon: str, title: str, subtitle: str = "") -> None:
    """A consistent section heading with a coloured accent bar."""
    st.markdown(
        '<div class="section-head"><span class="dot"></span>'
        '<h3>{} {}</h3></div>'.format(icon, title),
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown('<div class="section-sub">{}</div>'.format(subtitle),
                    unsafe_allow_html=True)


# =============================================================================
# SECTION 3 :: DATA BOOTSTRAP + SPARK PIPELINE (cached)
# =============================================================================

def bootstrap_hdfs_layer() -> Dict[str, object]:
    """Ensure the simulated HDFS RAW layer is committed before Spark reads it.

    `hdfs_dataset_exists()` checks BOTH the data files and the Hadoop
    `_SUCCESS` markers, so we never read a half-written dataset.  This is the
    "data self-service" contract: the dashboard heals its own upstream.
    """
    if not hdfs_dataset_exists():
        with st.spinner("HDFS RAW layer empty - generating the simulated "
                        "dataset (products / stock / sales ledger) ..."):
            return ensure_hdfs_dataset()
    return read_manifest()


@st.cache_resource(show_spinner="Running the PySpark distributed pipeline "
                               "(first run also starts the Spark JVM) ...")
def load_pipeline() -> Dict[str, object]:
    """Execute the Spark pipeline ONCE and cache the result for the session.

    st.cache_resource (not st.cache_data) is the right decorator because we
    want to KEEP the PySparkInventoryEngine instance -- and therefore its
    SparkContext/JVM -- alive between reruns.  A filter interaction then only
    re-runs the cheap Pandas layer, so the UI stays responsive.
    """
    bootstrap_hdfs_layer()
    engine = PySparkInventoryEngine(verbose=True)
    results = engine.run_pipeline()
    results["engine"] = engine          # keep it cached -> keep the JVM alive
    return results


def rollup_slice(products: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Derive the category & ABC rollups for the CURRENTLY FILTERED slice.

    Why recompute in Pandas instead of re-running Spark on every slider move?
    Because the slice is at most 60 rows -- the equivalent of a lookup table.
    Spark is a batch engine; re-submitting a 60-row job per interaction would
    add ~10 s of scheduling latency for no benefit.  The architectural rule:
    distributed engine for volume, in-memory engine for interactivity.  When
    NO filter is active we use the Spark pre-aggregates shipped by the engine
    (see `build_dashboard_data`) so the default view is 100% Spark-computed.
    """
    categories = (
        products.groupby("category", as_index=False)
        .agg(sku_count=("product_id", "count"),
             revenue=("revenue", "sum"),
             cogs=("cogs", "sum"),
             units_sold=("units_sold", "sum"),
             inventory_value=("inventory_value", "sum"),
             dead_stock_items=("is_dead_stock", "sum"),
             stockout_items=("is_stockout_risk", "sum"))
        .sort_values("revenue", ascending=False)
    )
    abc = (
        products.groupby(["abc_class", "abc_class_label"], as_index=False)
        .agg(sku_count=("product_id", "count"),
             revenue=("revenue", "sum"),
             inventory_value=("inventory_value", "sum"),
             dead_stock_items=("is_dead_stock", "sum"),
             stockout_items=("is_stockout_risk", "sum"))
        .sort_values("abc_class")
    )
    totals = {
        "revenue": float(products["revenue"].sum()),
        "cogs": float(products["cogs"].sum()),
        "inventory_value": float(products["inventory_value"].sum()),
        "average_inventory_value": float(products["average_inventory_value"].sum()),
        "stockout": int(products["is_stockout_risk"].sum()),
        "dead": int(products["is_dead_stock"].sum()),
    }
    return categories, abc, totals


# =============================================================================
# SECTION 4 :: SIDEBAR  (filters + system status)
# =============================================================================

def render_sidebar(products: pd.DataFrame, meta: Dict[str, object]
                   ) -> Dict[str, object]:
    """All dashboard filters live here.  Returns the active filter state."""
    st.sidebar.markdown(
        """
        <div style="margin-bottom:.4rem;">
          <div style="font-size:1.28rem;font-weight:800;color:#fff;letter-spacing:-.3px;">
            📦 SmartStock</div>
          <div style="font-size:.76rem;color:#9fd8d2;">HDFS × PySpark Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("🔎 Filters")

    # Category multiselect ("All" == empty selection, the Streamlit idiom).
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        options=sorted(products["category"].unique()),
        default=[],
        placeholder="All categories",
    )
    selected_classes = st.sidebar.multiselect(
        "ABC Inventory Class",
        options=["A", "B", "C"],
        default=[],
        placeholder="All classes (A / B / C)",
    )
    selected_zones = st.sidebar.multiselect(
        "Warehouse Zone",
        options=sorted(products["warehouse_zone"].dropna().unique()),
        default=[],
        placeholder="All zones",
    )
    selected_severities = st.sidebar.multiselect(
        "Alert Severity",
        options=["CRITICAL", "HIGH", "MEDIUM", "OK"],
        default=[],
        placeholder="All severities",
    )
    min_revenue = st.sidebar.slider(
        "Minimum SKU revenue ($)", min_value=0,
        max_value=int(max(products["revenue"].max(), 1)),
        value=0, step=500,
    )
    only_alerts = st.sidebar.toggle(
        "Show only items needing action", value=False,
        help="Hides fully healthy SKUs (alert_severity == OK).",
    )

    if st.sidebar.button("↺ Reset all filters", width="stretch"):
        for key in ("cat", "cls", "zone", "sev"):
            st.session_state.pop("__reset_{}".format(key), None)
        st.rerun()

    # ---------------- system status panel ----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ System status")
    ingestion = meta.get("ingestion", {}) or {}
    st.sidebar.markdown(
        """
        <div style="font-size:.8rem;line-height:1.85;">
          <b>Spark</b> {}<br>
          <b>Master</b> {}<br>
          <b>Shuffle partitions</b> {}<br>
          <b>Sales window</b> {} days<br>
          <b>Transactions ingested</b> {}<br>
          <b>Pipeline wall clock</b> {}s
        </div>
        """.format(
            meta.get("spark_version"), meta.get("spark_master"),
            meta.get("shuffle_partitions"), meta.get("sales_window_days"),
            fmt_number(ingestion.get("sales_ledger_rows")),
            meta.get("total_seconds"),
        ),
        unsafe_allow_html=True,
    )

    if st.sidebar.button("♻️ Regenerate HDFS dataset", width="stretch"):
        with st.spinner("Regenerating the simulated HDFS RAW layer ..."):
            ensure_hdfs_dataset(force=True)
        # Drop the cached pipeline so the next render re-runs Spark on the
        # fresh data.  This is the correct invalidation order: data first,
        # then the cache key.
        st.cache_resource.clear()
        st.sidebar.success("Dataset regenerated.")
        st.rerun()

    return {
        "categories": selected_categories,
        "classes": selected_classes,
        "zones": selected_zones,
        "severities": selected_severities,
        "min_revenue": min_revenue,
        "only_alerts": only_alerts,
    }


def apply_filters(products: pd.DataFrame, filters: Dict[str, object]
                  ) -> pd.DataFrame:
    """Slice the 60-row Pandas frame.  Empty list == no filter (show all)."""
    mask = pd.Series(True, index=products.index)

    if filters["categories"]:
        mask &= products["category"].isin(filters["categories"])
    if filters["classes"]:
        mask &= products["abc_class"].isin(filters["classes"])
    if filters["zones"]:
        mask &= products["warehouse_zone"].isin(filters["zones"])
    if filters["severities"]:
        mask &= products["alert_severity"].isin(filters["severities"])
    if filters["min_revenue"]:
        mask &= products["revenue"] >= float(filters["min_revenue"])
    if filters["only_alerts"]:
        mask &= products["alert_severity"] != "OK"

    return products[mask].copy()


# =============================================================================
# SECTION 5 :: KPI ROW
# =============================================================================

def render_kpi_row(totals: Dict[str, float], kpis: Dict[str, object],
                   sku_count: int, total_skus: int, filtered: bool) -> None:
    """The four required KPI cards (+ a fifth commercial one)."""
    avg_inventory = totals["average_inventory_value"] or 0.0
    itr_period = (totals["cogs"] / avg_inventory) if avg_inventory else 0.0
    window_days = int(kpis.get("sales_window_days") or 90) or 90
    itr_annual = itr_period * (365.0 / window_days)
    scope = "filtered slice" if filtered else "enterprise"

    # Weighted columns: the revenue figure is the widest string, the SKU
    # counter the narrowest, so the cards get proportional room.
    columns = st.columns([1.25, 1.0, 0.95, 1.05, 0.9], gap="small")

    with columns[0]:
        kpi_card(
            "Total Enterprise Revenue ($)",
            fmt_money(totals["revenue"]),
            "{:.1f}% gross margin".format(
                (totals["revenue"] - totals["cogs"]) / totals["revenue"] * 100
                if totals["revenue"] else 0.0),
            "COGS {} · {} scope".format(fmt_money(totals["cogs"]), scope),
            accent="#0b3c5d",
        )
    with columns[1]:
        kpi_card(
            "Critical Stockout Reorder Alerts",
            fmt_number(totals["stockout"]),
            "on-hand ≤ safety level" if totals["stockout"] else "All buffers healthy",
            "Replenishment value {}".format(
                fmt_money(kpis.get("pending_reorder_value", 0))),
            accent="#d63031" if totals["stockout"] else "#00b894",
        )
    with columns[2]:
        kpi_card(
            "Dead Stock Items Found",
            fmt_number(totals["dead"]),
            "no sales in {} days".format(window_days) if totals["dead"]
            else "No dead stock found",
            "Capital frozen · markdown",
            accent="#e17055" if totals["dead"] else "#00b894",
        )
    with columns[3]:
        kpi_card(
            "Overall Inventory Turnover Ratio",
            "{:.2f}x".format(itr_annual),
            "{:.2f}x over the {}d window".format(itr_period, window_days),
            "COGS ÷ avg inventory value",
            accent="#1d7874",
        )
    with columns[4]:
        kpi_card(
            "SKUs in Scope",
            "{} / {}".format(sku_count, total_skus),
            "class A drives 80% of revenue",
            "Stock value {}".format(fmt_money(totals["inventory_value"])),
            accent="#6c5ce7",
        )


# =============================================================================
# SECTION 6 :: VISUAL ANALYTICS
# =============================================================================

def chart_revenue_by_category(categories: pd.DataFrame) -> None:
    """Horizontal bar: revenue generated per product category (REQUIRED #1)."""
    ordered = categories.sort_values("revenue", ascending=True)
    figure = px.bar(
        ordered, x="revenue", y="category", orientation="h",
        color="revenue", color_continuous_scale=["#9fd8d2", "#0b3c5d"],
        text=ordered["revenue"].map(lambda v: fmt_money(v)),
        labels={"revenue": "Revenue generated ($)", "category": ""},
        custom_data=["sku_count", "units_sold", "inventory_value"],
        title=None,
    )
    figure.update_traces(
        textposition="outside", textfont_size=11,
        hovertemplate=(
            "<b>%{y}</b><br>Revenue: $%{x:,.2f}<br>"
            "SKUs: %{customdata[0]}<br>Units sold: %{customdata[1]}<br>"
            "Inventory value: $%{customdata[2]:,.2f}<extra></extra>"),
        marker_line_width=0,
    )
    figure.update_layout(
        height=380, margin=dict(l=10, r=90, t=10, b=10),
        xaxis_title=None, yaxis_title=None, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=12),
    )
    st.plotly_chart(figure, width="stretch",
                    config={"displayModeBar": False})


def chart_abc_distribution(abc: pd.DataFrame) -> None:
    """Donut: percentage breakdown of class A / B / C stock items (REQUIRED #3)."""
    figure = go.Figure(go.Pie(
        labels=abc["abc_class_label"], values=abc["sku_count"],
        hole=0.55, sort=False, direction="clockwise",
        marker=dict(colors=[ABC_COLORS.get(c, "#8395a7") for c in abc["abc_class"]],
                    line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>SKUs: %{value}<br>"
                      "Share: %{percent}<extra></extra>",
    ))
    total_skus = int(abc["sku_count"].sum())
    figure.add_annotation(
        text="<b>{}</b><br>SKUs".format(total_skus),
        x=0.5, y=0.5, showarrow=False, font=dict(size=15, color="#0b3c5d"),
    )
    figure.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5,
                    xanchor="center", font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch",
                    config={"displayModeBar": False})


def chart_pareto_curve(products: pd.DataFrame, window_days: int = 90) -> None:
    """Pareto: bar = revenue per SKU, line = cumulative revenue %.

    This is the visual proof of the ABC cut performed by the Spark window
    function: the line crosses 80% and 95% exactly where the classes change.
    """
    ordered = products.sort_values("revenue", ascending=False).head(25).copy()
    ordered["cumulative_pct"] = (
        ordered["revenue"].cumsum() / ordered["revenue"].sum() * 100
        if ordered["revenue"].sum() else 0.0
    )
    figure = go.Figure()
    figure.add_bar(
        x=ordered["product_id"], y=ordered["revenue"], name="SKU revenue",
        marker_color=[ABC_COLORS.get(c, "#8395a7") for c in ordered["abc_class"]],
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>",
    )
    figure.add_scatter(
        x=ordered["product_id"], y=ordered["cumulative_pct"],
        name="Cumulative revenue %", mode="lines+markers",
        line=dict(color="#d63031", width=2.5, shape="hv"),
        marker=dict(size=4),
        yaxis="y2",
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
    )
    for threshold, colour in ((80, "#00b894"), (95, "#fdcb6e")):
        figure.add_hline(y=threshold, line_dash="dash", line_color=colour,
                         annotation_text="{}%".format(threshold),
                         annotation_position="top left",
                         annotation_font=dict(size=10, color=colour))
    figure.update_layout(
        height=400, barmode="overlay", margin=dict(t=15, b=10, l=10, r=10),
        xaxis=dict(title="SKU (top 25 by revenue)", tickangle=-60,
                   tickfont=dict(size=9)),
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch",
                    config={"displayModeBar": False})


def chart_severity_donut(products: pd.DataFrame) -> None:
    """Donut of the alert-severity mix: how the SKUs are split by urgency."""
    severity = (
        products.groupby("alert_severity", as_index=False)
        .agg(sku_count=("product_id", "count"),
             revenue=("revenue", "sum"))
        .sort_values("sku_count", ascending=False)
    )
    figure = go.Figure(go.Pie(
        labels=severity["alert_severity"], values=severity["sku_count"],
        hole=0.5, sort=False,
        marker=dict(colors=[SEVERITY_COLORS.get(s, "#8395a7")
                            for s in severity["alert_severity"]],
                    line=dict(color="#ffffff", width=2)),
        textinfo="label+value", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>SKUs: %{value}<extra></extra>",
    ))
    figure.update_layout(
        height=400, margin=dict(t=10, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch",
                    config={"displayModeBar": False})


# =============================================================================
# SECTION 7 :: THE ACTION TABLE  (REQUIRED #2)
# =============================================================================

def render_restock_table(products: pd.DataFrame) -> None:
    """'Immediate Restock Actions Required' - the operational deliverable.

    st.dataframe is INTERACTIVE out of the box (column sorting, text search,
    column resizing, CSV download) because Streamlit renders it with a
    virtualised front-end grid.  column_config adds number formatting,
    progress bars and semantic colouring without a single line of HTML.
    """
    action_queue = products[products["alert_severity"] != "OK"].copy()
    if action_queue.empty:
        st.success("No SKUs require immediate action for the selected filters. "
                   "The warehouse is fully compliant.")
        return

    action_queue = action_queue.sort_values(
        ["alert_severity", "reorder_value"], ascending=[True, False]
    )

    display = action_queue[[
        "product_id", "product_name", "category", "warehouse_zone",
        "abc_class", "alert_severity", "alert_flags", "current_stock",
        "safety_stock_level", "stock_buffer_ratio", "reorder_quantity",
        "reorder_value", "days_of_supply", "inventory_turnover_ratio_annualized",
        "revenue", "recommended_action",
    ]].rename(columns={
        "product_id": "SKU",
        "product_name": "Product",
        "category": "Category",
        "warehouse_zone": "Zone",
        "abc_class": "ABC",
        "alert_severity": "Severity",
        "alert_flags": "Alert",
        "current_stock": "On Hand",
        "safety_stock_level": "Safety Level",
        "stock_buffer_ratio": "Buffer Ratio",
        "reorder_quantity": "Reorder Qty",
        "reorder_value": "Reorder Value ($)",
        "days_of_supply": "Days of Supply",
        "inventory_turnover_ratio_annualized": "ITR (annualised)",
        "revenue": "Revenue ($)",
        "recommended_action": "Recommended Action",
    })

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=min(420, 60 + 35 * len(display)),
        column_config={
            "Severity": st.column_config.TextColumn(
                "Severity", help="CRITICAL > HIGH > MEDIUM > OK"),
            "ABC": st.column_config.TextColumn("ABC"),
            "On Hand": st.column_config.NumberColumn("On Hand", format="%d"),
            "Safety Level": st.column_config.NumberColumn("Safety Level", format="%d"),
            "Buffer Ratio": st.column_config.ProgressColumn(
                "Buffer Ratio", min_value=0.0, max_value=2.0, format="%.2f",
                help="On-hand ÷ safety level. Below 1.0 means the buffer is "
                     "already breached."),
            "Reorder Qty": st.column_config.NumberColumn("Reorder Qty", format="%d"),
            "Reorder Value ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Days of Supply": st.column_config.NumberColumn(format="%.1f"),
            "ITR (annualised)": st.column_config.NumberColumn(format="%.2f"),
            "Revenue ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_full_inventory(products: pd.DataFrame) -> None:
    """The complete 60-row analytical frame, for auditors and examiners."""
    st.dataframe(
        products.drop(columns=[c for c in ("is_dead_stock", "is_stockout_risk",
                                           "opening_stock_est", "txn_count",
                                           "cost")]),
        width="stretch", hide_index=True, height=380,
        column_config={
            "revenue": st.column_config.NumberColumn("revenue", format="$%.2f"),
            "cogs": st.column_config.NumberColumn("cogs", format="$%.2f"),
            "margin": st.column_config.NumberColumn("margin", format="$%.2f"),
            "margin_pct": st.column_config.NumberColumn("margin_pct", format="%.2f%%"),
            "price": st.column_config.NumberColumn("price", format="$%.2f"),
            "inventory_value": st.column_config.NumberColumn(
                "inventory_value", format="$%.2f"),
            "average_inventory_value": st.column_config.NumberColumn(
                "average_inventory_value", format="$%.2f"),
            "revenue_share_pct": st.column_config.NumberColumn(
                "revenue_share_pct", format="%.3f%%"),
            "cumulative_revenue_pct": st.column_config.NumberColumn(
                "cumulative_revenue_pct", format="%.3f%%"),
        },
    )


# =============================================================================
# SECTION 8 :: ARCHITECTURE / VIVA PANEL
# =============================================================================

def render_architecture_panel(meta: Dict[str, object], manifest: Dict[str, object],
                              actions: List[dict]) -> None:
    """Documentation panel: what ran, on what, and what the Spark concepts were."""
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏗️ Architecture", "⚙️ Spark actions", "🗄️ HDFS manifest", "🎓 Viva notes"]
    )

    with tab1:
        st.markdown("""
**Data flow**

| Stage | Engine | What happens |
|---|---|---|
| 1. Ingestion | Python / Pandas | 3 CSV partitions written to a simulated HDFS namespace with Hadoop `_SUCCESS` commit markers |
| 2. Read | **PySpark** | `spark.read.schema(...).csv(...)` – schema-on-read, `FAILFAST` mode, explicit `timestampFormat` |
| 3. Aggregate | **PySpark** | `groupBy(product_id)` → units, revenue, COGS (one shuffle, cached because it feeds 3 consumers) |
| 4. ABC class | **PySpark** | `Window.orderBy(revenue desc).rowsBetween(unboundedPreceding, -1)` → cumulative % → 80/15/5 cut |
| 5. Turnover | **PySpark** | COGS ÷ average inventory value, plus days-of-supply |
| 6. Alerts | **PySpark** | `LEFT ANTI JOIN` for dead stock, `<=` for stockout risk |
| 7. Unify | **PySpark** | product master as the LEFT spine, three broadcast joins |
| 8. Action | **PySpark → Pandas** | `toPandas()` (Arrow) — the *only* place data leaves the JVM |
| 9. Present | Streamlit | filter a 60-row frame, KPI cards, Plotly charts |

**Why the product master is the spine**
An inner join would silently drop the five dead-stock SKUs — the exact rows the
dashboard exists to alert on. Dimension tables drive report shape; facts never do.
""")

    with tab2:
        st.markdown("**Every ACTION executed by the pipeline** "
                    "(`toPandas`/`first` are the only calls that trigger real Spark jobs):")
        st.dataframe(pd.DataFrame(actions), width="stretch",
                     hide_index=True)
        st.caption("Lazy evaluation: the ~40 transformations above the first "
                   "action built a DAG without touching a single byte of data.")

    with tab3:
        if manifest:
            st.json(manifest, expanded=False)
        else:
            st.info("No manifest found (the dataset was generated in an earlier run).")
        st.caption("Local mock root: `{}`".format(HDFS_LOCAL_ROOT))

    with tab4:
        st.markdown("""
**The five questions to be ready for**

1. **Why is Spark used at all here?** The 2,500-row ledger is small, but the
   architecture is identical at 2.5 billion rows: the same DAG, the same
   shuffles, the same Catalyst optimisations. The code does not change when
   the data volume does — that is the whole point of the DataFrame API.
2. **Lazy evaluation?** `filter/join/groupBy` only append nodes to the lineage
   graph. Nothing executes until an action (`toPandas`, `count`, `show`,
   `write`). Proof: the console prints every action with its duration.
3. **Broadcast vs shuffle?** `F.broadcast()` on the two 60-row tables avoids
   shuffling them. The one unavoidable shuffle is the `groupBy`.
4. **How do you find dead stock?** `LEFT ANTI JOIN` — a single hash anti-join,
   versus the `COUNT(*) = 0` anti-pattern which forces two extra aggregations.
5. **What is ABC analysis and where is the 80% line?** A windowed running
   total of revenue, cut at 80% and 95%. The SKU that *pushes* the cumulative
   total across a threshold stays in the higher class — we use an **exclusive**
   frame (`rowsBetween(unboundedPreceding, -1)`) to get the "before" value.
""")
        if meta.get("actions"):
            total_actions = len(meta["actions"])
            total_secs = sum(a["seconds"] for a in meta["actions"])
            st.info("Pipeline executed **{} Spark actions** totalling "
                    "**{:.2f}s** of cluster work.".format(total_actions, total_secs))


# =============================================================================
# SECTION 9 :: MAIN  (Streamlit executes top-to-bottom on every interaction)
# =============================================================================

def main() -> None:
    manifest = read_manifest()

    # ---- 1. LOAD (or bootstrap the HDFS layer, then run Spark) -----------
    try:
        results = load_pipeline()
    except Exception as exc:                       # pragma: no cover
        st.error("The PySpark pipeline failed to start: {}".format(exc))
        st.exception(exc)
        st.stop()

    products_all: pd.DataFrame = results["products"]
    meta: Dict[str, object] = results["meta"]
    kpis: Dict[str, object] = results["kpis"]

    # ---- 2. HERO HEADER --------------------------------------------------
    ingestion = meta.get("ingestion", {}) or {}
    st.markdown("""
    <div class="smartstock-hero">
      <h1>📦 SmartStock: HDFS &amp; PySpark Enterprise Analytics</h1>
      <p>60 SKUs · 2,500 ledger transactions · 5 warehouse zones —
         a <b>Simulated HDFS RAW layer</b> is parsed with explicit
         <b>StructType</b> schemas and analysed by a <b>PySpark</b> pipeline
         (lazy DAG → window functions → broadcast joins) before being safely
         converted to Pandas for this Streamlit console.</p>
      <div class="hero-pills">
        <span class="hero-pill">🗄️ HDFS RAW layer</span>
        <span class="hero-pill">⚡ PySpark {}</span>
        <span class="hero-pill">📊 {} transactions</span>
        <span class="hero-pill">🗓️ {} day window</span>
        <span class="hero-pill">🐍 Python {}</span>
      </div>
    </div>
    """.format(
        meta.get("spark_version"), fmt_number(ingestion.get("sales_ledger_rows")),
        meta.get("sales_window_days"), "{}.{}.{}".format(*sys.version_info[:3]),
    ), unsafe_allow_html=True)

    # ---- 3. SIDEBAR FILTERS + SLICE --------------------------------------
    filters = render_sidebar(products_all, meta)
    products = apply_filters(products_all, filters)
    slice_is_empty = products.empty

    # Chart-ready rollups.  When nothing is filtered we use the aggregates
    # Spark already computed (the "no extra work" fast path); once the user
    # slices, we recompute the tiny rollups in Pandas for instant feedback.
    filters_active = any([
        filters["categories"], filters["classes"], filters["zones"],
        filters["severities"], filters["min_revenue"], filters["only_alerts"],
    ])
    if filters_active and not slice_is_empty:
        categories, abc, totals = rollup_slice(products)
    elif slice_is_empty:
        categories = pd.DataFrame(columns=["category", "revenue", "sku_count"])
        abc = pd.DataFrame(columns=["abc_class", "abc_class_label", "sku_count"])
        totals = {"revenue": 0.0, "cogs": 0.0, "inventory_value": 0.0,
                  "average_inventory_value": 0.0, "stockout": 0, "dead": 0}
    else:
        categories, abc, totals = results["categories"], results["abc_summary"], {
            "revenue": float(kpis["total_revenue"]),
            "cogs": float(kpis["total_cogs"]),
            "inventory_value": float(kpis["total_inventory_value"]),
            "average_inventory_value": float(kpis["total_average_inventory_value"]),
            "stockout": int(kpis["stockout_risk_items"]),
            "dead": int(kpis["dead_stock_items"]),
        }

    if slice_is_empty:
        st.warning("No SKUs match the current filters. Widen the selection in "
                    "the sidebar (or press *Reset all filters*).")

    # ---- 4. KPI ROW -------------------------------------------------------
    render_kpi_row(totals, kpis, len(products), len(products_all), filters_active)

    if not slice_is_empty:
        # ---- 5. VISUAL ANALYTICS ------------------------------------------
        section_head("📊", "Revenue by Product Category",
                     "Horizontal bar chart — every dollar generated per "
                     "category over the analysis window.")
        chart_revenue_by_category(categories)

        left, right = st.columns(2, gap="medium")
        with left:
            section_head("🧭", "ABC Inventory Distribution",
                         "Share of SKUs in each value class (80 / 15 / 5 rule).")
            chart_abc_distribution(abc)
        with right:
            section_head("🚨", "Alert Severity Mix",
                         "How the current selection splits by urgency.")
            chart_severity_donut(products)

        section_head("📈", "Pareto Curve — Revenue Concentration",
                     "The Spark window function made visual: the dashed lines "
                     "mark the 80% and 95% ABC cut-offs.")
        chart_pareto_curve(products, window_days=int(meta.get("sales_window_days") or 90))

        # ---- 6. ACTION TABLE ----------------------------------------------
        section_head("🛠️", "Immediate Restock Actions Required",
                     "Prioritised by severity, then by capital at risk. "
                     "Click a column header to sort, or download as CSV.")
        render_restock_table(products)

        # ---- 7. FULL INVENTORY (collapsible) ------------------------------
        with st.expander("📋 Full analytical inventory frame (all {} SKUs in "
                         "scope)".format(len(products)), expanded=False):
            render_full_inventory(products)

    # ---- 8. ARCHITECTURE / VIVA PANEL ------------------------------------
    st.markdown("---")
    render_architecture_panel(meta, manifest, meta.get("actions", []))

    st.markdown(
        '<div class="footer">SmartStock: HDFS &amp; PySpark Inventory Analytics '
        'Suite · built with Streamlit {} · Plotly · PySpark {} · '
        'simulated HDFS namespace at <code>{}</code></div>'.format(
            st.__version__, meta.get("spark_version"), HDFS_LOCAL_ROOT),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
