#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SMARTSTOCK :: INVENTORY CONTROL TOWER   (main_app.py)
===============================================================================

An operations console for supply-chain planners. It answers three questions a
planner actually asks each morning, in order of cost:

    1. WHERE IS THE MONEY LEAKING?      net exposure ($) from open alerts
    2. WHAT MUST I DO TODAY?           a cost-ranked action queue
    3. WHICH PARAMETERS ARE WRONG?      safety stock set from demand volatility

DATA FLOW
    HDFS RAW layer (simulated, _SUCCESS committed)
      -> auto-bootstrap if not committed
      -> PySparkInventoryEngine: schema-on-read, groupBy, ABC window, turnover,
         LEFT ANTI JOIN, demand-volatility stats, economic impact, toPandas()
      -> st.cache_resource: the SparkSession runs ONCE and is reused, so every
         interaction costs milliseconds instead of a ~14 s Spark job
      -> this console

WHERE THE WORK HAPPENS
    Spark   heavy work: 2,500 typed rows parsed, the groupBy shuffle, the
            Pareto window, the daily-demand re-aggregation, broadcast joins.
    Driver  presentation: filtering 60 aggregated rows and drawing charts.
    The split is deliberate. A Pandas frame is single-process, so shipping raw
    events to the UI per slider move is wasteful, while shipping per-event
    aggregates would be a design error. We ship 60.

DESIGN PRINCIPLES
    * Show the number, not the decoration. A planner's screen is dense, not
      glossy: no gradients on data, colour reserved for STATUS only.
    * Never publish a confident wrong number. Where the data cannot support a
      claim (see demand censoring in the engine), the UI says so.
    * Every headline figure must be traceable to a Spark action.
===============================================================================
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from hdfs_storage_mock import (
    HDFS_LOCAL_ROOT,
    ensure_hdfs_dataset,
    hdfs_dataset_exists,
    read_manifest,
)
from pyspark_analytics_engine import PySparkInventoryEngine

# =============================================================================
# SECTION 1 :: PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SmartStock | Inventory Control Tower",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
# A deliberately small, semantic palette. Neutral greys do the structural work;
# colour is reserved for STATUS (critical / warning / ok / neutral) so that a
# red cell always means "act on this" and never merely "this is a header".
INK = "#10161c"          # primary text            17.4:1 on white
INK_MUTED = "#4a5c6a"    # secondary text          7.4:1 on white
INK_FAINT = "#4a5c6a"    # tertiary / axis labels  7.4:1 on white
RULE = "#d8e0e8"         # hairlines
PANEL = "#ffffff"
CANVAS_TOP = "#f4f6f9"
CANVAS_BOTTOM = "#e9eef3"
NAVY = "#0f2a3f"         # app bar                 15.2:1 with white text
NAVY_SOFT = "#17395a"

STATUS = {
    "critical": "#b3261e",   # 6.5:1  -- act now
    "warning": "#8a5a00",    # 5.9:1  -- act soon
    "ok": "#0f6b4f",         # 6.0:1  -- healthy
    "info": "#1d5f8a",       # 6.1:1  -- informational
    "muted": "#5c6f7d",      # 5.3:1  -- inactive
}
#: Class A/B/C are an ORDINAL scale, so they get an ordered blue ramp rather
#: than three unrelated hues (which read as three different meanings).
ABC_COLORS = {"A": "#0f3f63", "B": "#2f7ba8", "C": "#8fb4cc"}
SEVERITY_STATUS = {
    "CRITICAL": "critical", "HIGH": "critical",
    "MEDIUM": "warning", "OK": "ok",
}
RISK_STATUS = {
    "Censored": "critical", "Under-Buffered": "critical",
    "Over-Buffered": "warning", "Aligned": "ok", "No Demand": "muted",
}
CHART_TEXT = "#10161c"
CHART_GRID = "#e4eaf0"
ACCENT = "#1d5f8a"

#: Default ("no filter") value of every sidebar widget. Kept in one place so the
#: reset button can never drift out of sync with the widgets themselves.
FILTER_DEFAULTS: Dict[str, object] = {
    "filter_category": [],
    "filter_abc_class": [],
    "filter_zone": [],
    "filter_severity": [],
    "filter_min_revenue": 0,
    "filter_only_alerts": False,
}


def _reset_filters() -> None:
    """on_click callback for the reset button (see the call site)."""
    st.session_state.update(FILTER_DEFAULTS)


# =============================================================================
# SECTION 2 :: STYLESHEET
# =============================================================================
# Every text/background pair below clears WCAG AA (4.5:1); most are AAA.
# The theme is also pinned in .streamlit/config.toml, but a user can flip the
# theme from the Streamlit menu, so the surface and text colour are forced here
# rather than inherited.

CSS = """
<style>
/* ---- Theme lock: Streamlit follows the OS dark mode by default, which paints
   body text near-white and would make this light console unreadable. ---- */
.stApp, [data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"], [data-testid="stMain"] > div {
    background: linear-gradient(180deg, #f4f6f9 0%, #e9eef3 100%) !important;
    color: #10161c !important;
}
[data-testid="stMain"] p, [data-testid="stMain"] li,
[data-testid="stMain"] td, [data-testid="stMain"] th,
[data-testid="stMain"] span, [data-testid="stMain"] label,
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] summary { color: #10161c; }
[data-testid="stMain"] hr { border-color: #d8e0e8; }
[data-testid="stCaptionContainer"] { color: #4a5c6a; }
[data-testid="stTabs"] [role="tab"] { color: #4a5c6a; font-weight: 600; }
[data-testid="stTabs"] [aria-selected="true"] { color: #0f2a3f; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: #1d5f8a; }
.block-container { padding-top: 0; padding-bottom: 2.4rem; max-width: 1560px; }

/* ---- App bar ------------------------------------------------------------ */
/* margin-top clears Streamlit's own fixed header (the toolbar with the Deploy
   button), which otherwise overlaps the first line of the bar. */
.appbar {
    background: #0f2a3f; color: #ffffff; padding: .7rem 1.1rem;
    border-radius: 8px; margin: 2.9rem 0 .85rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap;
}
.appbar .brand { font-size: 1.02rem; font-weight: 700; letter-spacing: -.2px;
                 display: flex; align-items: center; gap: .5rem;
                 white-space: nowrap; }
.appbar .brand .mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 6px; background: #1d5f8a;
    font-size: .82rem; font-weight: 800; flex: 0 0 auto;
}
/* nowrap keeps the bar to a single row, so its height -- and therefore the
   alignment of everything below it -- does not change between renders. */
.appbar .meta { font-size: .74rem; color: #cfe0ec; display: flex; gap: 1.1rem;
                flex-wrap: nowrap; white-space: nowrap; }
.appbar .meta b { color: #ffffff; font-weight: 600; }

/* ---- Section headers ---------------------------------------------------- */
.sec { display: flex; align-items: baseline; gap: .5rem;
       margin: 1.35rem 0 .15rem; }
.sec .num { font-size: .68rem; font-weight: 700; color: #1d5f8a;
            border: 1px solid #c3d4e0; background: #eaf1f7;
            border-radius: 4px; padding: .06rem .34rem; }
.sec h3 { font-size: .98rem; font-weight: 700; color: #0f2a3f;
          margin: 0; letter-spacing: -.15px; }
.sec .hint { font-size: .74rem; color: #4a5c6a; margin: .1rem 0 .7rem 0; }

/* ---- Metric tiles ------------------------------------------------------- */
/* Flat cards, hairline border, no drop shadows: this is a control surface, not
   a marketing page. A 3px top rule carries the status colour. */
.tile { background: #ffffff; border: 1px solid #d8e0e8; border-radius: 8px;
        padding: .78rem .9rem .72rem; height: 100%;
        border-top: 3px solid var(--tone, #1d5f8a); }
/* word-break:keep-all is what stops "EXPOSURE" being split as "EXPOSUR E" in
   a narrow column. hyphenation is off for the same reason. The reserved height
   keeps all five values on one baseline regardless of label length. */
.tile .lbl { font-size: .62rem; font-weight: 700; letter-spacing: .05em;
             text-transform: uppercase; color: #4a5c6a; line-height: 1.32;
             min-height: 3.3em; word-break: keep-all; hyphens: none;
             overflow-wrap: normal; }
/* clamp() lets the figure shrink on a narrow screen instead of clipping:
   a truncated "$235.1K" is worse than a slightly smaller one. */
.tile .val { font-size: clamp(1.15rem, 1.9vw, 1.5rem); font-weight: 700;
             color: #0f2a3f; margin: .22rem 0 .1rem; line-height: 1.05;
             letter-spacing: -.4px; font-variant-numeric: tabular-nums;
             white-space: nowrap; }
.tile .sub { font-size: .69rem; font-weight: 600; line-height: 1.4;
             font-variant-numeric: tabular-nums; }
/* #4a5c6a (7.4:1) rather than the old #6b7f8f, which measured 4.15:1 here. */
.tile .foot { font-size: .64rem; color: #4a5c6a; line-height: 1.4;
              margin-top: .38rem; padding-top: .34rem;
              border-top: 1px dashed #dfe6ed;
              font-variant-numeric: tabular-nums; }
.t-critical { color: #b3261e; } .t-warning { color: #8a5a00; }
.t-ok { color: #0f6b4f; } .t-info { color: #1d5f8a; } .t-muted { color: #5c6f7d; }

/* ---- Status chips (st.markdown only -- see the chip() docstring) --------- */
.chip { display: inline-block; font-size: .64rem; font-weight: 700;
        letter-spacing: .05em; text-transform: uppercase;
        padding: .1rem .4rem; border-radius: 4px; white-space: nowrap; }
.chip-critical { background: #fbeae9; color: #8c1d16; border: 1px solid #f0c4c0; }
.chip-warning  { background: #fdf3e0; color: #6d4700; border: 1px solid #edd7a8; }
.chip-ok       { background: #e6f4ee; color: #0b513b; border: 1px solid #b6ddcd; }
.chip-info     { background: #e8f1f7; color: #14486a; border: 1px solid #bcd8e8; }
.chip-muted    { background: #eef2f5; color: #4a5c6a; border: 1px solid #d5dee6; }
.chip-A { background: #e4ecf3; color: #0f2a3f; border: 1px solid #b9cddd; }
.chip-B { background: #e9f1f7; color: #1b4c6e; border: 1px solid #c6dbe9; }
.chip-C { background: #f1f5f8; color: #4a6274; border: 1px solid #d3dfe8; }

/* ---- Sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] { background: #0f2a3f; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: #e6eef4 !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #7fb8dc !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
[data-testid="stSidebar"] button {
    color: #e6eef4 !important; background: rgba(255,255,255,.08) !important;
    border: 1px solid rgba(255,255,255,.24) !important; font-weight: 600; }
[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,.16) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.08) !important;
    border-color: rgba(255,255,255,.24) !important; }
/* The placeholder ("All categories") is the only text on screen before the
   user interacts, so it must be legible against navy. Streamlit renders it as
   a plain DIV carrying the light theme's near-black text plus its own 0.6
   alpha -- hence both the colour AND the opacity override, on every
   descendant rather than just on `input`. */
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color: #cfe0ec !important; opacity: 1 !important; }
/* The dropdown arrow is an SVG mask, so it needs re-colouring too. */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #cfe0ec !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(255,255,255,.20) !important; color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] svg { fill: #ffffff !important; }
/* st.toggle in this build renders its label outside the usual label element,
   so target the toggle block itself rather than only its inner paragraph. */
[data-testid="stSidebar"] [data-testid="stToggle"],
[data-testid="stSidebar"] [data-testid="stToggle"] p,
[data-testid="stSidebar"] [data-testid="stToggle"] span,
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #e6eef4 !important; }
/* Slider: Streamlit's default thumb/track are tuned for a light sidebar and
   were close to invisible on navy. */
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: #7fb8dc !important; box-shadow: 0 0 0 2px #0f2a3f !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #9dbdd4 !important; }
.side-title { font-size: 1.05rem; font-weight: 700; color: #ffffff;
              letter-spacing: -.2px; }
.side-sub { font-size: .7rem; color: #9dbdd4; margin-top: .1rem; }
.side-h { font-size: .63rem; font-weight: 700; letter-spacing: .09em;
          text-transform: uppercase; color: #7fb8dc; margin: .3rem 0 .1rem; }
.kv { font-size: .73rem; line-height: 1.75; color: #cfe0ec;
      font-variant-numeric: tabular-nums; }
.kv b { color: #ffffff; font-weight: 600; }

/* ---- Misc --------------------------------------------------------------- */
.dataframe-note { font-size: .68rem; color: #4a5c6a; margin: .3rem 0 0; }
.callout { background: #ffffff; border: 1px solid #d8e0e8;
           border-left: 3px solid #1d5f8a; border-radius: 6px;
           padding: .7rem .85rem; font-size: .78rem; color: #10161c;
           line-height: 1.55; margin: .4rem 0; }
.callout.warn { border-left-color: #8a5a00; }
.callout.crit { border-left-color: #b3261e; }
.callout .ct { font-weight: 700; display: block; margin-bottom: .2rem;
               color: #0f2a3f; }
code { background: #eef2f6 !important; color: #0f2a3f !important;
       border: 1px solid #dbe3ea; border-radius: 4px;
       padding: .04em .28em; }
[data-testid="stMain"] table th { color: #0f2a3f; background: #f4f7fa; }
[data-testid="stMain"] table td { color: #10161c; border-color: #e4eaf0; }
div[data-testid="stExpander"] details {
    border: 1px solid #d8e0e8; border-radius: 8px; background: #ffffff; }
div[data-testid="stExpander"] summary { color: #0f2a3f !important; font-weight: 600; }
div[data-testid="stExpander"] summary p { color: #0f2a3f !important; }
.foot-note { text-align: center; color: #4a5c6a; font-size: .68rem;
             margin-top: 1.6rem; padding-top: .7rem;
             border-top: 1px solid #d8e0e8; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# SECTION 3 :: FORMATTING
# =============================================================================

def money(value: float, decimals: int = 0) -> str:
    try:
        if value is None or pd.isna(value):
            return "--"
        return "${:,.{}f}".format(float(value), decimals)
    except (TypeError, ValueError):
        return "--"


def num(value: float, decimals: int = 0) -> str:
    try:
        if value is None or pd.isna(value):
            return "--"
        return "{:,.{}f}".format(float(value), decimals)
    except (TypeError, ValueError):
        return "--"


def compact_money(value: float) -> str:
    """$1.2M / $43.8K -- for headline tiles where width is tight."""
    try:
        if value is None or pd.isna(value):
            return "--"
        value = float(value)
        if abs(value) >= 1_000_000:
            return "${:,.2f}M".format(value / 1_000_000)
        if abs(value) >= 1_000:
            return "${:,.1f}K".format(value / 1_000)
        return "${:,.0f}".format(value)
    except (TypeError, ValueError):
        return "--"


def tile(label: str, value: str, sub: str = "", foot: str = "",
         tone: str = "info") -> None:
    """One KPI tile. `tone` maps to a status colour on the top rule + sub-text."""
    colour = STATUS.get(tone, STATUS["info"])
    st.markdown(
        """
        <div class="tile" style="--tone:{colour};">
          <div class="lbl">{label}</div>
          <div class="val">{value}</div>
          <div class="sub t-{tone}">{sub}</div>
          <div class="foot">{foot}</div>
        </div>
        """.format(colour=colour, tone=tone, label=label, value=value,
                   sub=sub or "&nbsp;", foot=foot or "&nbsp;"),
        unsafe_allow_html=True,
    )


def section(number: str, title: str, hint: str = "") -> None:
    st.markdown(
        '<div class="sec"><span class="num">{}</span><h3>{}</h3></div>'
        '<div class="hint">{}</div>'.format(number, title, hint),
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "muted") -> str:
    """Status chip as HTML.

    NOTE: only valid inside st.markdown. st.dataframe renders through a
    sanitising grid, which strips the markup -- table cells therefore use a
    plain-text glyph prefix instead. Keeping both paths documented avoids
    someone "fixing" a table that silently loses its colour.
    """
    return '<span class="chip chip-{}">{}</span>'.format(kind, text)


# =============================================================================
# SECTION 4 :: DATA BOOTSTRAP + SPARK PIPELINE (cached)
# =============================================================================

def bootstrap_hdfs_layer() -> Dict[str, object]:
    """Ensure the simulated HDFS RAW layer is committed before Spark reads it.

    `hdfs_dataset_exists()` checks both the data files and the Hadoop
    `_SUCCESS` markers, so we never read a half-written dataset.
    """
    if not hdfs_dataset_exists():
        with st.spinner("HDFS RAW layer empty -- generating the simulated "
                        "dataset (products / stock / sales ledger) ..."):
            return ensure_hdfs_dataset()
    return read_manifest()


@st.cache_resource(show_spinner="Running the PySpark pipeline "
                               "(first run also starts the Spark JVM) ...")
def load_pipeline() -> Dict[str, object]:
    """Execute the Spark pipeline ONCE and cache it for the session.

    st.cache_resource (not st.cache_data) keeps the engine instance -- and so
    its SparkContext/JVM -- alive between reruns, which is why filtering is
    instant rather than costing a fresh ~14 s Spark job.
    """
    bootstrap_hdfs_layer()
    engine = PySparkInventoryEngine(verbose=True)
    results = engine.run_pipeline()
    results["engine"] = engine
    return results


def rollup(products: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Recompute the category / ABC rollups for the CURRENTLY FILTERED slice.

    The slice is at most 60 rows -- effectively a lookup table -- so rolling it
    up in Pandas costs microseconds. Re-submitting a Spark job per slider move
    would add seconds of scheduling latency for no benefit. The rule:
    distributed engine for volume, in-memory engine for interactivity.
    """
    categories = (
        products.groupby("category", as_index=False)
        .agg(sku_count=("product_id", "count"),
             revenue=("revenue", "sum"),
             cogs=("cogs", "sum"),
             units_sold=("units_sold", "sum"),
             inventory_value=("inventory_value", "sum"),
             net_exposure=("net_exposure", "sum"),
             dead_stock_items=("is_dead_stock", "sum"),
             stockout_items=("is_stockout_risk", "sum"))
        .sort_values("revenue", ascending=False)
    )
    abc = (
        products.groupby(["abc_class", "abc_class_label"], as_index=False)
        .agg(sku_count=("product_id", "count"),
             revenue=("revenue", "sum"),
             inventory_value=("inventory_value", "sum"),
             net_exposure=("net_exposure", "sum"))
        .sort_values("abc_class")
    )
    totals = {
        "revenue": float(products["revenue"].sum()),
        "cogs": float(products["cogs"].sum()),
        "margin": float(products["margin"].sum()),
        "inventory_value": float(products["inventory_value"].sum()),
        "avg_inventory_value": float(products["average_inventory_value"].sum()),
        "net_exposure": float(products["net_exposure"].sum()),
        "lost_margin": float(products["lost_margin_estimate"].sum()),
        "lost_revenue": float(products["lost_revenue_estimate"].sum()),
        "trapped": float(products["trapped_capital"].sum()),
        "reorder_value": float(products["reorder_value"].sum()),
        "stockout": int(products["is_stockout_risk"].sum()),
        "dead": int(products["is_dead_stock"].sum()),
        "under_buffered": int((products["safety_stock_risk"] == "Under-Buffered").sum()),
        "over_buffered": int((products["safety_stock_risk"] == "Over-Buffered").sum()),
        "censored": int((products["safety_stock_risk"] == "Censored").sum()),
        "avg_cv": float(products["demand_cv"].mean()) if len(products) else 0.0,
    }
    return categories, abc, totals


# =============================================================================
# SECTION 5 :: SIDEBAR
# =============================================================================

def render_sidebar(products: pd.DataFrame, meta: Dict[str, object]
                   ) -> Dict[str, object]:
    st.sidebar.markdown(
        '<div class="side-title">▦ SmartStock</div>'
        '<div class="side-sub">Inventory Control Tower &middot; HDFS × PySpark</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown('<div class="side-h">Filters</div>', unsafe_allow_html=True)
    selected_categories = st.sidebar.multiselect(
        "Category", options=sorted(products["category"].unique()),
        default=[], placeholder="All categories", key="filter_category")
    selected_classes = st.sidebar.multiselect(
        "ABC class", options=["A", "B", "C"], default=[],
        placeholder="All classes", key="filter_abc_class")
    selected_zones = st.sidebar.multiselect(
        "Warehouse zone", options=sorted(products["warehouse_zone"].dropna().unique()),
        default=[], placeholder="All zones", key="filter_zone")
    selected_severities = st.sidebar.multiselect(
        "Alert severity", options=["CRITICAL", "HIGH", "MEDIUM", "OK"],
        default=[], placeholder="All severities", key="filter_severity")
    min_revenue = st.sidebar.slider(
        "Minimum SKU revenue", min_value=0,
        max_value=int(max(products["revenue"].max(), 1)),
        value=0, step=500, key="filter_min_revenue")
    only_alerts = st.sidebar.toggle(
        "Flagged SKUs only", value=False,
        help="Hides SKUs whose alert severity is OK.", key="filter_only_alerts")

    # on_click (not an inline assignment) because a callback runs BEFORE the
    # widgets are re-instantiated, which is the only legal moment to write to
    # a widget's session state.
    st.sidebar.button("Reset all filters", on_click=_reset_filters,
                      width="stretch")

    # ---------------- data freshness / provenance ----------------
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="side-h">Pipeline</div>', unsafe_allow_html=True)
    ingestion = meta.get("ingestion", {}) or {}
    st.sidebar.markdown(
        '<div class="kv">'
        '<b>Spark</b> {}<br>'
        '<b>Engine</b> {}<br>'
        '<b>Shuffle parts</b> {}<br>'
        '<b>Window</b> {} days<br>'
        '<b>Events read</b> {}<br>'
        '<b>Actions</b> {}<br>'
        '<b>Wall clock</b> {}s'
        '</div>'.format(
            meta.get("spark_version"), meta.get("spark_master"),
            meta.get("shuffle_partitions"), meta.get("sales_window_days"),
            num(ingestion.get("sales_ledger_rows")),
            len(meta.get("actions", []) or []),
            meta.get("total_seconds"),
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="side-h">Actions</div>', unsafe_allow_html=True)
    if st.sidebar.button("Regenerate HDFS dataset", width="stretch"):
        with st.spinner("Regenerating the simulated HDFS RAW layer ..."):
            ensure_hdfs_dataset(force=True)
        # Invalidation order matters: write the data first, then drop the cache.
        st.cache_resource.clear()
        st.rerun()

    return {
        "categories": selected_categories, "classes": selected_classes,
        "zones": selected_zones, "severities": selected_severities,
        "min_revenue": min_revenue, "only_alerts": only_alerts,
    }


def apply_filters(products: pd.DataFrame, f: Dict[str, object]) -> pd.DataFrame:
    """Slice the 60-row frame. An empty selection means 'no filter'."""
    mask = pd.Series(True, index=products.index)
    if f["categories"]:
        mask &= products["category"].isin(f["categories"])
    if f["classes"]:
        mask &= products["abc_class"].isin(f["classes"])
    if f["zones"]:
        mask &= products["warehouse_zone"].isin(f["zones"])
    if f["severities"]:
        mask &= products["alert_severity"].isin(f["severities"])
    if f["min_revenue"]:
        mask &= products["revenue"] >= float(f["min_revenue"])
    if f["only_alerts"]:
        mask &= products["alert_severity"] != "OK"
    return products[mask].copy()


# =============================================================================
# SECTION 6 :: CHART HELPERS
# =============================================================================

def _base_layout(height: int, **kwargs) -> dict:
    layout = dict(
        height=height, margin=dict(t=12, b=12, l=8, r=8),
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif",
                  size=12, color=CHART_TEXT),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
    )
    layout.update(kwargs)
    return layout


def chart_revenue_by_category(categories: pd.DataFrame) -> None:
    """Horizontal bars: revenue by category. Ordered ascending so the largest
    category sits at the top, which is how every finance deck reads a bar chart."""
    ordered = categories.sort_values("revenue", ascending=True)
    colours = [ABC_COLORS["A"]] * len(ordered)
    figure = go.Figure(go.Bar(
        x=ordered["revenue"], y=ordered["category"], orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[compact_money(v) for v in ordered["revenue"]],
        textposition="outside", textfont=dict(size=11, color=CHART_TEXT),
        customdata=np.c_[ordered["sku_count"], ordered["units_sold"],
                         ordered["inventory_value"], ordered["net_exposure"]],
        hovertemplate=(
            "<b>%{y}</b><br>Revenue $%{x:,.0f}<br>SKUs %{customdata[0]}"
            "<br>Units sold %{customdata[1]:,.0f}"
            "<br>Inventory $%{customdata[2]:,.0f}"
            "<br>Net exposure $%{customdata[3]:,.0f}<extra></extra>"),
        showlegend=False,
    ))
    figure.update_layout(**_base_layout(
        330, xaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                        tickprefix="$", separatethousands=True),
        yaxis=dict(tickfont=dict(color=CHART_TEXT), showgrid=False),
        bargap=0.35))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def chart_exposure_by_sku(products: pd.DataFrame) -> None:
    """The money chart: net exposure by SKU, split into lost margin vs trapped
    capital. A stacked bar makes the two failure modes comparable at a glance
    -- they need opposite decisions (buy more vs liquidate)."""
    flagged = products[products["net_exposure"] > 0].copy()
    if flagged.empty:
        st.success("No economic exposure in the current selection.")
        return
    flagged = flagged.sort_values("net_exposure").tail(12)
    figure = go.Figure()
    figure.add_bar(
        y=flagged["product_id"], x=flagged["lost_margin_estimate"],
        name="Lost margin (stockout)", orientation="h",
        marker=dict(color=STATUS["critical"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Lost margin $%{x:,.0f}<extra></extra>")
    figure.add_bar(
        y=flagged["product_id"], x=flagged["trapped_capital"],
        name="Trapped capital (dead stock)", orientation="h",
        marker=dict(color="#c9a227", line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Trapped capital $%{x:,.0f}<extra></extra>")
    figure.update_layout(**_base_layout(
        330, barmode="stack", showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0,
                    font=dict(size=10.5, color=CHART_TEXT)),
        xaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                   tickprefix="$", separatethousands=True,
                   title=dict(text="Net exposure ($)",
                              font=dict(color=CHART_TEXT))),
        # Title colour set explicitly: Plotly axis titles do NOT inherit the
        # layout font, so they fall back to a mid-grey that measures 3.7:1.
        yaxis=dict(tickfont=dict(color=CHART_TEXT), showgrid=False,
                   title=dict(text="SKU", font=dict(color=CHART_TEXT)))))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def chart_buffer_gap(products: pd.DataFrame) -> None:
    """Diverging bars: configured safety stock vs the statistical requirement.

    Bars left of zero are over-buffered (cash sitting on the shelf); bars right
    of zero are under-buffered (stockout risk). NULL gaps (demand-censored
    SKUs) are deliberately excluded -- see the callout under the chart.
    """
    valid = products[products["safety_stock_gap"].notna()].copy()
    censored = int((products["safety_stock_risk"] == "Censored").sum())
    if valid.empty:
        st.info("No SKUs with a usable demand signal in this selection.")
        return
    valid = valid.sort_values("safety_stock_gap").tail(14)
    colours = [STATUS["critical"] if g > 0 else "#4a5c6a"
               for g in valid["safety_stock_gap"]]
    figure = go.Figure(go.Bar(
        x=valid["safety_stock_gap"], y=valid["product_id"], orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[num(v) for v in valid["safety_stock_gap"]],
        textposition="outside", textfont=dict(size=10.5, color=CHART_TEXT),
        customdata=np.c_[valid["safety_stock_level"],
                         valid["recommended_safety_stock"],
                         valid["demand_cv"]],
        hovertemplate=(
            "<b>%{y}</b><br>Gap %{x:,.0f} units"
            "<br>Configured %{customdata[0]:,.0f}"
            "<br>Recommended %{customdata[1]:,.0f}"
            "<br>Demand CV %{customdata[2]:.2f}<extra></extra>"),
        showlegend=False))
    figure.add_vline(x=0, line_width=1.4, line_color="#10161c")
    figure.update_layout(**_base_layout(
        330, xaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                        title=dict(text="Safety stock gap (units)  "
                                         "[over-buffered | under-buffered]",
                                   font=dict(color=CHART_TEXT))),
        yaxis=dict(tickfont=dict(color=CHART_TEXT), showgrid=False,
                   title=dict(text="SKU", font=dict(color=CHART_TEXT))),
        bargap=0.35))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    if censored:
        st.markdown(
            '<div class="callout warn"><span class="ct">Demand censoring</span>'
            '{} SKU(s) in this selection are currently at or below their safety '
            'level. While out of stock they record no demand at all, so their '
            'observed mean is biased low and the statistical recommendation is '
            'unreliable. Those SKUs are reported as <b>Censored</b> and excluded '
            'from this chart rather than given a confident but wrong number.'
            '</div>'.format(censored),
            unsafe_allow_html=True)


def chart_volatility_vs_revenue(products: pd.DataFrame) -> None:
    """Scatter: demand volatility (CV) against revenue contribution.

    Colour = ABC class, marker size = current stock. The interesting population
    is the upper-right: high-value SKUs whose demand is also erratic, because
    those are the ones a static safety level cannot protect.
    """
    live = products[products["daily_demand_mean"] > 0].copy()
    if live.empty:
        st.info("No SKUs with demand in this selection.")
        return
    figure = px.scatter(
        live, x="demand_cv", y="revenue",
        color="abc_class", size="current_stock", size_max=34,
        color_discrete_map=ABC_COLORS,
        # "ABC" as an axis title is set explicitly in the layout below, because
        # Plotly legend/axis titles ignore the layout font colour.
        labels={"demand_cv": "Demand volatility (coefficient of variation)",
                "revenue": "Revenue ($)", "abc_class": "",
                "current_stock": "On hand"},
        custom_data=["product_id", "product_name", "safety_stock_risk",
                     "safety_stock_gap"],
    )
    figure.update_traces(
        marker=dict(opacity=0.82, line=dict(width=1, color="#ffffff")),
        hovertemplate=(
            "<b>%{customdata[0]}</b> %{customdata[1]}"
            "<br>ABC %{color} &middot; CV %{x:.2f}"
            "<br>Revenue $%{y:,.0f}"
            "<br>Buffer risk %{customdata[2]}<extra></extra>"))
    figure.update_layout(**_base_layout(
        330, showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0,
                    font=dict(size=10.5, color=CHART_TEXT)),
        xaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                   title=dict(text="Demand volatility (CV) — less predictable",
                              font=dict(color=CHART_TEXT))),
        yaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                   tickprefix="$", separatethousands=True,
                   title=dict(text="Revenue ($) — more valuable",
                              font=dict(color=CHART_TEXT)))))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def chart_abc_distribution(abc: pd.DataFrame) -> None:
    """Donut: SKU share by ABC class. The planning rule of thumb behind it is
    that class A deserves the management attention and class C the shelf space."""
    figure = go.Figure(go.Pie(
        labels=abc["abc_class_label"], values=abc["sku_count"],
        hole=0.56, sort=False, direction="clockwise",
        marker=dict(colors=[ABC_COLORS.get(c, "#8395a7") for c in abc["abc_class"]],
                    line=dict(color="#ffffff", width=2)),
        textinfo="label+percent", textfont=dict(size=11.5, color=CHART_TEXT),
        hovertemplate="<b>%{label}</b><br>SKUs %{value}"
                      "<br>%{percent} of catalogue<extra></extra>"))
    figure.add_annotation(
        text="<b>{}</b><br>SKUs".format(int(abc["sku_count"].sum())),
        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#0f2a3f"))
    figure.update_layout(**_base_layout(
        330, margin=dict(t=10, b=10), showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                    font=dict(size=10.5, color=CHART_TEXT))))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def chart_turnover_by_category(categories: pd.DataFrame) -> None:
    """Grouped bars: inventory value against revenue, per category. A category
    holding a lot of stock but generating little revenue is where capital is
    being trapped, and this is how a planner spots it in one glance."""
    ordered = categories.sort_values("inventory_value", ascending=False)
    figure = go.Figure()
    figure.add_bar(
        name="Inventory value at cost", x=ordered["category"],
        y=ordered["inventory_value"], marker=dict(color="#c9d6e0",
                                                   line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Inventory $%{y:,.0f}<extra></extra>")
    figure.add_bar(
        name="Revenue", x=ordered["category"], y=ordered["revenue"],
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Revenue $%{y:,.0f}<extra></extra>")
    figure.update_layout(**_base_layout(
        330, barmode="group", showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0,
                    font=dict(size=10.5, color=CHART_TEXT)),
        xaxis=dict(tickfont=dict(color=CHART_TEXT, size=10.5), showgrid=False),
        yaxis=dict(tickfont=dict(color=CHART_TEXT), gridcolor=CHART_GRID,
                   tickprefix="$", separatethousands=True)))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


# =============================================================================
# SECTION 7 :: TABLES
# =============================================================================

def render_action_queue(products: pd.DataFrame) -> None:
    """The operational deliverable: what to do today, most expensive first.

    Sorted by `net_exposure` rather than alphabetically, so the worst problem
    is always row one. Interactive by construction (Streamlit's virtualised
    grid gives sorting, search, resize and CSV download for free).
    """
    queue = products[products["alert_severity"] != "OK"].copy()
    if queue.empty:
        st.success("No SKU in this selection requires action. "
                   "Every buffer is above its safety level.")
        return
    queue = queue.sort_values("net_exposure", ascending=False)

    # Plain text, not HTML. st.dataframe renders through a sanitising grid, so
    # an injected <span class="chip"> is stripped to bare text. Status is
    # instead encoded as a leading glyph plus a text label, which survives
    # sanitisation and stays legible when the table is exported to CSV.
    glyph = {"Stockout Risk": "▲", "Dead Stock": "■"}
    display = pd.DataFrame({
        "SKU": queue["product_id"],
        "Product": queue["product_name"],
        "Category": queue["category"],
        "Zone": queue["warehouse_zone"],
        "ABC": queue["abc_class"],
        "Condition": [u"{} {}".format(glyph.get(a, "●"), a)
                      for a in queue["alert_flags"]],
        "On hand": queue["current_stock"],
        "Safety": queue["safety_stock_level"],
        "Days supply": queue["days_of_supply"],
        "Reorder qty": queue["reorder_quantity"],
        "Reorder value": queue["reorder_value"],
        "Demand missed": queue["unsellable_units"],
        "Lost margin": queue["lost_margin_estimate"],
        "Trapped capital": queue["trapped_capital"],
        "Net exposure": queue["net_exposure"],
        "Buffer risk": queue["safety_stock_risk"],
        "Action": queue["recommended_action"],
    })

    st.dataframe(
        display, width="stretch", hide_index=True,
        height=min(430, 62 + 35 * len(display)),
        column_config={
            "ABC": st.column_config.TextColumn("ABC", width="small"),
            "Condition": st.column_config.TextColumn("Condition", width="small"),
            "Buffer risk": st.column_config.TextColumn("Buffer risk",
                                                       width="small"),
            "On hand": st.column_config.NumberColumn("On hand", format="%d"),
            "Safety": st.column_config.NumberColumn("Safety", format="%d"),
            "Days supply": st.column_config.NumberColumn("Days supply",
                                                         format="%.1f"),
            "Reorder qty": st.column_config.NumberColumn("Reorder qty",
                                                         format="%d"),
            "Reorder value": st.column_config.NumberColumn("Reorder value",
                                                           format="$%.0f"),
            "Demand missed": st.column_config.NumberColumn(
                "Demand missed", format="%.1f",
                help="Lower bound on unmet demand: units short of one "
                     "safety-level of cover at the observed daily rate."),
            "Lost margin": st.column_config.NumberColumn("Lost margin",
                                                         format="$%.0f"),
            "Trapped capital": st.column_config.NumberColumn("Trapped capital",
                                                             format="$%.0f"),
            "Net exposure": st.column_config.NumberColumn(
                "Net exposure", format="$%.0f",
                help="Lost margin + trapped capital. The single number that "
                     "ranks the action queue."),
            "Action": st.column_config.TextColumn("Recommended action",
                                                  width="medium"),
        })
    st.markdown(
        '<p class="dataframe-note">Ordered by net exposure. Click any column '
        'header to re-sort, or use the download icon for CSV. Figures in the '
        'Condition and Buffer risk columns are status chips.</p>',
        unsafe_allow_html=True)


def render_risk_register(products: pd.DataFrame) -> None:
    """The parameter-review table: which SKUs have the wrong safety level.

    Deliberately separates SKUs we can measure from those we cannot, because
    the whole point is that a censored demand signal must not be mistaken for
    a low-risk one.
    """
    register = products[products["daily_demand_mean"] > 0].copy()
    if register.empty:
        st.info("No SKU in this selection recorded any demand in the window.")
        return
    register = register.sort_values("safety_stock_gap", ascending=False, na_position="last")

    signal_glyph = {"Observed": "●", "Censored": "▲", "No Demand": "—"}
    display = pd.DataFrame({
        "SKU": register["product_id"],
        "Product": register["product_name"],
        "Category": register["category"],
        "Signal": [u"{} {}".format(signal_glyph.get(q, "●"), q)
                   for q in register["demand_signal_quality"]],
        "Mean/day": register["daily_demand_mean"],
        "Std dev": register["daily_demand_sigma"],
        "Peak/day": register["daily_demand_peak"],
        "CV": register["demand_cv"],
        "Configured": register["safety_stock_level"],
        "Recommended": register["recommended_safety_stock"],
        "Gap": register["safety_stock_gap"],
        "Next 7d (low)": register["forecast_next_7d_low"],
        "Next 7d (mid)": register["forecast_next_7d"],
        "Next 7d (high)": register["forecast_next_7d_high"],
        "Verdict": register["safety_stock_risk"],
    })

    st.dataframe(
        display, width="stretch", hide_index=True,
        height=min(430, 62 + 35 * len(display)),
        column_config={
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "Verdict": st.column_config.TextColumn("Verdict", width="small"),
            "Mean/day": st.column_config.NumberColumn("Mean/day", format="%.2f"),
            "Std dev": st.column_config.NumberColumn("Std dev", format="%.2f"),
            "Peak/day": st.column_config.NumberColumn("Peak/day", format="%d"),
            "CV": st.column_config.NumberColumn(
                "CV", format="%.2f",
                help="Coefficient of variation: std dev / mean of daily "
                     "demand. Above ~0.8 is hard to forecast."),
            "Configured": st.column_config.NumberColumn("Configured",
                                                        format="%.0f"),
            "Recommended": st.column_config.NumberColumn(
                "Recommended", format="%.0f",
                help="z * sigma * sqrt(lead time) for a 95% service level "
                     "over a 21-day lead time."),
            "Gap": st.column_config.NumberColumn("Gap", format="%.0f"),
            "Next 7d (low)": st.column_config.NumberColumn("Next 7d (low)",
                                                           format="%.1f"),
            "Next 7d (mid)": st.column_config.NumberColumn("Next 7d (mid)",
                                                           format="%.1f"),
            "Next 7d (high)": st.column_config.NumberColumn("Next 7d (high)",
                                                            format="%.1f"),
        })
    st.markdown(
        '<p class="dataframe-note">Gap = recommended − configured. Blank means '
        'the demand signal is censored (see the note above), not zero.</p>',
        unsafe_allow_html=True)


def render_full_inventory(products: pd.DataFrame) -> None:
    """The complete analytical frame, for auditors and examiners."""
    st.dataframe(
        products.drop(columns=[c for c in (
            "is_dead_stock", "is_stockout_risk", "opening_stock_est",
            "txn_count", "cost", "safety_stock_cover_days",
            "daily_demand_peak", "active_sales_days", "service_level_z",
            "assumed_lead_time_days", "lost_revenue_estimate",
            "demand_cover_shortfall", "unsellable_units")]),
        width="stretch", hide_index=True, height=400,
        column_config={
            "revenue": st.column_config.NumberColumn("revenue", format="$%.2f"),
            "cogs": st.column_config.NumberColumn("cogs", format="$%.2f"),
            "margin": st.column_config.NumberColumn("margin", format="$%.2f"),
            "margin_pct": st.column_config.NumberColumn("margin_pct",
                                                        format="%.2f%%"),
            "price": st.column_config.NumberColumn("price", format="$%.2f"),
            "inventory_value": st.column_config.NumberColumn("inventory_value",
                                                             format="$%.2f"),
            "average_inventory_value": st.column_config.NumberColumn(
                "average_inventory_value", format="$%.2f"),
            "revenue_share_pct": st.column_config.NumberColumn(
                "revenue_share_pct", format="%.3f%%"),
            "cumulative_revenue_pct": st.column_config.NumberColumn(
                "cumulative_revenue_pct", format="%.3f%%"),
            "net_exposure": st.column_config.NumberColumn("net_exposure",
                                                          format="$%.2f"),
        })


# =============================================================================
# SECTION 8 :: PROVENANCE / VIVA PANEL
# =============================================================================

def render_provenance(meta: Dict[str, object], manifest: Dict[str, object]) -> None:
    t1, t2, t3, t4 = st.tabs(["Pipeline", "Spark actions", "HDFS layout",
                               "Viva notes"])

    with t1:
        st.markdown("""
| Stage | Engine | What happens |
|---|---|---|
| 1. Ingest | Python | Three CSV partitions written to a simulated HDFS namespace, each committed with a Hadoop `_SUCCESS` marker |
| 2. Read | **PySpark** | `spark.read.schema(...).csv(...)` — schema-on-read, `FAILFAST`, explicit `timestampFormat` |
| 3. Aggregate | **PySpark** | `groupBy(product_id)` → units, revenue, COGS. Cached: three downstream consumers |
| 4. Spine | **PySpark** | Product master as the LEFT side, so dead stock is never lost |
| 5. ABC | **PySpark** | `Window.orderBy(revenue desc).rowsBetween(unboundedPreceding, -1)` → 80/15/5 cut |
| 6. Turnover | **PySpark** | COGS ÷ average inventory value, plus days of supply |
| 7. Alerts | **PySpark** | Stockout (`<=`) and dead stock (`LEFT ANTI JOIN`) |
| 8. Risk | **PySpark** | Daily-demand re-aggregation → CV, service-level buffer, 7-day forecast band |
| 9. Impact | **PySpark** | Lost margin and trapped capital per SKU |
| 10. Action | **PySpark → Pandas** | `toPandas()` — the only place data leaves the JVM |
| 11. Present | Streamlit | Filter 60 rows, KPI tiles, charts, action queue |

**Two design decisions worth defending**
- The **product master is the LEFT spine** of every join. An inner join would
  silently drop the five dead-stock SKUs — the exact rows this console exists
  to surface.
- **`.cache()` on the sales aggregation** because that node feeds three
  consumers; without it Spark re-scans and re-shuffles the ledger three times.
""")

    with t2:
        st.markdown("**Every action the pipeline executed.** Only these calls "
                    "trigger real Spark jobs; everything above them is lazy.")
        st.dataframe(pd.DataFrame(meta.get("actions", [])), width="stretch",
                     hide_index=True)
        total = sum(a["seconds"] for a in meta.get("actions", []) or [])
        st.caption("{} actions totalling {:.2f}s of distributed work.".format(
            len(meta.get("actions", []) or []), total))

    with t3:
        if manifest:
            st.json(manifest, expanded=False)
        else:
            st.info("No manifest found (the dataset was built in an earlier run).")
        st.caption("Local mock root: `{}`".format(HDFS_LOCAL_ROOT))

    with t4:
        st.markdown("""
**Q. Why Spark for 2,500 rows?**
The architecture is identical at 2.5 billion rows — the same DAG, shuffles and
Catalyst optimisations. Scaling the data does not change the code; that is the
point of the DataFrame API.

**Q. Lazy evaluation?**
`filter / join / groupBy` only append to the lineage DAG. Nothing executes
until an action (`toPandas`, `count`, `show`, `first`, `write`). The *Spark
actions* tab lists all of them with their measured durations.

**Q. Broadcast versus shuffle?**
`F.broadcast()` on the 60-row dimensions avoids shuffling them. Every join in
the physical plan is a `BroadcastHashJoin`; the only unavoidable shuffle is the
`groupBy` and the daily-demand re-aggregation.

**Q. How is dead stock found?**
`LEFT ANTI JOIN` — one hash anti-join. The `COUNT(*) = 0` alternative forces
two extra aggregations and cannot be combined with other columns.

**Q. Where is the 80% cut?**
A windowed running total of revenue, classified on the share *before* each SKU
so the SKU that crosses 80% stays in class A. That requires the exclusive frame
`rowsBetween(Window.unboundedPreceding, -1)`.

**Q. What does `sqrt(lead time)` do in the safety-stock formula?**
Daily demand noise is independent, so variance accumulates linearly with the
lead time while standard deviation grows only as the square root. Under-ordering
this term is the textbook cause of chronic stockouts.

**Q. Why is the gap blank for the stockout SKUs?**
Because their demand is *censored*: an out-of-stock SKU records no transactions,
so its observed mean is biased low and any recommendation derived from it is
unreliable. We label those SKUs `Censored` and return `NULL` rather than
publishing a precise-looking number we know is wrong. Correcting the bias
properly needs substitution or EM estimation against out-of-stock-period data,
which we do not have.
""")


# =============================================================================
# SECTION 9 :: MAIN
# =============================================================================

def main() -> None:
    manifest = read_manifest()

    # ---- 1. LOAD (auto-bootstrap HDFS, then run Spark) --------------------
    try:
        results = load_pipeline()
    except Exception as exc:                       # pragma: no cover
        st.error("The PySpark pipeline failed to start: {}".format(exc))
        st.exception(exc)
        st.stop()

    all_products: pd.DataFrame = results["products"]
    meta: Dict[str, object] = results["meta"]
    kpis: Dict[str, object] = results["kpis"]
    ingestion = meta.get("ingestion", {}) or {}

    # ---- 2. APP BAR -------------------------------------------------------
    st.markdown("""
    <div class="appbar">
      <div class="brand"><span class="mark">▦</span> SmartStock Inventory Control Tower</div>
      <div class="meta">
        <span>Engine <b>PySpark {}</b></span>
        <span>SKUs <b>{}</b></span>
        <span>Events <b>{}</b></span>
        <span>Window <b>{}d</b></span>
        <span>Run <b>{}s</b></span>
      </div>
    </div>
    """.format(
        meta.get("spark_version"), len(all_products),
        num(ingestion.get("sales_ledger_rows")),
        meta.get("sales_window_days"), meta.get("total_seconds")),
        unsafe_allow_html=True)

    # ---- 3. FILTERS + SLICE ----------------------------------------------
    filters = render_sidebar(all_products, meta)
    products = apply_filters(all_products, filters)
    empty = products.empty
    filters_active = any([
        filters["categories"], filters["classes"], filters["zones"],
        filters["severities"], filters["min_revenue"], filters["only_alerts"]])

    if empty:
        st.warning("No SKU matches the current filters. Widen the selection, "
                    "or press **Reset all filters** in the sidebar.")
        categories = pd.DataFrame(columns=["category", "revenue", "sku_count"])
        abc = pd.DataFrame(columns=["abc_class", "abc_class_label", "sku_count"])
        totals = {k: 0.0 for k in (
            "revenue", "cogs", "margin", "inventory_value", "avg_inventory_value",
            "net_exposure", "lost_margin", "lost_revenue", "trapped",
            "reorder_value", "avg_cv")}
        totals.update({"stockout": 0, "dead": 0, "under_buffered": 0,
                       "over_buffered": 0, "censored": 0})
    else:
        categories, abc, totals = rollup(products)

    # ---- 4. HEADLINE EXPOSURE --------------------------------------------
    scope = "filtered selection" if filters_active else "whole catalogue"
    exposure_pct = (totals["net_exposure"] / totals["revenue"] * 100
                    if totals["revenue"] else 0.0)
    if totals["censored"]:
        buffer_tone = "critical"
    elif totals["under_buffered"]:
        buffer_tone = "warning"
    else:
        buffer_tone = "ok"

    st.markdown(
        '<div class="sec"><span class="num">01</span>'
        '<h3>Capital at risk</h3></div>'
        '<div class="hint">Cost of the open alerts across the {}. '
        'These are estimates from observed demand, not forecasts.</div>'
        .format(scope),
        unsafe_allow_html=True,
    )
    c = st.columns(5, gap="small")
    with c[0]:
        tile("Net exposure",
             compact_money(totals["net_exposure"]),
             "{}% of revenue".format(num(exposure_pct, 1)),
             "Lost margin {} + trapped {}".format(
                 compact_money(totals["lost_margin"]),
                 compact_money(totals["trapped"])),
             tone="critical" if totals["net_exposure"] > 0 else "ok")
    with c[1]:
        tile("Revenue in scope", compact_money(totals["revenue"]),
             "{:.1f}% margin".format(
                 (totals["margin"] / totals["revenue"] * 100)
                 if totals["revenue"] else 0.0),
             "COGS {} · {} SKUs".format(
                 compact_money(totals["cogs"]), len(products)),
             tone="info")
    with c[2]:
        tile("Stockout risk", num(totals["stockout"]),
             "on hand ≤ safety level" if totals["stockout"] else "all buffers held",
             "Replenishment {}".format(compact_money(totals["reorder_value"])),
             tone="critical" if totals["stockout"] else "ok")
    with c[3]:
        tile("Dead stock", num(totals["dead"]),
             "no demand in window" if totals["dead"] else "none found",
             "Capital frozen on shelf",
             tone="warning" if totals["dead"] else "ok")
    with c[4]:
        tile("Buffer misalignment", num(totals["under_buffered"]),
             "{} censored · {} over".format(totals["censored"],
                                            totals["over_buffered"]),
             "vs 95% service level, 21d lead",
             tone=buffer_tone)

    if not empty:
        # ---- 5. ACTION QUEUE ----------------------------------------------
        st.markdown(
            '<div class="sec"><span class="num">02</span>'
            '<h3>Action queue</h3></div>'
            '<div class="hint">Ranked by net exposure, so the most expensive '
            'problem is the first row. Every figure is traceable to a Spark '
            'action listed under Pipeline below.</div>',
            unsafe_allow_html=True)
        render_action_queue(products)

        # ---- 6. WHERE THE MONEY IS ----------------------------------------
        st.markdown(
            '<div class="sec"><span class="num">03</span>'
            '<h3>Composition of the exposure</h3></div>'
            '<div class="hint">Lost margin and trapped capital need opposite '
            'decisions — buy more versus liquidate — so they are shown as '
            'separate series.</div>',
            unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            chart_exposure_by_sku(products)
        with right:
            chart_revenue_by_category(categories)

        st.markdown(
            '<div class="sec"><span class="num">04</span>'
            '<h3>Portfolio structure</h3></div>'
            '<div class="hint">Where capital is held versus generated, and how '
            'the catalogue splits by value class.</div>',
            unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            chart_turnover_by_category(categories)
        with right:
            chart_abc_distribution(abc)

        # ---- 7. DEMAND RISK ------------------------------------------------
        st.markdown(
            '<div class="sec"><span class="num">05</span>'
            '<h3>Demand risk &amp; safety-stock parameters</h3></div>'
            '<div class="hint">The safety levels currently configured in the '
            'warehouse are static, but demand is not. These charts compare the '
            'configured buffer against a service-level requirement derived '
            'from each SKU&#39;s own demand volatility.</div>',
            unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            chart_buffer_gap(products)
        with right:
            chart_volatility_vs_revenue(products)

        st.markdown(
            '<div class="sec"><span class="num">06</span>'
            '<h3>Safety-stock parameter review</h3></div>'
            '<div class="hint">One row per SKU with usable demand data, sorted '
            'by the size of the gap. Blank gaps are demand-censored, not zero.</div>',
            unsafe_allow_html=True)
        render_risk_register(products)

        with st.expander("Full analytical frame — {} SKUs × {} columns"
                         .format(len(products), len(products.columns))):
            render_full_inventory(products)

    # ---- 8. PROVENANCE ----------------------------------------------------
    st.markdown("---")
    render_provenance(meta, manifest)

    st.markdown(
        '<div class="foot-note">SmartStock Inventory Control Tower &middot; '
        'Streamlit {} &middot; Plotly &middot; PySpark {} &middot; simulated '
        'HDFS at <code>{}</code></div>'.format(
            st.__version__, meta.get("spark_version"), HDFS_LOCAL_ROOT),
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
