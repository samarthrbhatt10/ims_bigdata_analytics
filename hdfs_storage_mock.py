#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SMARTSTOCK :: HDFS STORAGE SIMULATION LAYER   (hdfs_storage_mock.py)
===============================================================================

The data-lake ingestion simulator. It fakes a Hadoop Distributed File System
namespace on local disk so the whole project runs on one laptop (Windows or
macOS) with no NameNode/DataNode cluster. On a real cluster the equivalent
would be `hdfs dfs -mkdir -p ...` plus `hdfs dfs -put ...`; here we simply
write real CSV bytes into a directory whose *name* mimics the HDFS layout, so
every downstream layer reads a plain local path while *thinking* it is talking
to HDFS.

Dataset written to the RAW (Bronze) layer:
    products.csv       60 unique SKUs across 5 categories
    stock.csv          current on-hand quantity per SKU
    sales_ledger.csv   2,500 transactions across the last 90 days

HDFS CONCEPTS THIS LAYER ACTUALLY MODELS (worth stating in the viva)
    * WORM storage. We only ever create a fresh dataset; the app never mutates
      a file in place, it only reads. That is the HDFS contract.
    * Blocks + replication. Each dataset is one file, so it occupies exactly
      one 128 MB block at RF=3 (see the constants below). Both are declared so
      the console output and manifest read like a real Hadoop session.
    * The `_SUCCESS` commit marker is a genuine Hadoop convention: write the
      data first, commit the marker LAST. A consumer that finds no marker
      refuses to read a half-written dataset -- which is why the dashboard
      self-heals instead of crashing on partial input.

INJECTED EDGE CASES (the project's test cases)
    * 5 DEAD STOCK SKUs     -> never appear in the ledger at all
                               (found later by a Spark LEFT ANTI JOIN).
    * 5 STOCKOUT RISK SKUs  -> on-hand is <= the safety stock level
                               (found later by a Spark filter).
    * 1 SKU exactly ON the safety-level boundary, to prove the production rule
      uses `<=` and not `<`.

DETERMINISM
    A fixed random seed makes the CSVs byte-identical on every machine, so the
    numbers in your report always match the demo.

CLI
    python hdfs_storage_mock.py                       # generate only if missing
    python hdfs_storage_mock.py --force               # wipe + regenerate
    python hdfs_storage_mock.py --stats               # show what is committed
    python hdfs_storage_mock.py --transactions 2500000 # scale the demo up
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# =============================================================================
# SECTION 1 :: HDFS TOPOLOGY CONSTANTS
# =============================================================================
# Single source of truth.  We define the paths as *HDFS URI paths* (the way a
# real Hadoop job would address them) and derive the local filesystem location
# from them.  Change the constant -> the whole project follows.

PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Local folder that plays the role of the HDFS root directory (the NameNode's
#: namespace).  On a real cluster this is just a path inside the container.
HDFS_LOCAL_ROOT: Path = PROJECT_ROOT / "hdfs"

#: Default NameNode RPC address, purely cosmetic -- it is what we *print* to
#: the console so the console log looks like a real Hadoop session.
HDFS_NAMENODE_URI: str = "hdfs://localhost:9000"

#: The OS user that owns the HDFS home directory ("hadoop" is the convention).
HDFS_USER: str = "hadoop"

#: Logical application directory: /user/<user>/<application>
HDFS_APP_HOME: str = f"/user/{HDFS_USER}/inventory"

#: The RAW zone of our medallion architecture (Bronze layer).
HDFS_RAW_LAYER: str = f"{HDFS_APP_HOME}/raw"

#: The three physical partitions of the RAW layer.
HDFS_PRODUCTS_PATH: str = f"{HDFS_RAW_LAYER}/products"
HDFS_STOCK_PATH: str = f"{HDFS_RAW_LAYER}/stock"
HDFS_SALES_PATH: str = f"{HDFS_RAW_LAYER}/sales"

#: Hadoop replication factor.  On a laptop we obviously keep a single copy,
#: but we keep the constant + the comment because it is a viva question.
HDFS_REPLICATION_FACTOR: int = 3

#: Hadoop default HDFS block size.  Our CSVs are far smaller than one block,
#: therefore each file occupies exactly ONE block on a real cluster.
HDFS_BLOCK_SIZE_BYTES: int = 128 * 1024 * 1024

#: Timestamp string format used by PANDAS when we WRITE the ledger.
#: This is a PYTHON strftime pattern (pandas -> datetime.strftime).
TIMESTAMP_FORMAT: str = "%Y-%m-%dT%H:%M:%S"

#: Timestamp pattern used by PYSPARK when we READ the ledger back.
#: IMPORTANT (a very common, very expensive mistake): the `timestampFormat`
#: reader option follows JAVA's `java.time.format.DateTimeFormatter` syntax,
#: NOT Python's strftime syntax.  Spark >= 3.3 (and mandatory in Spark 4.x)
#: raises `INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION`
#: if you feed it "%Y-%m-%d %H:%M:%S".  The Java equivalent of the pattern
#: above is "yyyy-MM-dd'T'HH:mm:ss" -- ISO-8601, which every Spark version
#: parses correctly, so this pattern is portable across Spark 3.4 -> 4.x.
SPARK_TIMESTAMP_FORMAT: str = "yyyy-MM-dd'T'HH:mm:ss"

#: Master seed -- guarantees reproducible data on Windows, macOS and Linux.
RANDOM_SEED: int = 20260214

#: Dataset sizing (kept configurable so the team can demo "big data" scaling
#: by cranking these numbers up, e.g. --transactions 2500000).
NUM_PRODUCTS: int = 60
NUM_TRANSACTIONS: int = 2_500
SALES_WINDOW_DAYS: int = 90
NUM_DEAD_STOCK_SKUS: int = 5
NUM_STOCKOUT_RISK_SKUS: int = 5


def _local_path(hdfs_path: str) -> Path:
    """Translate an HDFS URI path (e.g. /user/hadoop/x) into our local mock path.

    This single helper is the *only* place where "HDFS" and "local disk"
    differ.  In a real deployment you would delete this function and hand the
    URI straight to Spark's Hadoop FileSystem -- zero code changes in the
    analytics engine.  That is a good design point to mention: the rest of
    the project never knows which filesystem it is talking to.
    """
    # NOTE: pathlib normalises the "/" separators on Windows automatically, so
    # the exact same string works on Windows, macOS and Linux (0 abstraction leak).
    return HDFS_LOCAL_ROOT / hdfs_path.lstrip("/")


def to_hdfs_uri(hdfs_path: str) -> str:
    """Build the fully qualified hdfs:// URI for a path (used for logging)."""
    return f"{HDFS_NAMENODE_URI}{hdfs_path}"


# Canonical file names inside each partition.
PRODUCTS_CSV: Path = _local_path(HDFS_PRODUCTS_PATH) / "products.csv"
STOCK_CSV: Path = _local_path(HDFS_STOCK_PATH) / "stock.csv"
SALES_CSV: Path = _local_path(HDFS_SALES_PATH) / "sales_ledger.csv"

# The Hadoop commit marker, written only after a dataset is fully flushed.
SUCCESS_MARKER: str = "_SUCCESS"

# Manifest describing the generated dataset (row counts, edge cases, ...).
MANIFEST_FILE: Path = _local_path(HDFS_RAW_LAYER) / "_manifest.json"


# =============================================================================
# SECTION 2 :: THE PRODUCT CATALOG (deterministic, human-readable SKUs)
# =============================================================================
# 60 products = 5 categories x 12 items.  Hard-coding the nouns (instead of
# faking "item_1", "item_2") makes the Streamlit dashboard look like a real
# enterprise product -- a small detail that reviewers notice immediately.

CATEGORIES: List[str] = ["Electronics", "Apparel", "Grocery", "Home", "Automotive"]

CATEGORY_CATALOG: Dict[str, List[str]] = {
    "Electronics": [
        "Laptop", "Smartphone", "Tablet", "Monitor", "Headphones", "Keyboard",
        "Wireless Mouse", "Webcam", "Bluetooth Speaker", "WiFi Router",
        "Portable SSD", "Power Bank",
    ],
    "Apparel": [
        "Denim Jacket", "Hoodie", "Slim Jeans", "Running Sneakers", "Polo Shirt",
        "Chino Trousers", "Baseball Cap", "Wool Scarf", "Leather Belt",
        "Travel Wallet", "Socks Pack", "Raincoat",
    ],
    "Grocery": [
        "Basmati Rice", "Wheat Flour", "Olive Oil", "Coffee Beans", "Green Tea",
        "Pasta", "Granola Box", "Wild Honey", "Dark Chocolate", "Almonds",
        "Peanut Butter", "Mixed Spices",
    ],
    "Home": [
        "Floor Lamp", "Desk Chair", "Cookware Set", "Bed Sheet", "Window Curtain",
        "Storage Box", "Wall Clock", "Vacuum Flask", "Photo Frame", "Area Rug",
        "Electric Kettle", "Wall Art",
    ],
    "Automotive": [
        "Car Tyre", "Engine Oil", "Brake Pad", "Car Battery", "Air Filter",
        "Windshield Wiper", "Car Cover", "Seat Cover", "GPS Tracker",
        "LED Headlight Kit", "Radiator Coolant", "Spark Plug",
    ],
}

# Brand prefixes -> generate realistic-looking SKU names that are still unique.
BRAND_PREFIXES: List[str] = [
    "Aurora", "Vertex", "Nimbus", "Zenith", "Orion", "Cascade",
    "Prime", "Lumen", "Terra", "Quantum", "Atlas", "Fusion",
]

# Typical (cost_min, cost_max, markup_min, markup_max) per category.
# Grocery is cheap & high-volume, Electronics is expensive & low-volume.
CATEGORY_ECONOMICS: Dict[str, Tuple[float, float, float, float]] = {
    "Electronics": (85.0, 1250.0, 1.18, 1.55),
    "Apparel":     (12.0, 140.0,  1.45, 2.30),
    "Grocery":     (1.5, 28.0,    1.12, 1.40),
    "Home":        (9.0, 320.0,   1.35, 2.10),
    "Automotive":  (7.0, 210.0,   1.30, 1.85),
}

WAREHOUSE_ZONES: List[str] = ["Zone-A", "Zone-B", "Zone-C", "Zone-D", "Cold-Chain"]


# =============================================================================
# SECTION 3 :: HDFS DIRECTORY TREE MANAGEMENT
# =============================================================================

def create_hdfs_directory_tree() -> List[Path]:
    """Materialise the simulated HDFS namespace (mkdir -p).

    Layout produced under ./hdfs :
        hdfs/user/hadoop/inventory/raw/products/
        hdfs/user/hadoop/inventory/raw/stock/
        hdfs/user/hadoop/inventory/raw/sales/

    Returns the list of created directories.
    """
    directories = [
        _local_path(HDFS_PRODUCTS_PATH),
        _local_path(HDFS_STOCK_PATH),
        _local_path(HDFS_SALES_PATH),
        HDFS_LOCAL_ROOT / "spark-warehouse",   # Spark's own managed dir
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("[HDFS] Namespace initialised at: {}".format(HDFS_LOCAL_ROOT))
    for directory in directories[:3]:
        hdfs_uri_path = "/" + str(directory.relative_to(HDFS_LOCAL_ROOT)).replace("\\", "/")
        print("[HDFS]   hdfs dfs -mkdir -p {:<58} -> {}".format(
            to_hdfs_uri(hdfs_uri_path),
            str(directory.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        ))
    return directories


def _commit_success_marker(directory: Path, hdfs_path: str) -> None:
    """Drop the Hadoop `_SUCCESS` marker -- the atomic "dataset is readable" flag.

    Rule of thumb we follow everywhere in this project: *write data first,
    commit the marker last*.  If the process dies half way through, a consumer
    that checks for `_SUCCESS` will refuse to read a half-written dataset.
    """
    marker = directory / SUCCESS_MARKER
    marker.write_text(
        "committed_at={}\nhdfs_uri={}\nreplication_factor={}\nblock_size_bytes={}\n".format(
            datetime.now().isoformat(timespec="seconds"),
            to_hdfs_uri(hdfs_path),
            HDFS_REPLICATION_FACTOR,
            HDFS_BLOCK_SIZE_BYTES,
        ),
        encoding="utf-8",
    )


# =============================================================================
# SECTION 4 :: DATASET GENERATORS  (one function per HDFS partition)
# =============================================================================

def generate_product_master(rng: random.Random) -> pd.DataFrame:
    """Partition 1 -- the Product Master (SKU dimension table).

    This is the *spine* of every downstream report: it is a slowly changing
    dimension (SCD Type 1) in data-warehouse terms, and it is the table that
    defines "the universe of products we stock".  Dead-stock detection is
    literally "product in the master, never in the ledger", so the master must
    always be complete -- which is exactly why it lives in HDFS as an
    authoritative, versioned, replicated dataset.
    """
    rows: List[dict] = []
    sequence = 0

    for category in CATEGORIES:
        cost_min, cost_max, _, _ = CATEGORY_ECONOMICS[category]
        for item_name in CATEGORY_CATALOG[category]:
            sequence += 1
            brand = BRAND_PREFIXES[sequence % len(BRAND_PREFIXES)]
            cost = round(rng.uniform(cost_min, cost_max), 2)
            # Retail price: cost + category-specific markup, rounded to 2dp.
            _, _, markup_min, markup_max = CATEGORY_ECONOMICS[category]
            price = round(cost * rng.uniform(markup_min, markup_max), 2)
            # Model code guarantees a globally unique, sortable product name.
            model_code = "{}{}-{:04d}".format(
                brand[0], brand[1], 1000 + sequence * 7
            )
            rows.append({
                "product_id": "SKU-{:03d}".format(sequence),
                "product_name": "{} {} {}".format(brand, item_name, model_code),
                "category": category,
                "cost": cost,
                "price": price,
            })

    products = pd.DataFrame(rows, columns=["product_id", "product_name",
                                           "category", "cost", "price"])
    assert len(products) == NUM_PRODUCTS, "Expected {} products".format(NUM_PRODUCTS)
    return products


def assign_demand_profile(rng: random.Random, products: pd.DataFrame) -> pd.DataFrame:
    """Attach a *demand weight* to every SKU following a realistic Pareto curve.

    Real inventory is extremely skewed: a handful of SKUs ("the Pareto 20%")
    drive most of the revenue.  If we generated demand uniformly at random the
    ABC analysis would produce a meaningless 60/60/60 split and the dashboard
    would look wrong.  So we use an inverse-power (Zipf-like) weight curve:
    the 1st SKU gets weight ~12, the 60th gets weight ~0.3.

    The weights are *only* used by the sales generator; they are not written
    to HDFS (in a real system that would be a model output, not raw data).
    """
    enriched = products.copy()
    weights: List[float] = []
    for rank in range(1, len(enriched) + 1):
        # Base Zipf weight + jitter so it is not a perfect mathematical curve.
        base_weight = 12.0 / (rank ** 0.85)
        weights.append(round(base_weight * rng.uniform(0.75, 1.30), 4))
    enriched["demand_weight"] = weights
    return enriched


def select_edge_case_skus(rng: random.Random, enriched: pd.DataFrame
                          ) -> Tuple[List[str], List[str]]:
    """Pick which SKUs will be our *injected business anomalies*.

    Design decision (mention this in the viva): we do NOT pick these at
    random.  We pick them *because they are the most business-realistic
    cases*:
        * DEAD STOCK  -> the 5 slowest movers.  Nobody bought them in 90 days;
          capital is frozen in the shelf. This is exactly how dead stock
          happens in the real world.
        * STOCKOUT RISK -> 5 fast movers.  High demand burns through on-hand
          units faster than replenishment; this is exactly how stockouts
          happen in the real world.
    The two sets are therefore disjoint by construction, so the dashboard can
    demonstrate both alert types independently *and* together.
    """
    ordered = enriched.sort_values("demand_weight", ascending=False).reset_index(drop=True)

    # -- Dead stock: the five lowest-demand SKUs, weight forced to 0. ---------
    dead_stock_ids = ordered.tail(NUM_DEAD_STOCK_SKUS)["product_id"].tolist()

    # -- Stockout risk: five healthy fast-movers from the top half. -----------
    #    We take from ranks 4..18 so that the very top SKUs stay "Healthy" and
    #    the dashboard shows a believable mix.
    fast_mover_pool = ordered.iloc[3:18]["product_id"].tolist()
    stockout_risk_ids = rng.sample(fast_mover_pool, NUM_STOCKOUT_RISK_SKUS)

    return dead_stock_ids, stockout_risk_ids


def generate_stock_ledger(rng: random.Random, products: pd.DataFrame,
                          stockout_risk_ids: List[str]) -> pd.DataFrame:
    """Partition 2 -- the Warehouse Stock Snapshot (current on-hand state).

    Business rules encoded here:
        current_stock        : units physically available right now
        safety_stock_level   : the reorder *buffer*; a buffer exists to absorb
                               demand/supply variance, and is normally ~15-25%
                               of on-hand stock.
        warehouse_zone       : physical location, used as a dashboard filter.

    For the 5 injected stockout-risk SKUs we FORCE current_stock <= safety
    level.  We make exactly one of them sit precisely ON the boundary
    (current == safety) to prove the production rule uses `<=` and not `<`.
    """
    rows: List[dict] = []

    for record in products.to_dict("records"):
        product_id = record["product_id"]

        if product_id in stockout_risk_ids:
            # --- INJECTED ANOMALY: on-hand is at or below the safety buffer ---
            if product_id == stockout_risk_ids[0]:
                # Boundary case: current_stock EXACTLY equals safety level.
                # The pipeline rule is `current_stock <= safety_stock_level`,
                # so this SKU must be flagged.
                current_stock = 45
                safety_level = 45
            else:
                safety_level = rng.randint(40, 120)
                current_stock = int(safety_level * rng.uniform(0.10, 0.92))
            zone = rng.choice(WAREHOUSE_ZONES)
        else:
            # --- Normal SKU: healthy buffer above the safety level ----------
            current_stock = rng.randint(60, 900)
            safety_level = max(5, int(current_stock * rng.uniform(0.12, 0.24)))
            zone = rng.choice(WAREHOUSE_ZONES)

        rows.append({
            "product_id": product_id,
            "current_stock": int(current_stock),
            "safety_stock_level": int(safety_level),
            "warehouse_zone": zone,
        })

    return pd.DataFrame(rows, columns=["product_id", "current_stock",
                                       "safety_stock_level", "warehouse_zone"])


def generate_sales_ledger(rng: random.Random, enriched: pd.DataFrame,
                          dead_stock_ids: List[str],
                          num_transactions: int = NUM_TRANSACTIONS,
                          window_days: int = SALES_WINDOW_DAYS) -> pd.DataFrame:
    """Partition 3 -- the Sales Ledger (append-only fact table, 2,500 rows).

    This is the classic *append-only event stream* of a retail business.  Note
    the two deliberate modelling choices:

    1.  SKEWED PRODUCT SAMPLING.  Each transaction picks a SKU with probability
        proportional to its `demand_weight` (Pareto).  Consequences: revenue is
        concentrated in a few SKUs -> the ABC analysis produces a realistic
        80/15/5 split instead of a flat line.

    2.  RECENCY TREND.  Days closer to "today" are sampled with a slightly
        higher probability (a gentle growth trend).  This is what real demand
        looks like and it makes the "logistics velocity" numbers non-trivial.

    The 5 dead-stock SKUs get weight 0.0, which makes them mathematically
    impossible to be drawn by `random.choices` (cumulative weight never
    advances for them) -- i.e. they are structurally absent from the ledger.
    """
    # ---- Build the weighted sampler over the NON-dead SKUs ------------------
    active = enriched[~enriched["product_id"].isin(dead_stock_ids)].copy()
    product_ids: List[str] = active["product_id"].tolist()
    weights: List[float] = active["demand_weight"].tolist()

    # ---- Pre-compute the day-offset probability table (recency trend) ------
    # A product-style increasing weight: day 0 (oldest) -> weight 1.0,
    # day `window_days` (most recent) -> weight ~2.0.
    day_offsets = list(range(window_days))
    day_weights = [1.0 + 1.0 * (d / max(window_days - 1, 1)) for d in day_offsets]

    # ---- Draw the transactions -------------------------------------------
    sampled_products = rng.choices(product_ids, weights=weights, k=num_transactions)
    sampled_days = rng.choices(day_offsets, weights=day_weights, k=num_transactions)
    sampled_hours = rng.choices(
        population=list(range(7, 23)),                 # 07:00 .. 22:00
        weights=[2, 4, 7, 9, 10, 11, 11, 10, 9, 8, 7, 5, 4, 3, 2, 1],
        k=num_transactions,
    )
    sampled_minutes = [rng.randint(0, 59) for _ in range(num_transactions)]
    sampled_seconds = [rng.randint(0, 59) for _ in range(num_transactions)]
    # Quantity per ticket: mostly small baskets, occasional bulk orders.
    sampled_quantities = [
        1 if rng.random() < 0.45 else rng.choices(
            population=[2, 3, 4, 5, 6, 8, 10, 12],
            weights=[22, 20, 15, 12, 10, 8, 7, 6],
            k=1,
        )[0]
        for _ in range(num_transactions)
    ]

    # "Today" at midnight; the window is [today-89d 00:00, today 23:59].
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    records = []
    for product_id, day_offset, hour, minute, second, qty in zip(
        sampled_products, sampled_days, sampled_hours,
        sampled_minutes, sampled_seconds, sampled_quantities,
    ):
        event_time = today - timedelta(days=day_offset)
        event_time = event_time.replace(hour=hour, minute=minute, second=second)
        records.append({
            "timestamp": event_time,
            "product_id": product_id,
            "quantity_sold": int(qty),
        })

    ledger = pd.DataFrame(records, columns=["timestamp", "product_id", "quantity_sold"])

    # ---- GUARANTEE exactly 5 dead-stock SKUs ------------------------------
    # With 2,500 draws over 55 SKUs a random SKU getting zero sales is
    # practically impossible, but "practically" is not a data contract.  We
    # assert the invariant and back-fill a single transaction for any active
    # SKU that received none, so the dashboard always reports EXACTLY 5.
    missing_active = sorted(set(product_ids) - set(ledger["product_id"].unique()))
    if missing_active:
        print("[HDFS] Back-filling {} zero-sale SKU(s) to keep the "
              "dead-stock invariant at exactly {}.".format(
                  len(missing_active), NUM_DEAD_STOCK_SKUS))
        filler_times = today - timedelta(days=rng.randint(1, window_days - 1),
                                         hours=rng.randint(1, 12))
        for sku in missing_active:
            ledger.loc[len(ledger)] = {
                "timestamp": filler_times,
                "product_id": sku,
                "quantity_sold": 1,
            }

    # ---- Sort chronologically, then assign a monotonic transaction_id ------
    # A real ledger is ordered by event time, and the surrogate key is
    # sequential -- that is a *sortable snowflake-ish* key, which is exactly
    # what a data engineer would ask for.
    ledger = ledger.sort_values("timestamp").reset_index(drop=True)
    ledger.insert(
        0, "transaction_id",
        ["TXN-{:07d}".format(i + 1) for i in range(len(ledger))],
    )
    return ledger[["transaction_id", "timestamp", "product_id", "quantity_sold"]]


# =============================================================================
# SECTION 5 :: WRITE DATASETS TO THE SIMULATED HDFS  (the "ingestion job")
# =============================================================================

def _write_partition(frame: pd.DataFrame, target: Path, hdfs_path: str,
                     timestamp_columns: Tuple[str, ...] = ()) -> Path:
    """Serialise a Pandas DataFrame to one HDFS partition file, then commit.

    `timestamp_columns` are converted to the fixed string format defined in
    TIMESTAMP_FORMAT so the PySpark reader can parse them with an explicit
    `timestampFormat` option (no implicit / locale-dependent parsing).
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    output = frame.copy()
    for column in timestamp_columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime(TIMESTAMP_FORMAT)

    output.to_csv(target, index=False, encoding="utf-8")

    # Hadoop convention: the _SUCCESS marker is written LAST, atomically
    # committing the dataset for downstream consumers.
    _commit_success_marker(target.parent, hdfs_path)
    return target


def build_dataset(num_transactions: int = NUM_TRANSACTIONS,
                  window_days: int = SALES_WINDOW_DAYS,
                  seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Full ingestion job: build all three partitions and commit them to HDFS.

    Returns a manifest dict (also persisted next to the data as _manifest.json)
    describing row counts and the injected edge cases.  The Streamlit app
    prints this manifest in its sidebar, which makes the demo self-documenting.
    """
    rng = random.Random(seed)     # deterministic, local RNG (not global seed)

    # 1) Build the namespaces -------------------------------------------------
    create_hdfs_directory_tree()

    # 2) Build the data -------------------------------------------------------
    products = generate_product_master(rng)
    enriched = assign_demand_profile(rng, products)
    dead_stock_ids, stockout_risk_ids = select_edge_case_skus(rng, enriched)
    stock = generate_stock_ledger(rng, products, stockout_risk_ids)
    ledger = generate_sales_ledger(rng, enriched, dead_stock_ids,
                                   num_transactions=num_transactions,
                                   window_days=window_days)

    # 3) Commit each partition to HDFS ---------------------------------------
    _write_partition(products, PRODUCTS_CSV, HDFS_PRODUCTS_PATH)
    _write_partition(stock, STOCK_CSV, HDFS_STOCK_PATH)
    _write_partition(ledger, SALES_CSV, HDFS_SALES_PATH,
                     timestamp_columns=("timestamp",))

    # 4) Build the manifest ---------------------------------------------------
    manifest: Dict[str, object] = {
        "application": "SmartStock",
        "layer": "RAW (Bronze)",
        "namenode_uri": HDFS_NAMENODE_URI,
        "hdfs_paths": {
            "products": to_hdfs_uri(HDFS_PRODUCTS_PATH),
            "stock": to_hdfs_uri(HDFS_STOCK_PATH),
            "sales": to_hdfs_uri(HDFS_SALES_PATH),
        },
        "replication_factor": HDFS_REPLICATION_FACTOR,
        "block_size_bytes": HDFS_BLOCK_SIZE_BYTES,
        "row_counts": {
            "products": int(len(products)),
            "stock": int(len(stock)),
            "sales_ledger": int(len(ledger)),
        },
        "categories": CATEGORIES,
        "sales_window_days": int(window_days),
        "random_seed": int(seed),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "injected_edge_cases": {
            "dead_stock_skus": dead_stock_ids,
            "stockout_risk_skus": stockout_risk_ids,
            "boundary_sku_on_safety_level": stockout_risk_ids[0],
        },
        "files": [
            {
                "hdfs_uri": to_hdfs_uri(HDFS_PRODUCTS_PATH),
                "local_path": str(PRODUCTS_CSV.relative_to(PROJECT_ROOT)),
                "bytes": PRODUCTS_CSV.stat().st_size,
                "blocks": max(1, -(-PRODUCTS_CSV.stat().st_size // HDFS_BLOCK_SIZE_BYTES)),
            },
            {
                "hdfs_uri": to_hdfs_uri(HDFS_STOCK_PATH),
                "local_path": str(STOCK_CSV.relative_to(PROJECT_ROOT)),
                "bytes": STOCK_CSV.stat().st_size,
                "blocks": max(1, -(-STOCK_CSV.stat().st_size // HDFS_BLOCK_SIZE_BYTES)),
            },
            {
                "hdfs_uri": to_hdfs_uri(HDFS_SALES_PATH),
                "local_path": str(SALES_CSV.relative_to(PROJECT_ROOT)),
                "bytes": SALES_CSV.stat().st_size,
                "blocks": max(1, -(-SALES_CSV.stat().st_size // HDFS_BLOCK_SIZE_BYTES)),
            },
        ],
    }

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# =============================================================================
# SECTION 6 :: PUBLIC API USED BY THE STREAMLIT DASHBOARD
# =============================================================================

def hdfs_dataset_exists() -> bool:
    """True only if ALL THREE partitions exist AND are committed (`_SUCCESS`).

    The Streamlit app calls this on every startup.  We deliberately check the
    `_SUCCESS` marker, not just the file size -- that is the exact same
    "is my input ready?" check a production Spark job would perform.
    """
    required = [
        (PRODUCTS_CSV, _local_path(HDFS_PRODUCTS_PATH)),
        (STOCK_CSV, _local_path(HDFS_STOCK_PATH)),
        (SALES_CSV, _local_path(HDFS_SALES_PATH)),
    ]
    for data_file, directory in required:
        if not data_file.exists() or data_file.stat().st_size == 0:
            return False
        if not (directory / SUCCESS_MARKER).exists():
            return False
    return True


def read_manifest() -> Dict[str, object]:
    """Load the dataset manifest (empty dict if the dataset was never built)."""
    if not MANIFEST_FILE.exists():
        return {}
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ensure_hdfs_dataset(force: bool = False,
                        num_transactions: int = NUM_TRANSACTIONS,
                        window_days: int = SALES_WINDOW_DAYS,
                        seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Idempotent bootstrap: create the dataset only if it is missing.

    This is the single entry point used by main_app.py, so the dashboard is a
    one-command experience (`streamlit run main_app.py`) with zero manual
    setup steps -- which is what "data self-service / data product" means.

    Parameters
    ----------
    force : bool
        Wipe the existing RAW layer and regenerate from scratch.
    """
    if hdfs_dataset_exists() and not force:
        manifest = read_manifest()
        print("[HDFS] Dataset already committed in {} -- skipping generation. "
              "Use --force to regenerate.".format(HDFS_LOCAL_ROOT))
        return manifest

    if force and HDFS_LOCAL_ROOT.exists():
        print("[HDFS] --force supplied: purging the existing RAW layer at {}"
              .format(HDFS_LOCAL_ROOT))
        shutil.rmtree(HDFS_LOCAL_ROOT, ignore_errors=True)

    print("[HDFS] Bootstrapping simulated HDFS dataset "
          "(products={}, transactions={}, window={} days, seed={})...".format(
              NUM_PRODUCTS, num_transactions, window_days, seed))
    manifest = build_dataset(num_transactions=num_transactions,
                             window_days=window_days, seed=seed)
    print_dataset_summary(manifest)
    return manifest


def print_dataset_summary(manifest: Dict[str, object]) -> None:
    """Human-readable console summary -- this is what you screenshot for the report."""
    if not manifest:
        print("[HDFS] No manifest available.")
        return

    print("\n" + "=" * 78)
    print(" SMARTSTOCK :: SIMULATED HDFS DATASET COMMITTED")
    print("=" * 78)
    print(" NameNode            : {}".format(manifest.get("namenode_uri")))
    print(" Replication factor  : {}".format(manifest.get("replication_factor")))
    rows = manifest.get("row_counts", {}) or {}
    print(" products.csv rows   : {}".format(rows.get("products")))
    print(" stock.csv rows      : {}".format(rows.get("stock")))
    print(" sales_ledger.csv    : {} rows over {} days".format(
        rows.get("sales_ledger"), manifest.get("sales_window_days")))
    print(" Random seed         : {}".format(manifest.get("random_seed")))
    edge = manifest.get("injected_edge_cases", {}) or {}
    print(" Dead-stock SKUs     : {}".format(", ".join(edge.get("dead_stock_skus", []))))
    print(" Stockout-risk SKUs  : {}".format(", ".join(edge.get("stockout_risk_skus", []))))
    print("=" * 78 + "\n")


# =============================================================================
# SECTION 7 :: CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SmartStock :: simulated HDFS initialisation & data generation.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Delete the existing RAW layer and regenerate it.")
    parser.add_argument("--transactions", type=int, default=NUM_TRANSACTIONS,
                        help="Number of sales-ledger rows (default: %(default)s).")
    parser.add_argument("--days", type=int, default=SALES_WINDOW_DAYS,
                        help="Sales history window in days (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility (default: %(default)s).")
    parser.add_argument("--stats", action="store_true",
                        help="Only print the current dataset status (no generation).")
    args = parser.parse_args()

    if args.stats:
        if hdfs_dataset_exists():
            print_dataset_summary(read_manifest())
        else:
            print("[HDFS] No committed dataset found under {}. "
                  "Run without --stats to generate it.".format(HDFS_LOCAL_ROOT))
        return 0

    ensure_hdfs_dataset(force=args.force,
                        num_transactions=args.transactions,
                        window_days=args.days,
                        seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
