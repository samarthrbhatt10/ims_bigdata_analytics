# 📦 SmartStock — HDFS & PySpark Enterprise Inventory Analytics Suite

> A production-style Big Data inventory analytics platform built with
> **Hadoop HDFS** (simulated), **Apache Spark / PySpark**, and **Streamlit**.

SmartStock ingests a raw product / stock / sales dataset from a simulated HDFS
namespace, runs a full distributed analytics pipeline (ABC classification,
inventory turnover ratio, supply-chain alert flags) on PySpark, and serves the
result as a live enterprise dashboard.

---

## 📸 What the dashboard gives you

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Hero header** | Title + a one-line explanation of the pipeline architecture, with live system pills (Spark version, row counts, window). |
| 2 | **KPI metrics bar** | Total Enterprise Revenue · Critical Stockout Reorder Alerts · Dead Stock Items · Overall Inventory Turnover Ratio · SKUs in scope. |
| 3 | **Sidebar filters** | Product Category · ABC Class · Warehouse Zone · Alert Severity · min revenue · "flagged items only". Every widget re-slices the entire dashboard. |
| 4 | **Horizontal bar chart** | Revenue generated per product category. |
| 5 | **Pareto curve** | Revenue per SKU + cumulative revenue %, with the 80% / 95% ABC cut lines. |
| 6 | **ABC donut** | Percentage breakdown of Class A / B / C stock items. |
| 7 | **Severity donut** | How the current selection splits by urgency (CRITICAL / HIGH / MEDIUM / OK). |
| 8 | **Restock action table** | Prioritised "Immediate Restock Actions Required" — sortable, searchable, downloadable. |
| 9 | **Full analytical frame** | All 60 SKUs × 29 analytical columns for auditing. |
| 10 | **Architecture panel** | Live Spark action log, HDFS manifest, and a viva Q&A crib sheet. |

---

## 🗂️ Project structure

```
ims_bigdata_analytics/
├── hdfs_storage_mock.py          # File 1: HDFS namespace + synthetic data generator
├── pyspark_analytics_engine.py   # File 2: PySpark analytics engine (the Big Data core)
├── main_app.py                   # File 3: Streamlit enterprise dashboard
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
└── hdfs/                         # AUTO-GENERATED simulated HDFS tree (git-ignored)
    └── user/hadoop/inventory/raw/
        ├── _manifest.json        # row counts, injected edge cases, seed
        ├── products/products.csv + _SUCCESS
        ├── stock/stock.csv + _SUCCESS
        └── sales/sales_ledger.csv + _SUCCESS
```

### The three source files

| File | Role | Key concepts demonstrated |
|------|------|---------------------------|
| `hdfs_storage_mock.py` | Creates `hdfs/user/hadoop/inventory/raw/{products,stock,sales}/`, writes 60 products, 60 stock rows, 2,500 ledger rows over 90 days. | HDFS as WORM storage, block/replication semantics, `_SUCCESS` commit markers, deterministic seeding, Pareto-skewed demand, injected edge cases. |
| `pyspark_analytics_engine.py` | Class `PySparkInventoryEngine` → `run_pipeline()`. | `StructType` schema-on-read, lazy evaluation, lineage graphs, actions vs transformations, window functions, `LEFT ANTI JOIN`, broadcast joins, `.cache()`, AQE, Arrow. |
| `main_app.py` | Streamlit dashboard with auto-bootstrap and caching. | `st.cache_resource` for a single shared SparkSession, driver-side filtering, Plotly charts, custom CSS KPI cards. |

---

## 🧪 The three analytical modules

### 1. ABC Inventory Analysis (80 / 15 / 5)
Ranks SKUs by revenue, computes the **cumulative revenue share** with a Spark
window function, then cuts at 80% and 95%:
- **Class A** (High Value) — the top 80% of revenue
- **Class B** (Medium Value) — the next 15%
- **Class C** (Low Value) — the remaining 5%

> **Subtle rule (be ready to defend it):** a SKU is classified by the cumulative
> share *before* adding itself, so the SKU that pushes the total past 80% stays
> in class **A**. We use an exclusive frame
> `rowsBetween(Window.unboundedPreceding, -1)`.

### 2. Inventory Turnover Ratio (ITR)
```
COGS                = SUM(quantity_sold × cost)
opening_stock (est) = current_stock + units_sold      (no receipts in-window)
avg_inventory_value = cost × (opening_stock + current_stock) / 2
ITR (period)        = COGS / avg_inventory_value
ITR (annualised)    = ITR × 365 / window_days
```
Also derived: `gross_margin`, `margin_pct`, `avg_daily_demand`, `days_of_supply`.

### 3. Supply Chain Alert Flags
- **Stockout Risk** → `current_stock <= safety_stock_level` (inclusive `<=`; one
  boundary SKU sits exactly on the buffer to prove it).
- **Dead Stock** → present in the product master but **absent** from the sales
  ledger, detected with a `LEFT ANTI JOIN`.
- **Severity** → CRITICAL (both) > HIGH (stockout) > MEDIUM (dead) > OK.
- **Reorder plan** → `(s, S)` policy: top up to `2 × safety_stock_level`.

---

## ⚙️ Prerequisites

| Requirement | Windows | macOS |
|---|---|---|
| **Python** | 3.9 – 3.13 (3.11/3.12 recommended) | 3.9 – 3.13 (3.11/3.12 recommended) |
| **Java (JDK)** | **17+** (Temurin/Zulu) | **17+** (Temurin/Zulu) |
| Python packages | `pip install -r requirements.txt` | `pip3 install -r requirements.txt` |
| Git | optional | optional |

> ⚠️ **Java is mandatory** — PySpark runs on the JVM. `java -version` must work
> before you install anything. PySpark 4.x needs **Java 17+**.
>
> Note: Streamlit **1.49+** is required (it replaced the `use_container_width`
> argument with `width="stretch"`); `pip install -r requirements.txt` handles it.

### Check Java

**Windows (PowerShell)**
```powershell
java -version
```
Expect `openjdk version "17.x"` or newer.

**macOS (Terminal)**
```bash
java -version
```
If not installed: `brew install openjdk@17` then
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

---

## 🚀 How to run (exact steps)

### 🪟 Windows 10/11 — PowerShell

```powershell
# 1. Open PowerShell in the project folder (right-click > "Open in Terminal")
cd C:\path\to\ims_bigdata_analytics

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#    (if PowerShell blocks the script, run this once:)
#    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify Java is visible to PySpark
java -version

# 5. (Optional) pre-generate the simulated HDFS layer
python hdfs_storage_mock.py

# 6. Launch the dashboard
streamlit run main_app.py
```

Streamlit prints `http://localhost:8501` → open it in Chrome/Edge. First load
takes ~25 s (JVM warm-up); every filter afterwards is instant.

### 🍎 macOS — Terminal (zsh)

```bash
# 1. Open Terminal and cd into the project folder
cd ~/path/to/ims_bigdata_analytics

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

# 4. Verify Java is visible to PySpark
java -version

# 5. (Optional) pre-generate the simulated HDFS layer
python3 hdfs_storage_mock.py

# 6. Launch the dashboard
streamlit run main_app.py
```

On Apple Silicon (M1/M2/M3) the same steps work; if `pyspark` complains about
the architecture, run under Rosetta or reinstall with
`arch -x86_64 python3 -m pip install pyspark`.

### ▶️ Batch mode (no Streamlit — useful for the viva demo)

```bash
# Run the full pipeline in the terminal and print the KPI summary
# (bootstraps the simulated HDFS layer automatically if it is missing)
python pyspark_analytics_engine.py

# Print the optimised physical plan / lineage graph
python pyspark_analytics_engine.py --explain-plan

# Export the unified 60-row frame to CSV
python pyspark_analytics_engine.py --export inventory_analytics.csv

# Fail instead of generating the dataset (useful in CI)
python pyspark_analytics_engine.py --no-bootstrap
```

### 🧹 Reset / regenerate the data

```bash
python hdfs_storage_mock.py --force           # wipe + regenerate the HDFS layer
python hdfs_storage_mock.py --stats            # print what is currently committed
python hdfs_storage_mock.py --transactions 2500000 --days 365   # scale up the demo
```

> Scaling up is a great viva move: the code is unchanged, only the constants move.

---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'JAVA_HOME' is not set` / `Gateway java.lang.ClassNotFoundException` | Java missing / wrong version | Install **JDK 17**, then set `JAVA_HOME` (see below) |
| `Unsupported class file major version` | Java too old for Spark 4 | Upgrade to Java 17+ |
| `INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION` | Strftime pattern passed to Spark's `timestampFormat` | Already handled: we use the Java pattern `yyyy-MM-dd'T'HH:mm:ss` |
| `Py4JJavaError: ... Python 3.13 not supported` | PySpark 3.5 on Python 3.13 | `pip install -U "pyspark>=4.0"` |
| `Did not find winutils.exe` (Windows) | Hadoop utilities missing | **Harmless warning** in local mode — ignore it |
| Dashboard shows empty KPIs | Filters exclude everything | Click **Reset all filters** in the sidebar |
| Port 8501 already in use | Another Streamlit instance | `streamlit run main_app.py --server.port 8502` |

### Setting `JAVA_HOME`

**Windows (PowerShell, run as Administrator) — persists for the user**
```powershell
[Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.0.8.1", "User")
# then close and reopen the terminal
```

**macOS (~/.zshrc)**
```bash
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
source ~/.zshrc
```

---

## 🎓 Viva cheat sheet (the five questions)

1. **Why Spark for 2,500 rows?** The architecture is identical at 2.5 billion
   rows — same DAG, shuffles, Catalyst optimisations. Scaling the data does not
   change the code; that is the point of the DataFrame API.
2. **Lazy evaluation?** `filter/join/groupBy` only append to the lineage DAG.
   Execution starts at the first action (`toPandas`, `count`, `show`, `write`).
   The console logs every action with its duration as proof.
3. **Broadcast vs shuffle?** `F.broadcast()` on the 60-row dimensions avoids
   shuffling them; the only unavoidable shuffle is the `groupBy`.
4. **How is dead stock found?** `LEFT ANTI JOIN` — one hash anti-join, versus the
   `COUNT(*) = 0` anti-pattern which forces two extra aggregations.
5. **Where is the 80% cut?** In a windowed running total of revenue
   (`rowsBetween(unboundedPreceding, -1)`), classifying by the share *before*
   each SKU so the SKU that crosses 80% stays in class A.

**Two design decisions worth highlighting:**
- The **product master is the LEFT spine** of every join — an inner join would
  silently drop the five dead-stock SKUs the dashboard must alert on.
- The **`.cache()` on the sales aggregation** exists because that node feeds
  three downstream consumers; without it Spark would re-scan and re-shuffle the
  ledger three times.

---

## 📄 License

MIT — academic / educational project.
