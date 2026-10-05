# 📦 SmartStock — HDFS & PySpark Enterprise Inventory Analytics Suite

> Inventory analytics on **Hadoop HDFS** (simulated), **Apache Spark / PySpark**
> and **Streamlit**.

A simulated HDFS RAW layer is parsed with explicit schemas, analysed by a PySpark
pipeline (ABC classification, inventory turnover ratio, supply-chain alert flags),
and served as a live enterprise dashboard.

**Verified output:** 60 SKUs · \$3,931,910 revenue · 29.71% margin · ITR 1.68×/yr ·
5 stockout alerts · 5 dead-stock items.

---

## Features

- **KPI bar** — revenue, stockout reorder alerts, dead stock, turnover ratio, SKUs in scope
- **Sidebar filters** — category, ABC class, warehouse zone, severity, min revenue, flagged-only
- **Charts** — revenue by category (bar), Pareto curve with the 80%/95% cuts, ABC donut, severity donut
- **Restock action table** — interactive, sortable, downloadable
- **Audit frame** — all 60 SKUs × 34 analytical columns
- **Architecture panel** — live Spark action log, HDFS manifest, viva Q&A

---

## Project structure

```
ims_bigdata_analytics/
├── hdfs_storage_mock.py          # File 1 · HDFS namespace + data generator
├── pyspark_analytics_engine.py   # File 2 · PySpark analytics engine
├── main_app.py                   # File 3 · Streamlit dashboard
├── requirements.txt
└── hdfs/                         # AUTO-GENERATED (git-ignored)
    └── user/hadoop/inventory/raw/
        ├── _manifest.json        # row counts, edge cases, seed
        ├── products/products.csv + _SUCCESS
        ├── stock/stock.csv + _SUCCESS
        └── sales/sales_ledger.csv + _SUCCESS
```

| File | Demonstrates |
|---|---|
| `hdfs_storage_mock.py` | HDFS as WORM storage, blocks + replication, `_SUCCESS` commit markers, deterministic seeding, Pareto-skewed demand |
| `pyspark_analytics_engine.py` | `StructType` schema-on-read, lazy evaluation, lineage graphs, actions vs transformations, window functions, `LEFT ANTI JOIN`, broadcast joins, `.cache()`, AQE, Arrow |
| `main_app.py` | `st.cache_resource` for one shared SparkSession, driver-side filtering, Plotly, custom CSS KPI cards |

---

## The three analytical modules

**1. ABC Inventory Analysis (80 / 15 / 5)** — rank SKUs by revenue, accumulate the
cumulative revenue share with a window function, cut at 80% and 95%.

> The subtle rule: a SKU is classified by the share *before* adding itself, so the
> SKU that pushes the total past 80% stays in class **A**. That "before" value needs
> an **exclusive** frame: `rowsBetween(Window.unboundedPreceding, -1)`.

**2. Inventory Turnover Ratio**
```
COGS                = SUM(quantity_sold × cost)
opening_stock (est) = current_stock + units_sold      (no receipts in-window)
avg_inventory_value = cost × (opening_stock + current_stock) / 2
ITR (period)        = COGS / avg_inventory_value
ITR (annualised)    = ITR × 365 / window_days
```
Also derived: `gross_margin`, `margin_pct`, `avg_daily_demand`, `days_of_supply`.

**3. Supply Chain Alert Flags**
- **Stockout Risk** → `current_stock <= safety_stock_level` (one boundary SKU sits exactly on the buffer to prove the `<=`)
- **Dead Stock** → in the product master, absent from the ledger, via `LEFT ANTI JOIN`
- **Severity** → CRITICAL > HIGH > MEDIUM > OK
- **Reorder plan** → `(s, S)` policy: top up to `2 × safety_stock_level`

---

## Prerequisites

| | Windows | macOS |
|---|---|---|
| Python | 3.9–3.13 (3.11/3.12 recommended) | 3.9–3.13 (3.11/3.12 recommended) |
| **Java (JDK)** | **17+** (Temurin/Zulu) | **17+** (Temurin/Zulu) |

> ⚠️ **Java is mandatory** — PySpark runs on the JVM. Run `java -version` first.
> Streamlit 1.49+ is required; `requirements.txt` pins it.

---

## Run it

### 🪟 Windows — PowerShell

```powershell
cd C:\path\to\ims_bigdata_analytics

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if PowerShell blocks the script:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

pip install -r requirements.txt
java -version
streamlit run main_app.py
```

### 🍎 macOS — Terminal

```bash
cd ~/path/to/ims_bigdata_analytics

python3 -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt
java -version          # not installed? brew install openjdk@17
streamlit run main_app.py
```

Open `http://localhost:8501`. First load ≈ 20 s (JVM warm-up); filters after that
are instant.

### Batch mode (good for the viva demo)

```bash
python pyspark_analytics_engine.py                      # KPI summary in the terminal
python pyspark_analytics_engine.py --explain-plan       # print the lineage graph
python pyspark_analytics_engine.py --export out.csv     # export the 60-row frame
```

### Regenerate / scale the data

```bash
python hdfs_storage_mock.py --force                     # wipe + regenerate
python hdfs_storage_mock.py --stats                     # show what is committed
python hdfs_storage_mock.py --transactions 2500000      # 2.5M rows, same code
```

> Scaling up is the best viva move: only the constants change, the DAG does not.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'JAVA_HOME' is not set` / `ClassNotFoundException` | Java missing or too old | Install **JDK 17** and set `JAVA_HOME` |
| `Unsupported class file major version` | Java < 17 with Spark 4 | Upgrade to Java 17+ |
| `DATETIME_PATTERN_RECOGNITION` | strftime pattern passed to Spark | Handled: we use the Java pattern `yyyy-MM-dd'T'HH:mm:ss` |
| `Python 3.13 not supported` | PySpark 3.5 on Python 3.13 | `pip install -U "pyspark>=4.0"` |
| `Did not find winutils.exe` | Hadoop utilities missing | **Harmless** warning in local mode — ignore |
| Empty KPIs | Filters exclude everything | **Reset all filters** in the sidebar |
| Port 8501 busy | Another instance | `streamlit run main_app.py --server.port 8502` |

**Set `JAVA_HOME` if needed**

```powershell
# Windows (Administrator PowerShell)
[Environment]::SetEnvironmentVariable("JAVA_HOME","C:\Program Files\Eclipse Adoptium\jdk-17.0.8.1","User")
```
```bash
# macOS (~/.zshrc)
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc && source ~/.zshrc
```

---

## Viva notes

A five-question crib sheet is built into the dashboard under
**Architecture → 🎓 Viva notes**, and every concept is explained at the point of use
in the source. The two design decisions worth highlighting:

- The **product master is the LEFT spine** of every join — an inner join would
  silently drop the five dead-stock SKUs the dashboard exists to alert on.
- **`.cache()` on the sales aggregation** because that node feeds three downstream
  consumers; without it Spark would re-scan and re-shuffle the ledger three times.

---

MIT — academic / educational project.
