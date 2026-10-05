# SmartStock — Inventory Control Tower

> Supply-chain analytics on **Hadoop HDFS** (simulated), **Apache Spark / PySpark**
> and **Streamlit**.

An operations console for inventory planners. A simulated HDFS RAW layer is
parsed with explicit schemas and analysed by a PySpark pipeline that answers
three questions, in order of cost: **where is the money leaking, what must I do
today, and which parameters are wrong.**

**Verified output:** 60 SKUs · \$3.93M revenue · \$235K net exposure (5.98% of
revenue) · \$212K trapped capital · \$23K lost margin · 5 stockouts · 5 dead-stock
SKUs · 4 under-buffered · 5 demand-censored.

---

## The business problems this solves

These are real operational problems that retailers and manufacturers pay to
address, not analytics for their own sake.

| Problem | What the pipeline does |
|---|---|
| **Stockouts with no visibility** | Flags every SKU at or below its safety level, and prices the unmet demand in lost margin rather than reporting a bare count |
| **Cash frozen on the shelf** | Values dead stock at *cost*, not retail, and ranks it against live demand in one exposure table |
| **Static safety levels** | Rebuilds a daily-demand series from the raw event stream, derives each SKU's volatility (CV), and compares the configured buffer against a 95% service-level requirement |
| **Censored demand** | Recognises that an out-of-stock SKU records *no* demand rather than zero demand, and refuses to publish a confident recommendation it cannot support |
| **Reporting with no traceability** | Logs every Spark action with its measured duration, and keeps the full 60-SKU × 42-column frame available for audit |

### Demand censoring — the point worth defending

A SKU that is out of stock does not record zero demand; it records **no demand at
all**, because the customer could not buy it. Its observed mean is therefore
biased downwards, and a forecast trained on it will under-order forever. This is
the classic censored-demand problem and the most common root cause of chronic
stockouts in real retail.

We cannot correct it exactly without transactional data from the out-of-stock
periods. So the engine does the honest thing: it labels those SKUs
**Censored**, sets the gap to `NULL`, and excludes them from the buffer chart —
rather than publishing a precise-looking wrong number. On this dataset that is
exactly the 5 stockout SKUs.

---

---

## Project structure

```
ims_bigdata_analytics/
├── hdfs_storage_mock.py          # File 1 · HDFS namespace + data generator
├── pyspark_analytics_engine.py   # File 2 · PySpark analytics engine
├── main_app.py                   # File 3 · Streamlit dashboard
├── requirements.txt
├── .streamlit/config.toml        # pins the light theme (see Accessibility)
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
| `main_app.py` | `st.cache_resource` for one shared SparkSession, driver-side filtering of a 60-row payload, Plotly, WCAG-AA design system |

---

## The five analytical modules

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

**4. Demand volatility & service-level buffer** — the forward-looking layer.
Rebuilds a daily-demand series from the raw event stream, then:
```
CV                   = stddev(daily demand) / mean(daily demand)
recommended_safety   = z × sigma × sqrt(lead_time)     z = 1.65, lead = 21 days
forecast_next_7d     = mean × 7   ±   z × sigma × sqrt(7)
```
The `sqrt(lead_time)` term is the important one: daily demand noise is
independent, so **variance** accumulates linearly over the lead time while
**standard deviation** grows only as its square root. Under-ordering this term is
the textbook cause of chronic stockouts.

**5. Economic impact** — attaches a dollar figure to every alert, because a
dashboard saying "5 stockout risks" gets ignored while one saying "these cost
\$23K in margin, and \$212K is frozen on dead stock" gets a budget.
```
lost_margin   = unmet_units × margin_rate × price
trapped       = current_stock × cost                (cost, not retail)
net_exposure  = lost_margin + trapped
```

---

## Prerequisites

| | Windows | macOS |
|---|---|---|
| Python | 3.9–3.13 (3.11/3.12 recommended) | 3.9–3.13 (3.11/3.12 recommended) |
| **Java (JDK)** | **17+** (Temurin/Zulu) | **17+** (Temurin/Zulu) |

> **Java is mandatory** — PySpark runs on the JVM. Run `java -version` first.
> Streamlit 1.49+ is required; `requirements.txt` pins it.

---

## Run it

### Windows — PowerShell

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

### macOS — Terminal

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

### How the console is organised

Numbered sections, in the order a planner works through them:

| § | Section | Answers |
|---|---|---|
| 01 | Capital at risk | How much is this costing us? |
| 02 | Action queue | What must I do today? (ranked by net exposure) |
| 03 | Composition of the exposure | Which failure mode — buy more, or liquidate? |
| 04 | Portfolio structure | Where is capital held vs generated? |
| 05 | Demand risk & safety-stock parameters | Which parameters are misconfigured? |
| 06 | Safety-stock parameter review | One row per SKU, with the gap |

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

## Design system & accessibility

Built as an operations console, not a marketing page: neutral greys carry the
structure, and **colour is reserved for status** — so a red value always means
"act on this" and never merely "this is a header".

| Element | Ratio |
|---|---|
| App bar brand, KPI values, section headings | 13.9–14.8:1 (AAA) |
| Sidebar text and buttons | 8.6–12.6:1 (AAA) |
| Tab panel body, table headers, callouts | 10.3–18.2:1 (AAA) |
| KPI labels, tile footers, section hints | 6.5–7.3:1 (AA) |
| All 98 chart text nodes (axes, ticks, legends, titles) | ≥ 4.5:1 (AA) |

Two non-obvious details worth knowing if you edit the stylesheet:

- **Plotly axis titles do not inherit the layout font.** They fall back to a
  mid-grey measuring 3.7:1, so every `title=` in a chart is passed as
  `dict(text=..., font=dict(color=...))` explicitly.
- **`st.dataframe` sanitises HTML**, so injected `<span class="chip">` markup
  is stripped to plain text. Table status uses a leading glyph (`▲` stockout,
  `■` dead stock) which survives sanitisation and still reads correctly in an
  exported CSV.

**Why `.streamlit/config.toml` exists.** Streamlit follows your operating
system's dark-mode setting. In dark mode it renders body text near-white
(`#fafafa`), which on a light page is unreadable. The config pins
`base = "light"` and brands the colours; `main_app.py` additionally forces the
surface and text colour in CSS, so the console stays readable even if someone
toggles the theme from the Streamlit menu.

To restyle, edit `[theme]` in `.streamlit/config.toml` and restart.

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
| Text looks faint / wrong colours | A dark OS theme is overriding Streamlit | Confirm `.streamlit/config.toml` exists and restart the app |

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

A six-question crib sheet is built into the dashboard under **Pipeline → Viva
notes**, and every concept is explained at the point of use in the source.

**Two design decisions worth highlighting**
- The **product master is the LEFT spine** of every join — an inner join would
  silently drop the five dead-stock SKUs the console exists to surface.
- **`.cache()` on the sales aggregation** because that node feeds three downstream
  consumers; without it Spark would re-scan and re-shuffle the ledger three times.

**Two domain decisions**
- The **exclusive window frame** in the ABC cut, so the SKU that crosses 80%
  stays in class A.
- **Refusing to publish a recommendation for censored SKUs.** Saying "we cannot
  measure this reliably" is a stronger engineering position than emitting a
  confident number that is known to be biased.

---

MIT — academic / educational project.
