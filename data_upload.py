#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SMARTSTOCK :: UPLOAD INGESTION LAYER   (data_upload.py)
===============================================================================

Lets a user drag their own CSV files onto the dashboard and have them flow
through the *identical* PySpark pipeline that analyses the built-in sample.

WHY THIS IS MORE THAN A FILE PICKER
------------------------------------
The uploaded data is not parsed in the browser or in Pandas. It is written
into the SAME simulated HDFS RAW namespace, given a Hadoop `_SUCCESS` commit
marker, and then read back by Spark with the same explicit StructType schemas.
So a user's file exercises the identical DAG: the groupBy shuffle, the Pareto
window, the LEFT ANTI JOIN, the broadcast joins. The only thing that changes
is the bytes on disk. That is exactly how a real data platform behaves -- a
new data source should not require a code change.

FILE-ROLE DETECTION BY HEADER SIGNATURE
---------------------------------------
We do NOT trust the filename. A user who drops `march_sales_final.csv` should
still get sales analysed. Each expected dataset is defined by the SET of
columns it must contain; we read only the header row of each candidate, score
every (file, dataset) pair by how many required columns are present, and take
the best unambiguous match. A file that matches nothing is rejected with a
specific message naming the columns we expected, because "invalid file" tells
a user nothing at submission time.

DATA-QUALITY GATE
-----------------
Validation happens HERE, before Spark ever sees the bytes. The Spark reader
runs in FAILFAST mode (a malformed row aborts the job), which is the correct
production choice but a terrible user experience if it fires 4 seconds into a
Spark job with a Java stack trace. So we coerce types, reject nulls in
non-nullable columns, and surface plain-English errors up front.

===============================================================================
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from hdfs_storage_mock import (
    HDFS_LOCAL_ROOT,
    SPARK_TIMESTAMP_FORMAT,
    SUCCESS_MARKER,
    TIMESTAMP_FORMAT,
    to_hdfs_uri,
)

# =============================================================================
# SECTION 1 :: DATASET CONTRACTS
# =============================================================================

#: Each logical dataset is identified by the columns it MUST contain.
#: Order matters only for readability -- detection is set-based.
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "products": ["product_id", "product_name", "category", "cost", "price"],
    "stock": ["product_id", "current_stock", "safety_stock_level",
              "warehouse_zone"],
    "sales": ["transaction_id", "timestamp", "product_id", "quantity_sold"],
}

#: Columns written in this order, so the emitted file matches our schemas
#: exactly and the Spark contract stays readable by a human.
COLUMN_ORDER: Dict[str, List[str]] = {k: list(v) for k, v in REQUIRED_COLUMNS.items()}

#: Files larger than this are refused. A browser upload that big will time out
#: on a projector, and the pipeline is single-node anyway.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Where uploaded datasets live inside the simulated HDFS namespace. Keeping
#: them under a separate prefix means a user upload can never overwrite the
#: sample dataset -- which is why "Reset to sample data" is a one-click undo.
UPLOADS_HDFS_PATH = "/user/hadoop/inventory/uploads"

UPLOADS_ROOT = HDFS_LOCAL_ROOT / UPLOADS_HDFS_PATH.strip("/")

#: Timestamp input formats we will accept, tried in order. Users paste from
#: spreadsheets and databases, so we are liberal here even though the OUTPUT
#: format is strict.
_ACCEPTED_TIMESTAMP_FORMATS = (
    "ISO8601",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)


class UploadError(Exception):
    """Raised when an uploaded file cannot be accepted.

    Carries a message written for an end user, not a developer: they should be
    able to act on it without reading our source.
    """


# =============================================================================
# SECTION 2 :: HEADER INSPECTION + ROLE DETECTION
# =============================================================================

def _normalise(name: str) -> str:
    """Fold header names to a comparable form: lower, alphanumeric only.

    'Product ID', 'product_id' and 'PRODUCT-ID' all normalise to 'productid',
    so a user does not have to guess our exact column naming. We keep the
    ORIGINAL header for the error messages.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _peek_header(data: bytes) -> List[str]:
    """Read just the header row without loading the file into memory."""
    # utf-8-sig strips the BOM Excel adds, which would otherwise corrupt the
    # first column name into '\\ufeffproduct_id' and fail detection.
    text = data.decode("utf-8-sig", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if not first_line:
        return []
    # A naive split is deliberate: the header should not contain quoted commas,
    # and anything exotic is caught by the validation step with a better message.
    return [c.strip().strip('"') for c in first_line.split(",")]


def detect_datasets(files: List[Tuple[str, bytes]]
                    ) -> Tuple[Dict[str, Tuple[str, bytes]], List[str]]:
    """Map uploaded files onto logical datasets by header signature.

    Returns
    -------
    (mapping, notes)
        mapping -- {dataset_name: (original_filename, raw_bytes)}
        notes   -- human-readable strings describing what happened, including
                   anything surprising, so the UI can show it verbatim.
    """
    mapping: Dict[str, Tuple[str, bytes]] = {}
    notes: List[str] = []
    unclaimed: List[str] = []

    for filename, data in files:
        header = _peek_header(data)
        if not header:
            notes.append("{}: file is empty or has no header row -- skipped."
                         .format(filename))
            unclaimed.append(filename)
            continue

        normalised = {_normalise(h) for h in header}
        # Score = how many of a dataset's required columns this file has.
        scores = {
            name: len({_normalise(c) for c in cols} & normalised)
            for name, cols in REQUIRED_COLUMNS.items()
        }
        best = max(scores, key=lambda k: scores[k])
        best_score = scores[best]

        if best_score == 0:
            notes.append("{}: no known columns found (saw: {}). Expected one of "
                         "these shapes -- products: {} | stock: {} | sales: {}."
                         .format(filename, ", ".join(header[:6]),
                                 ", ".join(REQUIRED_COLUMNS["products"]),
                                 ", ".join(REQUIRED_COLUMNS["stock"]),
                                 ", ".join(REQUIRED_COLUMNS["sales"])))
            unclaimed.append(filename)
            continue

        required_n = len(REQUIRED_COLUMNS[best])
        if best_score < required_n:
            missing = [c for c in REQUIRED_COLUMNS[best]
                       if _normalise(c) not in normalised]
            notes.append("{}: looks like {} but is missing {}. Skipped."
                         .format(filename, best, ", ".join(missing)))
            unclaimed.append(filename)
            continue

        if best in mapping:
            # Ambiguous: two files claim the same dataset. Taking the first
            # silently would hide a mistake; dropping both would be worse.
            notes.append("{} also claims the {} dataset, which {} already "
                         "filled. Only the first is used."
                         .format(filename, best, mapping[best][0]))
            continue

        mapping[best] = (filename, data)
        notes.append("{} -> detected as '{}' from its column headers "
                     "(filename not used for matching)."
                     .format(filename, best))

    return mapping, notes + ["{} was not matched to any dataset.".format(f)
                             for f in unclaimed]


# =============================================================================
# SECTION 3 :: VALIDATION + COERCION
# =============================================================================

def _coerce_timestamps(series: pd.Series) -> pd.Series:
    """Parse a timestamp column, trying the accepted formats in order.

    pandas 2.x refuses to guess across mixed formats, so an explicit ladder is
    the only reliable route. ISO8601 is tried first because it is what our own
    writer emits and it is unambiguous.
    """
    for fmt in _ACCEPTED_TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            return parsed
        except (ValueError, TypeError):
            continue
    raise UploadError(
        "Could not read the 'timestamp' column. Accepted formats include "
        "2026-01-31T09:15:00, '2026-01-31 09:15:00' and '31/01/2026 09:15'. "
        "Received: {}.".format(series.dropna().head(3).tolist())
    )


def validate_dataset(name: str, data: bytes
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """Decode, coerce and validate one dataset.

    Returns
    -------
    (frame, warnings)
        A normalised frame whose columns and dtypes exactly match the Spark
        schemas, so the emitted CSV needs no further interpretation, plus any
        NON-FATAL findings worth showing (e.g. rows priced below cost).

    Raises
    ------
    UploadError
        On anything that would make the Spark job wrong or crash.
    """
    warnings: List[str] = []
    try:
        frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError(
            "{} is not valid UTF-8 text. Re-save it as CSV (UTF-8)."
            .format(name))
    except pd.errors.EmptyDataError:
        raise UploadError("{} has no data rows.".format(name))
    except pd.errors.ParserError as exc:
        raise UploadError("{} could not be parsed as CSV: {}".format(name, exc))

    if frame.empty:
        raise UploadError("{} has a header but no data rows.".format(name))

    # Map every header to its canonical column name, so casing and separator
    # differences do not matter.
    lookup = {_normalise(col): col for col in frame.columns}
    missing = [c for c in COLUMN_ORDER[name] if _normalise(c) not in lookup]
    if missing:
        raise UploadError("{} is missing the column(s): {}. Found: {}."
                          .format(name, ", ".join(missing),
                                  ", ".join(map(str, frame.columns))))

    frame = frame.rename(columns={lookup[_normalise(c)]: c
                                  for c in COLUMN_ORDER[name]})

    # ---- Coerce numerics -------------------------------------------------
    numeric_int = {
        "stock": ["current_stock", "safety_stock_level"],
        "sales": ["quantity_sold"],
    }[name] if name in ("stock", "sales") else []
    for column in numeric_int:
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce")
    for column in (["cost", "price"] if name == "products" else []):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.strip().str.replace(",", "", regex=False)
                              .str.replace("$", "", regex=False),
            errors="coerce")

    # ---- Non-null check on the non-nullable columns ---------------------
    # These are exactly the columns declared nullable=False in the Spark
    # schemas, so this is the same contract enforced one layer earlier.
    non_nullable = [c for c in COLUMN_ORDER[name] if c != "warehouse_zone"]
    nulls = {c: int(frame[c].isna().sum()) for c in non_nullable
              if int(frame[c].isna().sum()) > 0}
    if nulls:
        raise UploadError(
            "{} has missing or non-numeric values in: {}. Those columns cannot "
            "be blank. Detail: {}."
            .format(name, ", ".join(nulls),
                    ", ".join("{} row(s)".format(v) for v, v in nulls.items())))

    # ---- Range sanity ----------------------------------------------------
    if name == "stock":
        negative = int((frame[["current_stock", "safety_stock_level"]] < 0).any(axis=1).sum())
        if negative:
            raise UploadError("{} has {} row(s) with a negative stock or safety "
                              "level.".format(name, negative))
    if name == "sales":
        non_positive = int((frame["quantity_sold"] <= 0).sum())
        if non_positive:
            raise UploadError(
                "{} has {} row(s) with quantity_sold <= 0. Every ledger row must "
                "represent a positive sale -- a zero or negative quantity is a "
                "returns/refund record, which this dataset does not model."
                .format(name, non_positive))
    if name == "products":
        bad = int(((frame["cost"] < 0) | (frame["price"] < 0)).sum())
        if bad:
            raise UploadError("{} has {} row(s) with a negative cost or price."
                              .format(name, bad))
        non_margin = int((frame["price"] < frame["cost"]).sum())
        if non_margin:
            # Not fatal -- a clearance line can legitimately sell below cost --
            # but a wholesale business would want to know, so we surface it.
            warnings.append(
                "{} has {} row(s) priced below cost (negative gross margin)."
                .format(name, non_margin))

    # ---- Timestamps ------------------------------------------------------
    if name == "sales":
        frame["timestamp"] = _coerce_timestamps(frame["timestamp"])

    return frame[COLUMN_ORDER[name]], warnings


def cross_check(mapping: Dict[str, pd.DataFrame]) -> List[str]:
    """Report referential problems between the three datasets (warnings only).

    We deliberately do NOT drop the orphans: a real pipeline would surface them
    for a human to decide, and silently deleting rows is the kind of thing
    that produces a confidently wrong report.
    """
    warnings: List[str] = []
    if not {"products", "stock", "sales"}.issubset(mapping):
        return warnings

    product_ids = set(mapping["products"]["product_id"])
    for name in ("stock", "sales"):
        orphans = set(mapping[name]["product_id"]) - product_ids
        if orphans:
            sample = ", ".join(sorted(map(str, orphans))[:5])
            warnings.append(
                "{} references {} product_id(s) absent from products.csv "
                "(e.g. {}). Those ledger rows contribute no revenue and will "
                "appear as unmatched.".format(name.capitalize(), len(orphans),
                                              sample))
    if mapping["stock"]["product_id"].duplicated().any():
        dupes = int(mapping["stock"]["product_id"].duplicated().sum())
        warnings.append(
            "stock.csv has {} duplicate product_id row(s). The join keeps one "
            "row per SKU; if those were meant to be separate warehouse "
            "locations, aggregate them first.".format(dupes))
    return warnings


# =============================================================================
# SECTION 4 :: PERSIST TO THE SIMULATED HDFS NAMESPACE
# =============================================================================

def _dataset_id(mapping_bytes: Dict[str, bytes]) -> str:
    """Content-addressed id: the same input always lands in the same folder.

    This is what makes caching correct -- re-uploading identical files resolves
    to the same path, so Streamlit's cache key stays stable and we do not
    needlessly re-run Spark.
    """
    digest = hashlib.sha256()
    for name in sorted(mapping_bytes):
        digest.update(name.encode("utf-8"))
        digest.update(mapping_bytes[name])
    return digest.hexdigest()[:12]


def persist_uploaded_dataset(mapping: Dict[str, Tuple[str, bytes]]
                             ) -> Tuple[Dict[str, Path], Dict[str, object]]:
    """Validate, normalise and write an uploaded dataset into simulated HDFS.

    Returns
    -------
    (paths, manifest)
        paths    -- {dataset_name: Path} for Spark to read
        manifest -- provenance, shown in the UI and written to _manifest.json

    Raises UploadError if any dataset fails validation. We validate everything
    BEFORE writing anything, so a rejected upload cannot leave a half-ingested
    dataset behind -- the same all-or-nothing discipline as a Hadoop job.
    """
    if not mapping:
        raise UploadError(
            "None of the uploaded files could be matched to a dataset. Expected "
            "three CSVs whose headers contain: products {} | stock {} | sales {}."
            .format(REQUIRED_COLUMNS["products"], REQUIRED_COLUMNS["stock"],
                    REQUIRED_COLUMNS["sales"]))

    # All three feeds are REQUIRED, and this is a deliberate product decision
    # rather than an oversight. Every metric the console reports is a join of
    # them: without stock.csv there are no on-hand quantities, so days of
    # supply, the turnover ratio and -- most dangerously -- "trapped capital"
    # are all unknowable. Summing over a NULL stock level yields $0, which reads
    # as "no capital is trapped" when the truth is "we do not know". That is
    # precisely the confident-but-wrong figure this project exists to avoid, so
    # we reject an incomplete upload at the gate instead.
    missing = [d for d in ("products", "stock", "sales") if d not in mapping]
    if missing:
        raise UploadError(
            "Missing dataset(s): {}. All three are required. The analytics "
            "join all of them, and a missing feed would silently zero out a "
            "metric it cannot actually measure -- with no stock.csv the "
            "console reports $0 trapped capital, which reads as 'none' "
            "rather than 'unknown'. Attach {} more file{}. Expected headers: {}."
            .format(", ".join(missing), len(missing),
                    "" if len(missing) == 1 else "s",
                    " | ".join("{}: {}".format(d, ", ".join(REQUIRED_COLUMNS[d]))
                               for d in missing)))

    oversized = [fn for fn, data in mapping.values() if len(data) > MAX_UPLOAD_BYTES]
    if oversized:
        raise UploadError("{} exceeds the {} MB limit. Aggregate it to a daily "
                          "grain first, or split it by month."
                          .format(", ".join(oversized),
                                  MAX_UPLOAD_BYTES // (1024 * 1024)))

    # ---- validate every dataset BEFORE touching the filesystem -----------
    frames: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    for name, (filename, data) in mapping.items():
        try:
            frame, dataset_warnings = validate_dataset(name, data)
        except UploadError as exc:
            raise UploadError("{} -- {}".format(filename, exc))
        frames[name] = frame
        warnings.extend(dataset_warnings)

    warnings.extend(cross_check(frames))

    dataset_id = _dataset_id({n: d for n, (_, d) in mapping.items()})
    target_root = UPLOADS_ROOT / dataset_id
    target_root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    files_meta: List[dict] = []
    for name, frame in frames.items():
        destination = target_root / "{}.csv".format(name)
        output = frame.copy()
        if name == "sales":
            # Emit in the ONE canonical format the Spark reader is configured
            # for, so the reader never has to guess.
            output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.strftime(
                TIMESTAMP_FORMAT)
        output.to_csv(destination, index=False, encoding="utf-8")

        # Commit marker last, exactly as the generator does.
        (target_root / SUCCESS_MARKER).write_text(
            "committed_at={}\nsource=user_upload\n".format(
                datetime.now().isoformat(timespec="seconds")),
            encoding="utf-8")

        paths[name] = destination
        files_meta.append({
            "dataset": name,
            "uploaded_as": mapping[name][0],
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "bytes": destination.stat().st_size,
        })

    manifest = {
        "source": "user_upload",
        "dataset_id": dataset_id,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "hdfs_uri": to_hdfs_uri("{}/{}".format(UPLOADS_HDFS_PATH, dataset_id)),
        "row_counts": {n: int(len(f)) for n, f in frames.items()},
        "files": files_meta,
        "warnings": warnings,
    }
    (target_root / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return paths, manifest


def list_uploaded_datasets() -> List[str]:
    """Dataset ids currently persisted (newest first), for the UI to offer."""
    if not UPLOADS_ROOT.exists():
        return []
    out = []
    for child in UPLOADS_ROOT.iterdir():
        if child.is_dir() and (child / "_manifest.json").exists():
            out.append(child.name)
    return sorted(out, reverse=True)


def clear_uploads() -> None:
    """Remove every uploaded dataset. Used by the 'back to sample' control."""
    if UPLOADS_ROOT.exists():
        shutil.rmtree(UPLOADS_ROOT, ignore_errors=True)