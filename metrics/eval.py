"""Utilities for evaluating recommendation predictions.

The module expects prediction files with ``user_id`` and ``item_id`` columns and
an optional ``score`` or ``rank`` column that describes the ordering of
recommendations per user. Ground-truth files must contain at least
``user_id`` and ``item_id`` columns. CSV, whitespace-delimited ``.txt`` and
Parquet files are supported for both predictions and ground truth.

Example
-------
Evaluate a predictions file against the validation split and report the mean
NDCG@20::

    python metrics/eval.py --preds path/to/preds.csv --split val --k 20

You can also point directly to a ground-truth file::

    python metrics/eval.py --preds path/to/preds.csv --ground-truth interim_data/test_split.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

Record = Dict[str, str]


def _coerce_int(value: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Could not parse integer for {field!r}: {value!r}") from exc


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Could not parse score value: {value!r}") from exc


def _load_parquet(path: Path) -> List[Record]:
    spec = importlib.util.find_spec("pandas")
    if spec is None:
        raise ImportError(
            "Reading Parquet files requires pandas. Install pandas to enable Parquet support."
        )
    # Importing here avoids an unconditional dependency on pandas.
    import pandas as pd  # type: ignore

    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _load_csv(path: Path) -> List[Record]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_txt(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            columns = stripped.split()
            if len(columns) < 2:
                raise ValueError(
                    f"Expected at least two columns per line in {path}, found: {columns!r}"
                )
            row: Record = {"user_id": columns[0], "item_id": columns[1]}
            if len(columns) >= 3:
                row["score"] = columns[2]
            rows.append(row)
    return rows


def load_rows(path: Path) -> List[Record]:
    """Load table-like data from ``path``.

    Supported formats are CSV (with headers), whitespace-delimited ``.txt``
    files (without headers) and Parquet. Columns are returned as raw strings and
    parsed by downstream helpers.
    """

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".txt":
        return _load_txt(path)
    if suffix == ".parquet":
        return _load_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}: {suffix}")


def _pick_value(row: Record, keys: Sequence[str], field_name: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    raise KeyError(f"Missing required column {field_name!r} in row: {row}")


def load_ground_truth(path: Path) -> Dict[int, Set[int]]:
    """Load ground-truth interactions into a mapping of user -> items."""

    rows = load_rows(path)
    truth: DefaultDict[int, Set[int]] = defaultdict(set)
    for row in rows:
        user = _coerce_int(_pick_value(row, ["user_id", "user"], "user_id"), "user_id")
        item = _coerce_int(_pick_value(row, ["item_id", "item"], "item_id"), "item_id")
        truth[user].add(item)
    return dict(truth)


def load_predictions(path: Path) -> Dict[int, List[int]]:
    """Load and order predictions per user.

    Predictions may contain a ``score`` or ``rank`` column. If ``rank`` is
    present, lower values are treated as better. If ``score`` is present, higher
    values are preferred. When neither column is available, the original file
    order is preserved for each user.
    """

    rows = load_rows(path)
    grouped: DefaultDict[int, List[Tuple[int, Optional[float], Optional[int], int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        user = _coerce_int(_pick_value(row, ["user_id", "user"], "user_id"), "user_id")
        item = _coerce_int(_pick_value(row, ["item_id", "item"], "item_id"), "item_id")
        score = _coerce_float(row.get("score"))
        rank = None
        rank_value = row.get("rank")
        if rank_value not in (None, ""):
            rank = _coerce_int(rank_value, "rank")
        grouped[user].append((item, score, rank, idx))

    ordered: Dict[int, List[int]] = {}
    for user, entries in grouped.items():
        entries.sort(key=lambda entry: _prediction_sort_key(entry[1], entry[2], entry[3]))
        seen: Set[int] = set()
        ordered_items: List[int] = []
        for item, _, _, _ in entries:
            if item in seen:
                continue
            ordered_items.append(item)
            seen.add(item)
        ordered[user] = ordered_items
    return ordered


def _prediction_sort_key(score: Optional[float], rank: Optional[int], order: int) -> Tuple[int, float, int]:
    if rank is not None:
        return (0, float(rank), order)
    if score is not None:
        return (1, -score, order)
    return (2, 0.0, order)


def ndcg_at_k(predicted: Sequence[int], truth: Set[int], k: int) -> float:
    """Compute NDCG@k for a single user."""

    top_predictions = predicted[:k]
    dcg = 0.0
    for idx, item in enumerate(top_predictions):
        if item in truth:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def evaluate(predictions_path: Path, ground_truth_path: Path, k: int = 20) -> Dict[str, object]:
    truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(predictions_path)

    scores: List[float] = []
    missing_users: List[int] = []

    for user, true_items in truth.items():
        predicted_items = predictions.get(user, [])
        if not predicted_items:
            missing_users.append(user)
        scores.append(ndcg_at_k(predicted_items, true_items, k))

    mean_ndcg = sum(scores) / len(scores) if scores else 0.0
    covered_users = len(truth) - len(missing_users)

    return {
        "k": k,
        "users": len(truth),
        "covered_users": covered_users,
        "missing_users": len(missing_users),
        "mean_ndcg": mean_ndcg,
        "predictions_path": str(predictions_path),
        "ground_truth_path": str(ground_truth_path),
    }


def _find_ground_truth(split: str) -> Path:
    candidates = [
        Path(f"{split}_split.parquet"),
        Path(f"{split}_split.csv"),
        Path(f"{split}_split.txt"),
        Path("interim_data") / f"{split}_split.parquet",
        Path("interim_data") / f"{split}_split.csv",
        Path("interim_data") / f"{split}_split.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate ground-truth split for '{split}'.")


def _format_report(results: Dict[str, object]) -> str:
    covered = results["covered_users"]
    total = results["users"]
    missing = results["missing_users"]
    coverage = (covered / total * 100) if total else 0.0
    lines = [
        "Evaluation report",
        "-----------------",
        f"Ground truth: {results['ground_truth_path']}",
        f"Predictions:  {results['predictions_path']}",
        f"Users evaluated: {total}",
        f"Users with predictions: {covered} ({coverage:.2f}%)",
        f"Users missing predictions: {missing}",
        f"Mean NDCG@{results['k']}: {results['mean_ndcg']:.6f}",
    ]
    return "\n".join(lines)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation predictions with NDCG@K.")
    parser.add_argument(
        "--preds",
        required=True,
        type=Path,
        help="Path to the predictions file (CSV, TXT, or Parquet) containing user_id, item_id, and optional score/rank columns.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Explicit path to the ground-truth file. If omitted, the script looks for <split>_split files.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Data split to evaluate against when --ground-truth is not provided.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Cutoff rank for NDCG calculation.",
    )
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(args)
    predictions_path = parsed.preds
    ground_truth_path: Path
    if parsed.ground_truth is not None:
        ground_truth_path = parsed.ground_truth
    else:
        ground_truth_path = _find_ground_truth(parsed.split)

    results = evaluate(predictions_path, ground_truth_path, parsed.k)
    print(_format_report(results))


if __name__ == "__main__":
    main()
