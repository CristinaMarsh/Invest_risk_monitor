from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TENDENCY_COLUMNS = [
    "sell_all_softmax",
    "reduce_on_rebound_softmax",
    "keep_softmax",
]
BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000000001]
BUCKET_LABELS = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]


def load_audit_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "observation_windows",
        "severe_drawdown_threshold_pct",
        "min_bucket_samples",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"Audit config missing keys: {', '.join(missing)}")
    data["observation_windows"] = sorted(
        {int(value) for value in data["observation_windows"]}
    )
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_audit_frame(snapshot_path: Path, outcome_path: Path) -> pd.DataFrame:
    snapshots = pd.json_normalize(read_jsonl(snapshot_path))
    outcomes = pd.json_normalize(read_jsonl(outcome_path))
    if snapshots.empty or outcomes.empty:
        return pd.DataFrame()

    keys = ["symbol", "prediction_timestamp_utc", "price_date"]
    for key in keys:
        if key not in snapshots.columns or key not in outcomes.columns:
            return pd.DataFrame()
    merged = snapshots.merge(outcomes, on=keys, suffixes=("", "_outcome"))
    if "prediction_date" in merged.columns:
        merged["prediction_year"] = pd.to_datetime(
            merged["prediction_date"], errors="coerce"
        ).dt.year
    return merged


def numeric_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def numeric_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).apply(
        pd.to_numeric, errors="coerce"
    )
    frame = frame.dropna()
    if len(frame) < 3:
        return None
    if frame["left"].std(ddof=0) == 0 or frame["right"].std(ddof=0) == 0:
        return None
    return float(frame["left"].corr(frame["right"]))


def bucket_statistics(
    frame: pd.DataFrame,
    scope_type: str,
    scope_value: str,
    tendency_column: str,
    windows: list[int],
    severe_drawdown_threshold: float,
    min_bucket_samples: int,
) -> list[dict[str, Any]]:
    if frame.empty or tendency_column not in frame.columns:
        return []

    working = frame.copy()
    working["bucket"] = pd.cut(
        pd.to_numeric(working[tendency_column], errors="coerce"),
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )
    rows = []
    for bucket in BUCKET_LABELS:
        group = working[working["bucket"] == bucket]
        row: dict[str, Any] = {
            "scope_type": scope_type,
            "scope_value": scope_value,
            "tendency": tendency_column,
            "bucket": bucket,
            "sample_count": int(len(group)),
            "sample_warning": len(group) < min_bucket_samples,
        }
        for window in windows:
            return_col = f"forward_return_{window}d"
            drawdown_col = f"forward_max_drawdown_{window}d"
            rebound_col = f"forward_max_rebound_{window}d"
            row[f"forward_return_{window}d_mean"] = numeric_mean(
                group.get(return_col, pd.Series(dtype=float))
            )
            row[f"forward_return_{window}d_median"] = numeric_median(
                group.get(return_col, pd.Series(dtype=float))
            )
            row[f"forward_max_drawdown_{window}d_mean"] = numeric_mean(
                group.get(drawdown_col, pd.Series(dtype=float))
            )
            row[f"forward_max_drawdown_{window}d_median"] = numeric_median(
                group.get(drawdown_col, pd.Series(dtype=float))
            )
            row[f"forward_max_rebound_{window}d_mean"] = numeric_mean(
                group.get(rebound_col, pd.Series(dtype=float))
            )
            row[f"forward_max_rebound_{window}d_median"] = numeric_median(
                group.get(rebound_col, pd.Series(dtype=float))
            )

        severe = pd.to_numeric(
            group.get("forward_max_drawdown_20d", pd.Series(dtype=float)),
            errors="coerce",
        ) <= severe_drawdown_threshold
        row["severe_drawdown_ratio_20d"] = (
            float(severe.mean()) if len(group) else None
        )
        first_rebound = group.get("first_clear_move_pct", pd.Series(dtype=str)) == "rebound"
        row["first_rebound_then_drawdown_ratio_20d"] = (
            float((first_rebound & severe).mean()) if len(group) else None
        )
        rows.append(row)
    return rows


def correlation_statistics(
    frame: pd.DataFrame,
    scope_type: str,
    scope_value: str,
    windows: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for tendency in TENDENCY_COLUMNS:
        if tendency not in frame.columns:
            continue
        for window in windows:
            for metric_name, column in [
                ("future_return", f"forward_return_{window}d"),
                ("future_max_drawdown", f"forward_max_drawdown_{window}d"),
            ]:
                if column not in frame.columns:
                    continue
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "tendency": tendency,
                        "target": column,
                        "metric": metric_name,
                        "sample_count": int(
                            frame[[tendency, column]].apply(
                                pd.to_numeric, errors="coerce"
                            ).dropna().shape[0]
                        ),
                        "pearson_corr": safe_corr(frame[tendency], frame[column]),
                    }
                )
    return rows


def build_reports(
    frame: pd.DataFrame,
    audit_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = [int(value) for value in audit_config["observation_windows"]]
    severe_threshold = float(audit_config["severe_drawdown_threshold_pct"])
    min_samples = int(audit_config["min_bucket_samples"])

    scopes: list[tuple[str, str, pd.DataFrame]] = [("all_assets", "ALL", frame)]
    for symbol, group in frame.groupby("symbol", dropna=False):
        scopes.append(("symbol", str(symbol), group))
    if "prediction_year" in frame.columns:
        for year, group in frame.groupby("prediction_year", dropna=True):
            scopes.append(("year", str(int(year)), group))

    bucket_rows = []
    corr_rows = []
    for scope_type, scope_value, group in scopes:
        for tendency in TENDENCY_COLUMNS:
            bucket_rows.extend(
                bucket_statistics(
                    group,
                    scope_type,
                    scope_value,
                    tendency,
                    windows,
                    severe_threshold,
                    min_samples,
                )
            )
        corr_rows.extend(correlation_statistics(group, scope_type, scope_value, windows))

    return pd.DataFrame(bucket_rows), pd.DataFrame(corr_rows)


def print_report(title: str, frame: pd.DataFrame, max_rows: int = 80) -> None:
    print(f"\n## {title}")
    if frame.empty:
        print("No data.")
        return
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        print(f"... truncated {len(frame) - max_rows} rows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate uncalibrated model tendencies against realized paths."
    )
    parser.add_argument("--history-dir", default="history")
    parser.add_argument("--config", default="audit_config.json")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    snapshot_path = history_dir / "prediction_snapshots.jsonl"
    outcome_path = history_dir / "prediction_outcomes.jsonl"
    audit_config = load_audit_config(Path(args.config))
    frame = load_audit_frame(snapshot_path, outcome_path)

    if frame.empty:
        print("No evaluated predictions yet.")
        print(f"Snapshots: {snapshot_path}")
        print(f"Outcomes: {outcome_path}")
        print("Run the monitor over time until at least the largest forward window matures.")
        return 0

    bucket_report, corr_report = build_reports(frame, audit_config)
    low_sample = bucket_report[
        bucket_report.get("sample_warning", pd.Series(dtype=bool)) == True
    ]

    print_report("Bucket Statistics", bucket_report)
    print_report("Correlations", corr_report)
    if not low_sample.empty:
        print("\n## Sample Size Warning")
        print(
            "Some buckets are below min_bucket_samples="
            f"{audit_config['min_bucket_samples']}; treat those rows as exploratory only."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bucket_report.to_csv(output_dir / "bucket_statistics.csv", index=False)
        corr_report.to_csv(output_dir / "correlations.csv", index=False)
        print(f"\nSaved CSV reports to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
