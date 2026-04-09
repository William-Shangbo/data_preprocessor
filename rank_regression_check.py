import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(WORKSPACE_ROOT / "willden" / "data_method"))

from data_processing import rank  # noqa: E402
from ewma import causal_ewma  # noqa: E402

FEATURE_COLS = [f"f{i}" for i in range(384)]
KEY_COLS = ["stockid", "dateid", "timeid"]


def _build_imputed_source(source_df: pd.DataFrame, order_cols: list[str], half_life: int) -> pd.DataFrame:
    ordered = source_df.sort_values(["stockid"] + order_cols).reset_index(drop=True)
    ewma_values = causal_ewma(
        df=ordered,
        value_cols=FEATURE_COLS,
        group_cols=["stockid"],
        order_cols=order_cols,
        half_life=half_life,
        min_count=1,
    )
    original = ordered[FEATURE_COLS]
    imputed = original.where(original.notna(), ewma_values)
    return pd.concat([ordered[KEY_COLS].reset_index(drop=True), imputed.reset_index(drop=True)], axis=1)


def assert_rank_edge_cases() -> None:
    df = pd.DataFrame(
        {
            "dateid": [0, 0, 0, 0],
            "timeid": [0, 0, 0, 0],
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "singleton": [np.nan, 3.0, np.nan, np.nan],
            "ordered": [1.0, 2.0, 3.0, np.nan],
            "ties": [5.0, 5.0, 7.0, np.nan],
        }
    )
    out = rank(df, ["all_nan", "singleton", "ordered", "ties"], by=["dateid", "timeid"])
    assert out["all_nan_r"].isna().all()
    assert out.loc[1, "singleton_r"] == 0.5
    assert np.isclose(out.loc[0, "ordered_r"], 0.0)
    assert np.isclose(out.loc[1, "ordered_r"], 0.5)
    assert np.isclose(out.loc[2, "ordered_r"], 1.0)
    assert np.isclose(out.loc[0, "ties_r"], 0.25)
    assert np.isclose(out.loc[1, "ties_r"], 0.25)
    assert np.isclose(out.loc[2, "ties_r"], 1.0)


def build_rank_input(dateid: int) -> pd.DataFrame:
    raw = pd.read_parquet(WORKSPACE_ROOT / "ljcomp" / "daily_data_raw" / f"d{dateid}.parquet")[KEY_COLS + FEATURE_COLS]
    imputed = _build_imputed_source(raw, ["timeid"], 29)
    ewma = causal_ewma(imputed, FEATURE_COLS, ["stockid"], ["timeid"], half_life=30, min_count=1)
    current = pd.concat([imputed[KEY_COLS].reset_index(drop=True), ewma.reset_index(drop=True)], axis=1)
    return current[current["dateid"] == dateid].copy().sort_values(["dateid", "timeid", "stockid"]).reset_index(drop=True)


def check_day33_non_all_nan_stability() -> None:
    day33 = build_rank_input(33)
    start = time.time()
    out = rank(day33, FEATURE_COLS, by=["dateid", "timeid"])
    print(f"d33 rank elapsed={time.time() - start:.3f}s")

    non_all_nan_cols = []
    for col in FEATURE_COLS:
        has_signal = day33.groupby(["dateid", "timeid"], dropna=False)[col].count().gt(0).all()
        if has_signal:
            non_all_nan_cols.append(col)
    if not non_all_nan_cols:
        raise AssertionError("Expected at least one non-all-NaN column")
    subset = out[[f"{col}_r" for col in non_all_nan_cols]]
    if subset.isna().all().any():
        raise AssertionError("Non-all-NaN columns unexpectedly collapsed to all NaN")


if __name__ == "__main__":
    assert_rank_edge_cases()
    check_day33_non_all_nan_stability()
    print("rank regression checks passed")
