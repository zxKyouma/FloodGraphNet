#!/usr/bin/env python3
import argparse
import os

import pandas as pd


KEY_COLS = ["event_id", "node_id", "node_type", "t_rel"]


def _load_pred(path: str, label: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = KEY_COLS + ["y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    dup = int(df.duplicated(KEY_COLS).sum())
    if dup:
        raise ValueError(f"{label} has {dup} duplicate prediction keys.")
    return df[required].copy()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge Model_1 and Model_2 XGBoost prediction dumps into a Kaggle submission parquet."
    )
    p.add_argument("--sample", required=True, help="Path to sample_submission.parquet")
    p.add_argument("--model1", required=True, help="Path to Model_1 prediction parquet from --dump_test_predictions")
    p.add_argument("--model2", required=True, help="Path to Model_2 prediction parquet from --dump_test_predictions")
    p.add_argument("--output", required=True, help="Path to output submission parquet")
    args = p.parse_args()

    sample = pd.read_parquet(args.sample)
    required_sample = ["model_id", "event_id", "node_type", "node_id", "water_level"]
    missing_sample = [c for c in required_sample if c not in sample.columns]
    if missing_sample:
        raise ValueError(f"Sample submission is missing required columns: {missing_sample}")

    sort_col = None
    for cand in ("t_rel", "row_id"):
        if cand in sample.columns:
            sort_col = cand
            break
    if sort_col is None:
        raise ValueError("Sample submission must contain either `t_rel` or `row_id`.")
    if sort_col != "t_rel":
        raise ValueError(
            "Sample submission is missing `t_rel`. The XGBoost merge script requires `t_rel` to align predictions."
        )

    m1 = _load_pred(args.model1, "Model_1 predictions")
    m2 = _load_pred(args.model2, "Model_2 predictions")

    sample = sample.copy()
    sample["water_level"] = pd.Series([float("nan")] * len(sample), index=sample.index, dtype="float32")

    mask1 = sample["model_id"] == 1
    mask2 = sample["model_id"] == 2

    left1 = sample.loc[mask1, KEY_COLS].reset_index().rename(columns={"index": "_row"})
    left2 = sample.loc[mask2, KEY_COLS].reset_index().rename(columns={"index": "_row"})

    merged1 = left1.merge(m1, on=KEY_COLS, how="left", validate="one_to_one")
    merged2 = left2.merge(m2, on=KEY_COLS, how="left", validate="one_to_one")

    miss1 = int(merged1["y_pred"].isna().sum())
    miss2 = int(merged2["y_pred"].isna().sum())
    if miss1 or miss2:
        raise ValueError(
            f"Missing merged predictions: model1_missing={miss1}, model2_missing={miss2}. "
            "Prediction dumps do not fully cover sample_submission keys."
        )

    sample.loc[merged1["_row"].to_numpy(), "water_level"] = merged1["y_pred"].to_numpy(dtype="float32")
    sample.loc[merged2["_row"].to_numpy(), "water_level"] = merged2["y_pred"].to_numpy(dtype="float32")

    if sample["water_level"].isna().any():
        raise ValueError("Final submission still contains NaN water_level values after merge.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sample.to_parquet(args.output, index=False)
    print(f"[SUBMISSION_WRITTEN] path={args.output} rows={len(sample)}", flush=True)


if __name__ == "__main__":
    main()
