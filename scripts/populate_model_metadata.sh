#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/Models}"

declare -A METADATA_DIR=(
  [Model_1]=metadata/model1
  [Model_2]=metadata/model2
)

declare -A PCA_FILES=(
  [Model_1]=model1_node_pca.joblib
  [Model_2]=model2_node_pca.joblib
)

declare -A PROCESSED_STATS=(
  [Model_1]=processed/csv_features_stats.yaml
  [Model_2]=processed/csv_features_stats.yaml
)

declare -A EVENTS_SUMMARIES=(
  [Model_1]=train/events.csv
  [Model_2]=train/events.csv
)

declare -A TEST_SUMMARIES=(
  [Model_1]=test/events.csv
  [Model_2]=test/events.csv
)

for MODEL in Model_1 Model_2; do
  SRC_META="${METADATA_DIR[$MODEL]}"
  DST_MODEL="$TARGET_DIR/${MODEL}"

  if [[ ! -d "$DST_MODEL" ]]; then
    echo "Create the raw ${MODEL} data under $DST_MODEL before running this script."
    exit 1
  fi

  mkdir -p "$DST_MODEL/processed" "$DST_MODEL/train/events_split_seed42"
  cp -f "$ROOT_DIR/$SRC_META/${PROCESSED_STATS[$MODEL]}" "$DST_MODEL/processed/csv_features_stats.yaml"
  cp -f "$ROOT_DIR/$SRC_META/${EVENTS_SUMMARIES[$MODEL]}" "$DST_MODEL/train/events.csv"
  cp -f "$ROOT_DIR/$SRC_META/${TEST_SUMMARIES[$MODEL]}" "$DST_MODEL/test/events.csv"

  if [[ -n "${PCA_FILES[$MODEL]:-}" ]]; then
  cp -f "$ROOT_DIR/$SRC_META/${PCA_FILES[$MODEL]}" "$DST_MODEL/"
  fi

    cp -f "$ROOT_DIR/$SRC_META/train/events_split_seed42/train_split.csv" "$DST_MODEL/train/events_split_seed42/"
    cp -f "$ROOT_DIR/$SRC_META/train/events_split_seed42/val_split.csv" "$DST_MODEL/train/events_split_seed42/"
  HARD_SRC="$SRC_META/train/events_hardholdout_seed42"
  if [[ -d "$HARD_SRC" ]]; then
    mkdir -p "$DST_MODEL/train/events_hardholdout_seed42"
    for FILE in train_split.csv val_split.csv val_event_ids.txt hardness_summary.csv; do
      if [[ -f "$HARD_SRC/$FILE" ]]; then
        cp -f "$HARD_SRC/$FILE" "$DST_MODEL/train/events_hardholdout_seed42/"
      fi
    done
  fi
done

echo "Model metadata populated under $TARGET_DIR."
