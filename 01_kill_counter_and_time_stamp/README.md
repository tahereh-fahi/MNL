# Kill Counter and Timestamp Extraction

This project converts the *Vampire Survivors* HUD timer and cumulative enemy-kill counter into structured time-series data. It combines region-of-interest detection, contrast-aware preprocessing, EasyOCR, and temporal consistency checks.

## Contents

- [`vampire_survivors_kill_counter_extractor.ipynb`](notebooks/vampire_survivors_kill_counter_extractor.ipynb): explained end-to-end extraction workflow.
- [`rmse_kill_counter_accuracy.ipynb`](notebooks/rmse_kill_counter_accuracy.ipynb): comparison of extracted values with human-coded intervals.
- `templates/`: compact HUD anchors and digit references used during localization and quality control.

## Validation snapshot

For the documented `video4_Imelda_100` run, 60 human-coded intervals were matched to OCR-derived values, with an RMSE of **2.9749 kills**. The notebook preserves the comparison logic and diagnostic output; the private human-coded workbook is not distributed.

## Local use

1. Install the root [`requirements.txt`](../requirements.txt).
2. Open the extractor notebook and set `VIDEO_PATH` to a local gameplay recording.
3. Run the notebook to create full-resolution and five-second-interval CSVs under the ignored `results/<video_name>/` directory.
4. Use the RMSE notebook when an authorized reference sheet is available.

Author: **Tahereh Fahi**
