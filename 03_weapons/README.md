# Weapon Screen Recorder

This project detects and records the six weapon icons in the first row of the *Vampire Survivors* HUD. It uses template similarity, runner-up margins, occupancy checks, and temporal stabilization to avoid treating transient enemies or effects behind translucent slots as weapon changes.

## Outputs

- `weapon_timeline.csv`: one row per sampled video time, with stabilized weapon identities and confidence information.
- `weapon_events.csv`: initial, acquired, changed, and removed slot events.
- Optional annotated debug frames for visual quality control.
- Raw and stabilized predictions, including `needs_review` flags for uncertain cases.

## Main entry points

- [`notebooks/weapon_screen_recorder_explained.ipynb`](notebooks/weapon_screen_recorder_explained.ipynb): explained workflow and short-run validation.
- [`scripts/weapon_screen_recorder.py`](scripts/weapon_screen_recorder.py): reusable command-line detector.
- [`data/wiki_weapon_manifest.csv`](data/wiki_weapon_manifest.csv): reproducible mapping between weapon names, local reference filenames, and original image URLs.

## Command-line use

```bash
python scripts/weapon_screen_recorder.py \
  --video /path/to/gameplay.mp4 \
  --output-dir results/local_run \
  --icon-dir templates/wiki_icons \
  --manifest data/wiki_weapon_manifest.csv
```

Use `python scripts/weapon_screen_recorder.py --help` for sampling, debug, and manual-layout options. Gameplay recordings and generated result directories remain local and are not versioned.

Author: **Tahereh Fahi**
