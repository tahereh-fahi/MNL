# XP and Trajectory Gem Classification

This project detects collected blue, green, and red experience gems from gameplay video. A persistent XP-bar increase confirms a pickup event; color and quantity evidence are then estimated from level-normalized XP change, template detections, gem trajectories, disappearance evidence, and the timing quality of each event.

The workflow explicitly separates three questions:

1. Did a pickup occur?
2. Which gem color is most consistent with the event?
3. How many gems may have contributed to the same XP increase?

Uncertain color or quantity estimates remain unresolved rather than being silently forced.

## Active components

- [`XP and Trajectory Gem Classification Model/`](<XP and Trajectory Gem Classification Model/>): explanatory notebook and compact model outputs.
- [`scripts/detect_collected_gems_from_xp_ab.py`](scripts/detect_collected_gems_from_xp_ab.py): reusable XP-jump and trajectory detector.
- [`scripts/compare_unresolved_color_methods.py`](scripts/compare_unresolved_color_methods.py): evaluation of alternative color-classification methods.
- [`scripts/render_event_trajectory_audit.py`](scripts/render_event_trajectory_audit.py): visual audit renderer for selected events.
- `templates/`: blue, green, and red gem references used by the detector.

## Validation snapshot

The current weak-visual classifier was evaluated on 1,254 visually referenced events and achieved **96.89% out-of-fold accuracy**, **86.65% balanced accuracy**, and **0.861 macro F1**. The reference set is highly imbalanced—971 blue, 280 green, and 3 red events—so red-gem performance requires substantially more validation data.

![XP and weak-visual confusion matrix](<XP and Trajectory Gem Classification Model/model_outputs/xp_weak_visual_confusion_matrix.png>)

## Data boundary

The active detector is label-free: it does not use the human-coded Gaming Content workbook as a model input. Human annotations are used only in the documented evaluation stage. Raw videos, source annotations, bulk event previews, and internal archives are intentionally excluded from the public repository.

Author: **Tahereh Fahi**
