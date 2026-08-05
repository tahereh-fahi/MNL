<div align="center">

# MNL: Gameplay Reward Analytics

**Computer-vision pipelines that transform gameplay recordings into structured, time-aligned research data**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-5C3EE8?logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-reproducible%20notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20research-2E7D32)

</div>

## Overview

MNL is a research-oriented toolkit for extracting reward-related events from *Vampire Survivors* gameplay. The project combines optical character recognition, template matching, temporal tracking, interpretable classification, and interval-coded reward-trajectory modeling to convert video into analysis-ready tables.

The repository is intentionally curated: it contains active methods, explanatory notebooks, reusable scripts, selected validation summaries, and compact reference files. Source recordings, human-coded workbooks, internal lab material, and bulk-generated outputs are not distributed.

## Project portfolio

| Project | Research task | Approach | Primary outputs |
|---|---|---|---|
| [Kill counter and timestamp](<01_kill_counter_and_time_stamp/>) | Recover run time and cumulative enemy kills | HUD localization, image preprocessing, OCR, and temporal consistency checks | Time-aligned CSVs and RMSE evaluation |
| [XP and trajectory gem classification](<02_blue_gems/>) | Detect collected experience gems and estimate color | XP-bar change detection, player anchoring, multi-frame trajectories, template evidence, and a weak-visual classifier | Event, per-second, five-second, and validation tables |
| [Weapon screen recorder](<03_weapons/>) | Track the six weapon slots in the top-left HUD | Template matching, confidence margins, and temporal stabilization | Weapon timeline, acquisition/change events, and review flags |
| [Reward trajectory 01](<04_reward_trajectory_01/>) | Model interval-coded reward trajectories with Hawkes processes | Event-level temporal point-process modeling, self-excitation analysis, and burstiness checks | Thesis companion notebook and write-up |

## Selected result

The XP-and-trajectory model was evaluated on 1,254 visually referenced pickup events. Its out-of-fold performance was **96.89% accuracy**, **86.65% balanced accuracy**, and **0.861 macro F1**. These results should be interpreted with the class distribution in mind: the reference set contains 971 blue, 280 green, and only 3 red events, so red-gem performance is not yet estimated reliably.

![Out-of-fold confusion matrix](<02_blue_gems/XP and Trajectory Gem Classification Model/model_outputs/xp_weak_visual_confusion_matrix.png>)

The complete explanatory notebook and compact result summaries are available in the [XP and Trajectory Gem Classification Model](<02_blue_gems/XP and Trajectory Gem Classification Model/>).

## Repository layout

```text
MNL/
├── 01_kill_counter_and_time_stamp/   # OCR-based HUD extraction and evaluation
├── 02_blue_gems/                     # XP-event and collected-gem modeling
├── 03_weapons/                       # Weapon-slot detection and stabilization
├── 04_reward_trajectory_01/          # Hawkes-process reward-trajectory thesis materials
├── CITATION.cff                      # Citation metadata
├── requirements.txt                  # Core Python dependencies
└── README.md                         # Project overview
```

Each project directory contains an entry-point README or an explanatory notebook. Generated `results/` directories are ignored so experiments can be run locally without expanding the Git history.

## Getting started

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Open the notebooks

```bash
jupyter lab
```

Start with one of these notebooks:

- [HUD kill-counter extractor](<01_kill_counter_and_time_stamp/notebooks/vampire_survivors_kill_counter_extractor.ipynb>)
- [Kill-counter RMSE evaluation](<01_kill_counter_and_time_stamp/notebooks/rmse_kill_counter_accuracy.ipynb>)
- [XP and trajectory gem classifier](<02_blue_gems/XP and Trajectory Gem Classification Model/xp_trajectory_gem_classification.ipynb>)
- [Weapon screen recorder](<03_weapons/notebooks/weapon_screen_recorder_explained.ipynb>)
- [Reward trajectory Hawkes analysis](<04_reward_trajectory_01/CRAVE_HawkesProcess_Analysis.ipynb>)

### 3. Supply a local recording

Gameplay videos are not included. Place recordings in the ignored `00-Videos/` directory or update the relevant `VIDEO_PATH` in a notebook. Outputs are written beneath each project's ignored `results/` directory.

The reusable detectors also provide command-line documentation:

```bash
python 02_blue_gems/scripts/detect_collected_gems_from_xp_ab.py --help
python 03_weapons/scripts/weapon_screen_recorder.py --help
```

## Research and reproducibility notes

- Human-coded annotations are used only for evaluation where stated; the active gem detector does not use the human-coded workbook as a model input.
- Confidence values, unresolved cases, and review flags are retained rather than silently forcing uncertain predictions.
- Raw videos, private research records, meeting transcripts, personal documents, and high-volume debug artifacts are excluded from version control.
- The included results describe the current evaluation data and should not be generalized beyond the represented gameplay conditions without additional validation.

## Author

**Tahereh Fahi**

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). If this work supports a publication or derivative research project, please cite the repository version used in the analysis.
