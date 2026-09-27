# VSS Framework

**Author and maintainer:** Tahereh Fahi  
**Last updated:** September 26, 2026

Evidence-based reconstruction of Vampire Survivors events from gameplay video.

The personal GitHub copy is a **source-code snapshot**, not a runnable release:
source videos, generated runs and inputs, OCR weights, and game artwork/reference
assets are not published there. The asset redistribution rights are not yet
established; see `delivery/THIRD_PARTY_NOTICES.md`. A complete local research
workspace with authorized assets is required for video processing. Passing a
run's technical audit does not establish detector accuracy or publication
readiness.

Video 4 is the reference dataset used to establish detector behavior, but the
framework contracts and dashboard projections are source-independent and are
used across every currently published video. Production detector implementations
and their required assets are owned by this framework. Human-coded data is
not used for training, testing, or ground truth. Five-second tables are derived
from versioned canonical events and are not the primary data source.

## Current release

Code version `0.18.0` integrates gem, inventory, weapon, and HUD detectors plus
observation normalization into this package. Their detection algorithms are
preserved; input handling also accepts confirmed zero-event gem outputs and
bounds inventory initialization to the requested interval. Historical
artifact-import commands remain available explicitly.

For a fresh validation run from the complete local MNL workspace root, after
providing the authorized video, assets, OCR models, and a working Python
environment outside OneDrive:

```bash
PYTHONPATH='08-VSS frame work/src' /path/to/venv/bin/python -m vss_framework.cli run-video \
  --config '08-VSS frame work/configs/video4.json' \
  --output '09-VSS-validation-results/video4_fresh_validation_01'
```

Add `--max-seconds 20` for a smoke run. The runner generates initial XP results,
weapon observations, Level-Up inventory changes, chest intervals and reward
identities, reconciled inventory progression, and final gem results in order.
Chest identities use agreement across multiple reveal frames; accepted automatic
chest rewards replace lower-confidence HUD replacement guesses. Cached
stages are reused only after code, settings, input, and output integrity checks.
Partial runs and failed quality checks are not publication-ready. Multiple-game
recordings require separate run configurations. OCR model files must be available
locally; install the framework's `ocr` extra to obtain its software dependencies.
This code migration alone does not certify that historical datasets were rerun.

The framework also provides:

- an Event Catalog covering the current coding definitions;
- source-independent canonical event contracts;
- SHA-256 verification for the Video 4 asset and every reused artifact;
- adapters for the existing HUD/Gem candidate release;
- an adapter for the existing inventory and Level-Up output;
- an adapter for reviewed late-game Big Coin Bag and Floor Chicken outcomes;
- deterministic canonical JSONL and a derived five-second CSV;
- preservation of review flags, unresolved gem identity, and estimated-quantity
  semantics.
- a direct OpenCV reader with explicit frame and media-time clocks;
- a common frame-detector contract and frame-level observation schema;
- a screen-state wrapper that reuses the existing Level-Up menu geometry and
  weapon-HUD locator rather than copying them.
- direct sampled XP-bar observations with raw measurement, quality, HUD score,
  Level-Up overlay score, and an explicit accepted value;
- automated Kill Counter and game-clock signals that distinguish direct OCR
  observations from carried or assumed state;
- automated XP-linked gem pickup signals with color and estimated quantity;
- a separate `signal_observations.jsonl` so interval signals are not confused
  with exact sampled frames.
- a batch wrapper that executes the complete existing XP/gem detector without
  changing its algorithm, then normalizes its outputs into framework signals.
- a batch wrapper that executes the existing Kill Counter/Game Clock OCR worker
  unchanged, preserving direct observations, carried Kill state, missing clocks,
  worker QC, and evidence artifacts.
- a framework-controlled runner for the existing automated Level-Up/inventory
  recorder that emits canonical selection and inventory-transition events.
- Level-Up transaction resolution that separates persistent inventory changes
  from late-game instant rewards and retains unresolved action identity when
  direct visual evidence is insufficient;
- video-only classification of the two-choice Big Coin Bag/Floor Chicken menu,
  including suppression of false passive-item upgrades.
- direct Coin Counter observations with monotonic OCR rejection and no
  forward-filling;
- direct player health-bar red-fill width observations, retained in
  1440p-normalized pixels until an HP calibration is established.
- player Game Level reconstruction from XP/HUD evidence, kept explicitly
  separate from weapon and passive-item levels;
- same-level XP bridges that preserve lower reliability instead of drawing a
  false continuous measurement;
- reusable Kill Counter repairs for truncated OCR and persistent leading-place
  shifts, without video-specific timestamp constants;
- reusable inventory, gem, five-second reward-trajectory, and exact-event
  dashboard projections;
- validated cumulative Gold observations plus framework-generated Gold delta
  five-second bars and the centered 30-second mean of the cumulative counter;
- canonical dashboard releases that include exact gem events and inventory
  selections, plus time-safe merging of separately recorded video parts.

See `MIGRATION_STATUS.md` for the dashboard-to-framework migration ledger and
the boundary between analytical computation and presentation-only code.

The direct scan command is the first controlled exception: it decodes only the
requested source-video interval so detector wrappers can be checked against
real frames before a full run.

## Reused components

| Capability | Existing source |
|---|---|
| Game clock and Kill Counter observations | `01_kill_counter_and_time_stamp` through the sealed `06-Pipeline` adapter output |
| XP-linked gem candidates | `02_gems` through the sealed `06-Pipeline` candidate release |
| Level-Up option and inventory changes | `03_weapons/scripts/inventory_event_recorder.py` output |
| Late reward outcomes | Reviewed Video 4 evidence already published to the dashboard |

The framework records artifact paths and hashes. It does not import Human-coded
spreadsheets or treat comparison reports as detector inputs.

## Run

From the MNL workspace root:

```bash
PYTHONPATH='VSS frame work/src' \
python3 -m vss_framework.cli check-catalog
```

Build the cross-video Gold publication file, including counter bins, Gold
change bars, and cumulative-counter mean:

```bash
PYTHONPATH='VSS frame work/src' \
python3 -m vss_framework.cli build-gold-series \
  --runs-root 'VSS frame work/runs' \
  --config 'VSS frame work/configs/dashboard_gold_series.json' \
  --output '07-Dashboard/public/data/gold-coin-series.json'
```

Build the first Video 4 run:

```bash
PYTHONPATH='VSS frame work/src' \
python3 -m vss_framework.cli bootstrap-video4
```

Scan a half-open interval directly from Video 4 with the workspace OpenCV
environment:

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli scan-video4 \
  --start-second 8 --end-second 11 --sample-fps 10
```

The scan writes `frame_observations.jsonl`, `signal_observations.jsonl`, and
`scan_manifest.json`. Video SHA-256 verification is on by default.
`--skip-video-hash` is available for a quick local iteration, and the manifest
explicitly records that the asset was not verified for that run.

`frame_observations.jsonl` contains screen-state and XP-bar measurements at the
requested sample rate. XP values are accepted only when bar quality and the
gameplay HUD pass their thresholds and no strong Level-Up overlay is present.
The raw measurement is retained when rejected, but the accepted value remains
null.

`signal_observations.jsonl` contains the sealed automated Kill Counter,
game-clock, and XP-linked gem signals. A carried Kill state retains its numeric
state but has `observed: false`; it must not be interpreted as a new OCR
observation. Human-coded sources are excluded.

## Run the complete existing XP/gem detector

For a short framework-controlled smoke run:

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-gem-xp \
  --max-seconds 20 \
  --output 'VSS frame work/runs/video4_gem_xp_smoke_20'
```

For the entire Video 4, omit `--max-seconds` and use a new empty output
directory:

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-gem-xp \
  --output 'VSS frame work/runs/video4_gem_xp_full'
```

This command calls `02_gems/scripts/detect_collected_gems_from_xp_ab.py`
directly. It does not call the candidate-comparison job and does not load Human
validation files. Video, detector, inventory, and template hashes are recorded
in `run_manifest.json`.

## Run the automated Level-Up/inventory recorder

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-inventory \
  --max-seconds 60 \
  --output 'VSS frame work/runs/video4_inventory_smoke_60'
```

Omit `--max-seconds` for the complete video. The runner uses automated XP
events, menu observations, icon manifests, and the automated weapon timeline.
It does not load Human coding or the reviewed treasure-chest audit. A visible
Banish/Reroll/Skip control is not treated as evidence that it was used. Banish
is accepted only for the supported single-option, already-maxed inventory case;
other action-button cases remain unresolved until a direct click-state detector
is available.

## Track visible world pickups

The world-pickup stage searches pinned sprite references, joins repeated
detections into tracks, and emits only conservative review candidates when a
persistent track disappears near the player. A visible sprite alone is never
published as a pickup, and no Human-coded labels are used.

```bash
PYTHONPATH='VSS frame work/src' '.venv/bin/python' -m vss_framework.cli \
  run-video4-world-pickups \
  --output 'VSS frame work/runs/video4_world_pickups'
```

For a fast calibration run, add `--start-second 0 --end-second 60`. Outputs
retain frame-level detections separately from bounded pickup candidates.

Before a publication scan, self-calibrate Video 4 without Human labels:

```bash
PYTHONPATH='VSS frame work/src' '.venv/bin/python' -m vss_framework.cli \
  calibrate-video4-world-pickups
```

The calibration compares alpha-composited references with untouched Video 4
hard negatives. Non-separable type/scale variants are disabled rather than
forced through one global threshold. The scan command loads this calibration
by default.

## Detect freeze-like effect intervals

```bash
PYTHONPATH='VSS frame work/src' '.venv/bin/python' -m vss_framework.cli \
  run-video4-freeze-effect \
  --sample-fps 4 \
  --output 'VSS frame work/runs/video4_freeze_effect'
```

This stage compensates global camera translation, measures residual scene
motion and pale/cyan visual state, and requires at least 1.5 seconds of
continuous freeze-like evidence. Level-Up menus break the series and gaps are
not filled. The resulting interval remains a review candidate; without a
corroborating Orologion sprite it is not assigned Orologion identity.

## Detect Vacuum-like gem flow

```bash
PYTHONPATH='VSS frame work/src' '.venv/bin/python' -m vss_framework.cli \
  run-video4-vacuum-flow \
  --sample-fps 5 \
  --output 'VSS frame work/runs/video4_vacuum_flow'
```

This stage reuses the established template-based gem detector and dynamic
player anchor. A candidate requires mass radial inward motion for at least
three consecutive samples and at least 25% depletion of detected on-screen
gems. General HSV-colored components are not accepted as gems. The effect
remains unassigned to Vacuum until direct pickup evidence corroborates it.

## Detect Gold Fever intervals

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-gold-fever \
  --sample-fps 4 \
  --output 'VSS frame work/runs/video4_gold_fever'
```

The detector follows the persistent gold gauge fixed to the bottom HUD edge.
It requires a wide, multi-row gold structure for at least one second, so level
art, coins, and short gold flashes are not sufficient. The emitted interval
confirms the visible Gold Fever state only; it does not claim that a Gilded
Clover caused the state unless separate pickup evidence is available.

## Detect Treasure Chest lifecycle and tier

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-chests \
  --sample-fps 4 \
  --output 'VSS frame work/runs/video4_chests'
```

This stage detects the persistent chest panel and central animation beam, then
requires a visible reveal of exactly one, three, or five reward orbs before it
publishes an interval as Tier 1, 2, or 3. Panel-like animation without a valid
reward reveal is retained only in the frame signals and is not published as a
chest. Reward identities are deliberately left for the next stage.

Identify the revealed rewards and publish their inventory transitions:

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli identify-video4-chest-rewards \
  --output 'VSS frame work/runs/video4_chest_rewards'
```

Reward-orb sprites are matched only against the inventory that had already
been established by Automated level-up observations. Max-level base weapons
are excluded from ordinary upgrades. Evolution candidates additionally
require their supporting passive item; Vandalier requires both birds. No
reviewed chest audit or Human-coded record is loaded.

## Detect Reroll, Skip, and Banish actions

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-menu-actions \
  --output 'VSS frame work/runs/video4_menu_actions'
```

The detector reads action counters directly from successive Level-Up menus and
emits a bounded transaction only when a readable counter decreases. Merely
showing a Reroll, Skip, or Banish button is not treated as use. Disabled and
unreadable values remain null rather than becoming zero.

## Scan Coin Counter and player health-bar fill

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-telemetry \
  --start-second 295 --end-second 305 --sample-fps 1 \
  --output 'VSS frame work/runs/video4_telemetry_295_305'
```

Coin observations are direct OCR values. A decreasing candidate is rejected
as missing and is never carried forward. Health-bar output is the directly
observed red-fill width normalized to a 1440p frame.

## Self-calibrate player health percentage

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli calibrate-video4-health \
  --sample-fps 2 \
  --output 'VSS frame work/runs/video4_health_calibration'
```

This samples the complete pinned Video 4 and infers the full-bar reference from
the highest repeatedly observed width cluster. It does not use a lone maximum,
Human coding, or an assumed HP value. Each direct width is then expressed as a
percentage of that reference. Missing detections remain missing rather than
becoming zero. A candidate below 40 normalized pixels must appear in adjacent
samples; isolated short red components are rejected instead of being mistaken
for critically low HP.

## Run the existing Kill Counter/Game Clock worker

For a short framework-controlled smoke run:

```bash
PYTHONPYCACHEPREFIX='VSS frame work/.cache/pycache' \
PYTHONPATH='VSS frame work/src' \
'.venv/bin/python' -m vss_framework.cli run-video4-hud \
  --max-seconds 20 \
  --output 'VSS frame work/runs/video4_hud_smoke_20'
```

For the entire Video 4, omit `--max-seconds` and select a new empty output
directory. The command runs
`01_kill_counter_and_time_stamp/scripts/extract_hud_worker.py` directly. It
does not invoke the comparison/validation job and does not load Human-coded
Kill or clock values. Direct Kill OCR, carried/assumed Kill state, rejected or
missing clock rows, and QC results remain distinguishable in the outputs.

Default outputs are written under `runs/video4_bootstrap/`:

```text
run_manifest.json
canonical_events.jsonl
derived_5s_windows.csv
```

## Evidence policy

Evidence grades are not accuracy probabilities:

- `A`: multiple strong and consistent automated signals;
- `B`: supported inference or a strong but incomplete source;
- `C`: candidate that must retain a review flag;
- `unresolved`: identity or occurrence cannot be responsibly assigned.

`0` is valid only when the relevant source was observable. Missing, occluded,
unprocessed, or unsupported evidence remains null or unresolved.

## Remaining detector work

1. Add a second visual confirmation for low-margin Treasure Chest reward sprite
   matches.
2. Strengthen direct pickup corroboration for world pickups, Freeze, Vacuum,
   and Gold Fever candidates.
3. Resolve missing intermediate Game Levels only when later HUD/XP evidence can
   establish the skipped transitions; until then, jumps remain auditable in
   `unresolvedLevelChanges` and are not invented for the chart.
4. Replace remaining command names containing `video4` with dataset-oriented
   aliases while retaining backward-compatible commands.

Each new detector must emit observations or candidates. It must not write
directly into the five-second publication table.
