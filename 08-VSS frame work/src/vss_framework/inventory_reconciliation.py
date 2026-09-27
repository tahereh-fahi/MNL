"""Merge automatically detected chest rewards into inventory progression."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .detectors.inventory import EVENT_COLUMNS, flag_out_of_range_slots, rebuild_inventory_progression
from .hashing import sha256_file
from .io import write_json

VERSION = "0.3.0"


def _latest_value(rows: pd.DataFrame, second: float, item: str, column: str) -> Any:
    prior = rows.loc[
        pd.to_numeric(rows["video_second"], errors="coerce").le(second)
        & (rows["item_after"].astype(str).eq(item) | rows["item_before"].astype(str).eq(item))
    ]
    values = prior[column].dropna().astype(str)
    values = values.loc[~values.isin({"", "nan", "unresolved"})]
    return values.iloc[-1] if not values.empty else "unresolved"


def _latest_character_level(rows: pd.DataFrame, second: float) -> Any:
    """Return the latest player level, independent of the rewarded item.

    Character level is global game state. Looking it up through the rewarded
    item's history can incorrectly fall back to that item's opening row when
    the item has not changed recently.
    """

    prior = rows.loc[
        pd.to_numeric(rows["video_second"], errors="coerce").le(second)
    ].copy()
    prior["_video_second"] = pd.to_numeric(
        prior["video_second"], errors="coerce"
    )
    prior = prior.sort_values("_video_second", kind="stable")
    values = prior["character_level"].dropna().astype(str)
    values = values.loc[~values.isin({"", "nan", "unresolved"})]
    return values.iloc[-1] if not values.empty else "unresolved"


def apply_initial_level_observations(
    events: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    """Infer segment-opening levels from later automated HUD pip evidence."""

    output = events.copy()
    if observations.empty:
        return output
    event_times = pd.to_numeric(output["video_second"], errors="coerce")
    for observation in observations.sort_values("video_second").itertuples(index=False):
        item = str(observation.item_name)
        item_type = str(observation.item_type)
        observed_level = int(observation.observed_level)
        observation_second = float(observation.video_second)
        initial = output.loc[
            output["event_source"].astype(str).eq("initial_state")
            & output["item_type"].astype(str).eq(item_type)
            & output["item_after"].astype(str).eq(item)
        ]
        if initial.empty:
            continue
        prior_upgrades = output.loc[
            event_times.lt(observation_second)
            & ~output["event_source"].astype(str).eq("initial_state")
            & output["event_type"].astype(str).eq("upgrade")
            & output["item_after"].astype(str).eq(item)
        ]
        inferred = observed_level - len(prior_upgrades)
        if inferred < 1:
            continue
        index = initial.index[0]
        output.at[index, "level_after"] = inferred
        output.at[index, "inference_method"] = (
            "first_menu_pips_minus_prior_upgrades"
        )
    return output


def reconcile_inventory_with_chest_rewards(*, base_inventory_path: Path,
                                           chest_rewards_path: Path,
                                           output_dir: Path, fps: float,
                                           level_observations_path: Path | None = None) -> dict[str, Any]:
    base = pd.read_csv(base_inventory_path)
    rewards = [json.loads(line) for line in chest_rewards_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted = [row for row in rewards if row.get("publication_status") == "auto_accepted"]
    if len(accepted) != len(rewards):
        raise ValueError("Inventory reconciliation requires every detected chest reward identity to be accepted")
    # HUD replacement inference is useful before chest analysis, but detected
    # reward sprites are the authoritative source once the chest stage succeeds.
    retained = base.loc[~base["event_source"].astype(str).eq("treasure_chest")].copy()
    additions = []
    for reward in accepted:
        second = float(reward["anchor_time_ms"]) / 1000
        item_after = str(reward["item_name"])
        base_item = str(reward.get("attributes", {}).get("base_item") or "")
        item_before = base_item if reward.get("action") == "evolution" else item_after
        character_level = _latest_character_level(retained, second)
        slot = _latest_value(retained, second, item_before, "slot")
        additions.append({
            "video": str(retained.iloc[0]["video"]), "video_second": second,
            "frame_number": round(second * fps), "character_level": character_level,
            "event_source": "treasure_chest", "event_type": reward["action"],
            "item_type": reward["item_type"], "slot": slot,
            "item_before": item_before, "item_after": item_after,
            "level_before": "", "level_after": "", "normal_max_level": "",
            "suggested_item": item_after, "confidence": "high", "needs_review": False,
        })
    combined = pd.concat([retained.drop(columns=["event_id"], errors="ignore"), pd.DataFrame(additions)], ignore_index=True)
    if level_observations_path is not None and level_observations_path.is_file():
        combined = apply_initial_level_observations(
            combined, pd.read_csv(level_observations_path)
        )
    combined = rebuild_inventory_progression(combined)
    combined = flag_out_of_range_slots(combined).reindex(columns=EVENT_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "inventory_events.csv"
    combined.to_csv(output, index=False)
    manifest = {
        "artifact_type": "vss_framework_inventory_chest_reconciliation",
        "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi", "status": "complete",
        "counts": {"base_rows": len(base), "removed_hud_chest_inferences": len(base) - len(retained),
                   "automatic_chest_rewards": len(accepted), "inventory_rows": len(combined),
                   "by_event_type": dict(Counter(combined["event_type"].astype(str)))},
        "inputs": {
            "base_inventory_sha256": sha256_file(base_inventory_path),
            "chest_rewards_sha256": sha256_file(chest_rewards_path),
            "initial_level_observations_sha256": (
                sha256_file(level_observations_path)
                if level_observations_path is not None
                and level_observations_path.is_file()
                else None
            ),
        },
        "outputs": {"inventory_events": {"path": output.name, "sha256": sha256_file(output), "row_count": len(combined)}},
        "policies": {"human_coded_ground_truth_used": False, "manual_chest_audit_loaded": False},
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
