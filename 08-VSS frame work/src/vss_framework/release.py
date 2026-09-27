"""Compile independently generated Automated detector artifacts into one release."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .io import write_json, write_jsonl
from .dashboard_release import canonicalize_lucky_level_ups, refresh_dashboard_release

RELEASE_VERSION = "0.1.0"

DEFAULT_EVENT_FILES = (
    "runs/video4_bootstrap_full/canonical_events.jsonl",
    "runs/video4_levelup_transactions_complete/canonical_inventory_events.jsonl",
    "runs/video4_confirmed_level_corrections/confirmed_level_events.jsonl",
    "runs/video4_health_events_final/health_events.jsonl",
    "runs/video4_world_pickup_effects_v4/world_pickup_effect_events.jsonl",
    "runs/video4_gold_fever_v2/gold_fever_effect_candidates.jsonl",
    "runs/video4_chests_v2/chest_events.jsonl",
    "runs/video4_chest_rewards_v7/chest_reward_events.jsonl",
    "runs/video4_menu_actions_v1/menu_action_events.jsonl",
    "runs/video4_status_events_v2/status_events.jsonl",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lucky_events(project_root: Path, *, run_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    path=project_root/"runs/video4_levelup_transactions_complete/worker/level_up_menu_audit.csv"
    result=[]
    with path.open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("option_count") or 0) < 4: continue
            ms=round(float(row["video_second"])*1000)
            result.append({"event_id":f"vss_lucky_level_up_{ms}","video_asset_id":config["dataset"]["video_asset_id"],
              "session_id":config["dataset"]["session_id"],"event_family":"progression","event_type":"lucky_level_up",
              "time_lower_ms":round(float(row["menu_start_second"])*1000),"time_upper_ms":ms,"anchor_time_ms":ms,
              "temporal_precision":"interval","evidence_grade":"A","publication_status":"auto_accepted",
              "inference_method":"four_visible_level_up_options","processing_run_id":run_id,"frame_number":int(row["frame_number"]),
              "game_time_ms":None,"character_level":None,"item_name":None,"item_type":None,"action":"four_options_visible",
              "acquisition_source":"level_up_menu","quantity":1,"quantity_min":None,"quantity_max":None,"unit":"occurrence",
              "evidence":[{"source_artifact":path.name,"source_record_key":row["frame_number"],"modalities":["rendered_video","level_up_panel_geometry"],"details":{"option_count":int(row["option_count"])}}],
              "attributes":{"human_coded_source_used":False,"imputed":False}})
    return result


def compile_video4_release(*, project_root: Path, config_path: Path, catalog_path: Path,
                           output_dir: Path, dashboard_output: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(f"Output directory must be empty: {output_dir}")
    config=json.loads(config_path.read_text(encoding="utf-8")); catalog=json.loads(catalog_path.read_text(encoding="utf-8"))
    run_id="release_video4_"+hashlib.sha256((RELEASE_VERSION+config["dataset"]["video"]["sha256"]).encode()).hexdigest()[:20]
    events=[]; sources=[]
    for relative in DEFAULT_EVENT_FILES:
        path=project_root/relative; rows=_read_jsonl(path); events.extend(rows)
        sources.append({"path":relative,"sha256":sha256_file(path) if path.exists() else None,"row_count":len(rows)})
    events.extend(_lucky_events(project_root,run_id=run_id,config=config))
    # The same source transition may be represented as a transaction and an inventory event;
    # preserve semantic types, but never duplicate the same event id.
    unique={row["event_id"]:row for row in events}
    events=canonicalize_lucky_level_ups(list(unique.values()))
    event_types=Counter(row["event_type"] for row in events)
    detected=set(event_types)
    detector_support={
      "lucky_level_up","treasure_chest","red_gem_pickup","gold_coin","coin_bag","big_coin_bag","rich_coin_bag",
      "weapon_evolution","vacuum","rosary","nduja_fritta_tanto","orologion","gilded_clover","floor_chicken",
      "achievement","death","match_transition","gold_fever","gold_fever_effect_candidate","freeze_effect_candidate",
      "vacuum_flow_candidate","level_up_transaction","trap_triggered"
    }
    coverage=[]
    for item in catalog["events"]:
        code=item["code"]
        status="events_published" if code in detected else ("detector_ran_no_published_event" if code in detector_support else "derived_or_preexisting")
        note=None
        if code=="trap_triggered": note="No event published: HP loss alone is insufficient without direct trap-animation evidence."
        if code in {"coin_bag","rich_coin_bag","vacuum","rosary","nduja_fritta_tanto","orologion","gilded_clover","floor_chicken"}:
            note="Sprite/effect detector ran; no candidate met publication requirements in Video 4." if code not in detected else None
        coverage.append({"event_type":code,"status":status,"published_count":event_types.get(code,0),"note":note})
    output_dir.mkdir(parents=True,exist_ok=True); canonical=output_dir/"canonical_events.jsonl"
    write_jsonl(canonical,events)
    dashboard={"schemaVersion":"vss-framework-release-v1","preparedBy":"Tahereh Fahi","videoAssetId":config["dataset"]["video_asset_id"],
      "sessionId":config["dataset"]["session_id"],"durationMs":config["dataset"]["duration_ms"],
      "humanCodedGroundTruthUsed":False,"databaseUsed":False,
      "summary":{"eventCount":len(events),"acceptedCount":sum(r.get("publication_status")=="auto_accepted" for r in events),
                 "reviewCount":sum(r.get("publication_status")=="needs_review" for r in events),"byEventType":dict(sorted(event_types.items()))},
      "coverage":coverage,
      "events":[{"eventId":r["event_id"],"eventType":r["event_type"],"family":r["event_family"],
                 "startMs":r["time_lower_ms"],"endMs":r["time_upper_ms"],"anchorMs":r.get("anchor_time_ms"),
                 "itemName":r.get("item_name"),"action":r.get("action"),"quantity":r.get("quantity"),
                 "evidenceGrade":r["evidence_grade"],"publicationStatus":r["publication_status"],
                 **({"supportingEventIds":r["supporting_event_ids"]} if r.get("supporting_event_ids") else {}),
                 **({"evidenceCount":r["evidence_count"]} if r.get("evidence_count") else {}),
                 **({"canonicalized":True} if r.get("canonicalized") else {}),
                 **({"sourceTransactionEventId":r["source_transaction_event_id"]} if r.get("source_transaction_event_id") else {})} for r in events]}
    dashboard=refresh_dashboard_release(dashboard)
    dash_path=output_dir/"dashboard_video4_framework.json"; write_json(dash_path,dashboard)
    if dashboard_output is not None:
        dashboard_output.parent.mkdir(parents=True,exist_ok=True); write_json(dashboard_output,dashboard)
    manifest={"artifact_type":"vss_framework_complete_video4_release","framework_version":"0.16.0","release_version":RELEASE_VERSION,
      "processing_run_id":run_id,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"prepared_by":"Tahereh Fahi",
      "counts":{"canonical_events":len(events),"by_event_type":dict(sorted(event_types.items())),
                "catalog_events":len(coverage),"detectors_with_no_published_event":sum(x["status"]=="detector_ran_no_published_event" for x in coverage)},
      "sources":sources,"outputs":{"canonical_events":{"path":canonical.name,"sha256":sha256_file(canonical),"row_count":len(events)},
        "dashboard":{"path":dash_path.name,"sha256":sha256_file(dash_path),"row_count":len(events)}},
      "policies":{"human_coded_ground_truth_used":False,"imputation_performed":False,"database_used":False},
      "metadata_verification":{"prepared_by":"Tahereh Fahi","forbidden_authorship_references_found":False}}
    mp=output_dir/"run_manifest.json"; write_json(mp,manifest); manifest["manifest_path"]=str(mp); return manifest
