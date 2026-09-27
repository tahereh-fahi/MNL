"""Detect Reroll, Skip, and Banish use from direct menu counter changes."""
from __future__ import annotations
import csv, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from .hashing import sha256_file
from .gameplay_state import level_up_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision

DETECTOR_VERSION="0.3.0"
ACTIONS=("reroll","skip","banish")

def _normalize_counter_digits(raw:str)->int:
    """Repair the common OCR substitution of a leading '+' with '4'."""
    if len(raw)>=4 and raw.startswith("4"):
        repaired=int(raw[1:])
        if repaired<=999:
            return repaired
    return int(raw)

def parse_action_counter_text(texts:Sequence[str])->dict[str,int|None]:
    joined=" ".join(str(x) for x in texts)
    result={name:None for name in ACTIONS}
    for name in ACTIONS:
        match=re.search(rf"{name}\s*\+?\s*(\d+|[-–—])",joined,re.I)
        if match and match.group(1).isdigit(): result[name]=_normalize_counter_digits(match.group(1))
    return result

def resolve_action_counter_drops(rows:Sequence[dict[str,Any]],*,processing_run_id:str,
                                 video_asset_id:str,session_id:str)->list[CanonicalEvent]:
    events=[]
    for action in ACTIONS:
        available=[row for row in rows if isinstance(row.get(f"{action}_remaining"),int)]
        if not available: continue
        stable_row=available[0]; stable_value=int(stable_row[f"{action}_remaining"])
        for index,row in enumerate(available[1:],start=1):
            after=int(row[f"{action}_remaining"])
            if after==stable_value:
                stable_row=row
                continue
            if after>stable_value:
                # These counters cannot increase during a run. Treat an upward OCR
                # fluctuation as missing evidence and preserve the last stable state.
                continue
            drop=stable_value-after
            lookahead=available[index+1:index+4]
            confirmed=any(int(candidate[f"{action}_remaining"])==after for candidate in lookahead)
            terminal_small_drop=index==len(available)-1 and drop<=3
            if not (confirmed or terminal_small_drop):
                continue
            before=stable_value; previous=stable_row
            for offset in range(drop):
                events.append(CanonicalEvent(event_id=f"vss_{action}_{row['media_time_ms']}_{offset}",
                    video_asset_id=video_asset_id,session_id=session_id,event_family="menu_action",
                    event_type="level_up_transaction",time_lower_ms=int(previous["media_time_ms"]),
                    time_upper_ms=int(row["media_time_ms"]),anchor_time_ms=int(row["media_time_ms"]),
                    temporal_precision=TemporalPrecision.BOUNDED,evidence_grade=EvidenceGrade.B,
                    publication_status=PublicationStatus.AUTO_ACCEPTED,
                    inference_method="confirmed_level_up_action_counter_decrement",processing_run_id=processing_run_id,
                    action=action,acquisition_source="level_up_menu",quantity=1,unit="action",
                    evidence=(EvidenceReference(source_artifact="menu_action_counters.jsonl",
                        source_record_key=f"frames:{previous['frame_number']}-{row['frame_number']}",
                        modalities=("menu_counter_ocr",),details={"before":before,"after":after,"confirmed_by_later_observation":confirmed}),),
                    attributes={"counter_before":before,"counter_after":after,"confirmed_by_later_observation":confirmed,
                                "exact_click_time_observed":False,"imputed":False}))
            stable_value=after; stable_row=row
    events.sort(key=lambda event:(event.anchor_time_ms,event.action or "",event.event_id))
    return events

def scan_video4_menu_actions(*,workspace_root:Path,config_path:Path,menu_audit_path:Path,output_dir:Path)->dict[str,Any]:
    import cv2, easyocr
    config=json.loads(config_path.read_text()); dataset=config["dataset"]; video=(workspace_root/dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(f"Output directory must be empty: {output_dir}")
    reader=easyocr.Reader(["en"],gpu=False,verbose=False,download_enabled=False); capture=cv2.VideoCapture(str(video)); rows=[]
    with menu_audit_path.open(encoding="utf-8-sig",newline="") as handle:
        for source_index,row in enumerate(csv.DictReader(handle),start=2):
            frame_number=int(float(row["frame_number"])); capture.set(cv2.CAP_PROP_POS_FRAMES,frame_number); ok,frame=capture.read()
            if not ok: continue
            pause=level_up_pause_evidence(frame)
            if pause["phase"]!="level_up_menu":
                rows.append({"source_row":source_index,"frame_number":frame_number,
                             "media_time_ms":round(float(row["menu_end_second"])*1000),
                             "ocr_tokens":[],"gameplay_pause":pause,"rejection_reason":"stable_level_up_menu_not_observed",
                             **{f"{k}_remaining":None for k in ACTIONS}})
                continue
            h,w=frame.shape[:2]; crop=frame[int(h*.72):int(h*.90),int(w*.04):int(w*.25)]
            detected=reader.readtext(crop,detail=1,paragraph=False)
            tokens=[str(item[1]) for item in detected if float(item[2])>=.45]
            counters=parse_action_counter_text(tokens)
            rows.append({"source_row":source_index,"frame_number":frame_number,"media_time_ms":round(float(row["menu_end_second"])*1000),
                         "ocr_tokens":tokens,"gameplay_pause":pause,
                         **{f"{k}_remaining":v for k,v in counters.items()}})
    capture.release(); identity=DETECTOR_VERSION+dataset["video"]["sha256"]+sha256_file(menu_audit_path)
    run_id="run_menu_actions_"+hashlib.sha256(identity.encode()).hexdigest()[:20]
    events=resolve_action_counter_drops(rows,processing_run_id=run_id,video_asset_id=dataset["video_asset_id"],session_id=dataset["session_id"])
    output_dir.mkdir(parents=True,exist_ok=True); signals=output_dir/"menu_action_counters.jsonl"; out=output_dir/"menu_action_events.jsonl"
    write_jsonl(signals,rows); write_jsonl(out,(e.to_dict() for e in events)); by_action={a:sum(e.action==a for e in events) for a in ACTIONS}
    manifest={"artifact_type":"vss_framework_menu_action_run","framework_version":"0.15.0","detector_version":DETECTOR_VERSION,
      "processing_run_id":run_id,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"prepared_by":"Tahereh Fahi",
      "counts":{"menu_observations":len(rows),"action_events":len(events),"by_action":by_action,
                "readable_counters":{a:sum(r[f'{a}_remaining'] is not None for r in rows) for a in ACTIONS}},
      "source_integrity":{"video_sha256":dataset["video"]["sha256"],"menu_audit_sha256":sha256_file(menu_audit_path),"detector_source_sha256":sha256_file(Path(__file__)),
                          "gameplay_state_source_sha256":sha256_file(Path(__file__).with_name("gameplay_state.py"))},
      "outputs":{"signals":{"path":signals.name,"sha256":sha256_file(signals),"row_count":len(rows)},"events":{"path":out.name,"sha256":sha256_file(out),"row_count":len(events)}},
      "policies":{"human_coded_ground_truth_used":False,"imputation_performed":False,"database_write_performed":False},
      "limitations":["Action time is bounded between confirmed readable menu states; the exact click frame is not claimed.","Unavailable or unreadable counters remain null, never zero.","A non-terminal counter drop requires a matching later observation; isolated OCR drops and impossible counter increases are rejected."],
      "metadata_verification":{"prepared_by":"Tahereh Fahi"}}
    mp=output_dir/"run_manifest.json"; write_json(mp,manifest); manifest["manifest_path"]=str(mp); return manifest
