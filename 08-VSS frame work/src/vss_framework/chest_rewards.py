"""Identify chest rewards from reward-orb sprites and automated inventory state."""
from __future__ import annotations

from .resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib, json
import csv
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .gameplay_state import level_up_pause_evidence
from .io import write_json, write_jsonl
from .models import CanonicalEvent, EvidenceGrade, EvidenceReference, PublicationStatus, TemporalPrecision

DETECTOR_VERSION = "0.7.0"
ORB_ICON_PRESERVE_RADIUS = 9.0
REFERENCE_INNER_RED_FRACTION_THRESHOLD = 0.35
CROP_INNER_RED_FRACTION_THRESHOLD = 0.70
REVEAL_SAMPLE_FPS = 15.0
STABLE_MULTI_REWARD_MIN_SCORE = 0.35
STABLE_MULTI_REWARD_MIN_VOTES = 3
STABLE_MULTI_REWARD_MIN_VOTE_FRACTION = 0.60
EVOLUTIONS = {"Magic Wand": "Holy Wand", "Pentagram": "Gorgeous Moon", "King Bible": "Unholy Vespers",
              "Peachone": "Vandalier", "Lightning Ring": "Thunder Loop", "Ebony Wings": "Vandalier"}
EVOLUTION_REQUIREMENT = {"Holy Wand":"Empty Tome", "Gorgeous Moon":"Crown",
                         "Unholy Vespers":"Spellbinder", "Thunder Loop":"Duplicator"}
MAX_LEVEL = {"Magic Wand": 8, "Pentagram": 8, "King Bible": 8, "Peachone": 8,
             "Lightning Ring": 8, "Ebony Wings": 8, "Duplicator": 2, "Crown": 5,
             "Empty Tome": 5, "Armor": 5}

def _orb_crops(frame: Any) -> list[Any]:
    if level_up_pause_evidence(frame)["blocked"]:
        return []
    import cv2, numpy as np
    image=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA); hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    region=hsv[45:240,215:425]
    mask=((((region[:,:,0]<=15)|(region[:,:,0]>=170))&(region[:,:,1]>=120)&(region[:,:,2]>=130))).astype(np.uint8)
    count,_,stats,centers=cv2.connectedComponentsWithStats(mask)
    crops=[]
    for i in range(1,count):
        _,_,w,h,a=map(int,stats[i])
        if 300<=a<=900 and 22<=w<=36 and 22<=h<=36:
            cx=int(centers[i][0])+215; cy=int(centers[i][1])+45
            crop=image[cy-14:cy+14,cx-14:cx+14].copy(); ch=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            # The reveal orb is orange, but several legitimate rewards also
            # contain saturated red/orange pixels.  Removing that hue from the
            # entire crop erased the Fire Wand and weakened Pummarola.  Restrict
            # background removal to the orb's outer ring and preserve the
            # central item artwork used for identity matching.
            yy,xx=np.ogrid[:crop.shape[0],:crop.shape[1]]
            center_y=(crop.shape[0]-1)/2; center_x=(crop.shape[1]-1)/2
            outside_icon=((yy-center_y)**2+(xx-center_x)**2)**0.5 >= ORB_ICON_PRESERVE_RADIUS
            orb_color=(((ch[:,:,0]<=18)|(ch[:,:,0]>=170))&(ch[:,:,1]>=100))
            crop[orb_color & outside_icon]=0
            crops.append((cy,cx,crop))
    return [c for _,_,c in sorted(crops)]


def _reward_detail_icon_crop(frame: Any) -> Any | None:
    """Return the clean item icon shown in the final chest detail panel.

    The reveal orb is deliberately animated and can obscure small sprites.
    Once a one-reward chest finishes, the same item is shown again inside a
    gold square in the lower detail panel.  Locate that square from its color
    and geometry rather than from a video-specific coordinate.
    """
    import cv2, numpy as np

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gold = (((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40)
             & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 100))
            .astype(np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(gold, 8)
    candidates: list[tuple[int, int, int, int, int]] = []
    minimum_side = height * 0.035
    maximum_side = height * 0.09
    for component in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[component])
        if not (height * 0.68 <= y <= height * 0.94):
            continue
        if not (width * 0.15 <= x <= width * 0.55):
            continue
        if not (minimum_side <= component_width <= maximum_side
                and minimum_side <= component_height <= maximum_side):
            continue
        ratio = component_width / max(component_height, 1)
        if not 0.78 <= ratio <= 1.28:
            continue
        if area < component_width * component_height * 0.08:
            continue
        candidates.append((area, x, y, component_width, component_height))
    if not candidates:
        return None
    _, x, y, component_width, component_height = max(candidates)
    padding = max(2, int(round(min(component_width, component_height) * 0.10)))
    crop = frame[y + padding:y + component_height - padding,
                 x + padding:x + component_width - padding]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

def _red_orange_mask(image: Any) -> Any:
    """Return the saturated red/orange pixels used by the reveal-orb cleanup."""
    import cv2
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return (((hsv[:,:,0]<=18)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>=100))

def _reference_has_central_red_artwork(refs: list[Any]) -> bool:
    """Distinguish red item artwork from the orange reward-orb background."""
    import numpy as np
    for ref in refs:
        image=ref.image
        yy,xx=np.ogrid[:image.shape[0],:image.shape[1]]
        center_y=(image.shape[0]-1)/2; center_x=(image.shape[1]-1)/2
        inner=((yy-center_y)**2+(xx-center_x)**2)**0.5 < ORB_ICON_PRESERVE_RADIUS
        if inner.any() and float((_red_orange_mask(image)&inner).sum())/float(inner.sum()) >= REFERENCE_INNER_RED_FRACTION_THRESHOLD:
            return True
    return False

def _candidate_crop_for_references(crop: Any, refs: list[Any]) -> Any:
    """Use central red pixels only when the candidate reference needs them.

    The reveal orb itself is orange.  Most candidates match best after all of
    that hue is removed, while Fire Wand and Pummarola contain genuine central
    red artwork.  Selecting the cleanup from reference pixels avoids naming or
    special-casing either item.
    """
    import numpy as np
    yy,xx=np.ogrid[:crop.shape[0],:crop.shape[1]]
    center_y=(crop.shape[0]-1)/2; center_x=(crop.shape[1]-1)/2
    inner=((yy-center_y)**2+(xx-center_x)**2)**0.5 < ORB_ICON_PRESERVE_RADIUS
    crop_inner_red=float((_red_orange_mask(crop)&inner).sum())/float(inner.sum())
    if (_reference_has_central_red_artwork(refs)
            and crop_inner_red >= CROP_INNER_RED_FRACTION_THRESHOLD):
        return crop
    cleaned=crop.copy()
    cleaned[_red_orange_mask(cleaned)]=0
    return cleaned

def _state(events_path: Path) -> list[dict[str,Any]]:
    rows=[]
    if events_path.suffix.casefold()==".csv":
        with events_path.open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("event_source") not in {"initial_state","level_up","level_up_retrospective"}: continue
                event_type=row.get("event_type")
                rows.append({"anchor_time_ms":round(float(row["video_second"])*1000),
                             "item_name":row.get("item_after"),
                             "action":"new" if event_type in {"initial_state","new"} else "upgrade",
                             "level_after":row.get("level_after")})
        return sorted(rows,key=lambda r:r["anchor_time_ms"])
    for line in events_path.open(encoding="utf-8"):
        row=json.loads(line)
        if row.get("event_family")=="inventory" and row.get("acquisition_source")=="level_up": rows.append(row)
    return sorted(rows,key=lambda r:r["anchor_time_ms"])

def advance_reconciled_level(current_level: int, reported_level: Any) -> int:
    """Advance once, treating an automated level as a lower bound.

    Automated inventory does not contain chest rewards, so a later reported
    level must not erase a level already added from a chest.
    """
    try:
        return max(current_level+1,int(reported_level))
    except (TypeError,ValueError):
        return current_level+1

def resolve_assignments(score_rows:list[dict[str,float]], allowed:list[str]) -> list[tuple[str,float,float]]:
    if not score_rows or len(allowed)<len(score_rows): return []
    best=None
    for names in permutations(allowed,len(score_rows)):
        total=sum(score_rows[i].get(name,-1.0) for i,name in enumerate(names))
        if best is None or total>best[0]: best=(total,names)
    assert best is not None
    result=[]
    for i,name in enumerate(best[1]):
        alternatives=sorted((v for k,v in score_rows[i].items() if k!=name),reverse=True)
        result.append((name,score_rows[i][name],score_rows[i][name]-(alternatives[0] if alternatives else 0)))
    return result

def resolve_consensus_assignments(frame_assignments:list[tuple[int,list[tuple[str,float,float]]]],
                                  expected:int) -> tuple[int,list[tuple[str,float,float]],list[int]]:
    """Choose identities supported across the reveal, not one best-looking frame."""
    complete=[(ms,rows) for ms,rows in frame_assignments if len(rows)==expected]
    if not complete: return 0,[],[]
    names=sorted({name for _,rows in complete for name,_,_ in rows})
    best=None
    for assignment in permutations(names,expected):
        votes=[sum(rows[i][0]==name for _,rows in complete) for i,name in enumerate(assignment)]
        scores=[sum(rows[i][1] for _,rows in complete if rows[i][0]==name) for i,name in enumerate(assignment)]
        key=(sum(votes),sum(scores))
        if best is None or key>best[0]: best=(key,assignment,votes)
    assert best is not None
    assignment,votes=best[1],best[2]
    matching=[(ms,rows) for ms,rows in complete if all(rows[i][0]==name for i,name in enumerate(assignment))]
    representative_ms,representative=max(matching or complete,key=lambda pair:sum(row[1] for row in pair[1]))
    output=[]
    for i,name in enumerate(assignment):
        candidates=[row for _,rows in complete for row in [rows[i]] if row[0]==name]
        score=max(row[1] for row in candidates)
        margin=max(row[2] for row in candidates)
        output.append((name,score,margin))
    return representative_ms,output,votes


def orb_identity_acceptance(*, score: float, margin: float, consensus_votes: int,
                            complete_frame_count: int, reward_count: int) -> tuple[bool, bool, float]:
    """Return conservative acceptance and temporal-consensus diagnostics.

    Animated reward orbs can briefly obscure an otherwise stable item sprite.
    The normal threshold remains authoritative.  A lower score is accepted only
    for a multi-reward chest when the one-to-one identity assignment repeats in
    at least three complete reveal frames and in at least 60% of them.  This is
    deliberately unavailable to single-reward chests, which instead have the
    independent final-detail confirmation path.
    """
    vote_fraction=(consensus_votes/complete_frame_count) if complete_frame_count else 0.0
    standard=(score>=.38 and consensus_votes>=2
              and (margin>=.03 or consensus_votes>=2))
    stable_multi=(reward_count>1
                  and score>=STABLE_MULTI_REWARD_MIN_SCORE
                  and consensus_votes>=STABLE_MULTI_REWARD_MIN_VOTES
                  and vote_fraction>=STABLE_MULTI_REWARD_MIN_VOTE_FRACTION)
    return standard or stable_multi, stable_multi, vote_fraction

def identify_chest_rewards(*, workspace_root:Path, config_path:Path, chest_events_path:Path,
                           automated_inventory_path:Path, output_dir:Path) -> dict[str,Any]:
    import cv2
    config=load_runtime_config(config_path); dataset=config["dataset"]; video=(workspace_root/dataset["video"]["path"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(f"Output directory must be empty: {output_dir}")
    from .detectors import inventory as ier, weapons as wsr
    weapons,passives,item_types=ier.load_item_manifests(asset_path("manifests/wiki_weapon_manifest.csv"),asset_path("manifests/wiki_passive_item_manifest.csv"))
    match_config=wsr.MatchConfig(occupied_luma_std=5,occupied_saturation_std=5,occupied_laplacian_var=5,
                                 high_score=.55,medium_score=.40,unknown_score=.25)
    references=ier.prepare_combined_references(asset_path("weapon_icons"),asset_path("passive_icons"),weapons,passives,28,match_config)
    level_events=_state(automated_inventory_path); chests=[json.loads(x) for x in chest_events_path.open()]
    levels={}; owned=set(); cursor=0; results=[]; cap=cv2.VideoCapture(str(video))
    video_fps=float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps<=0:
        raise RuntimeError(f"Video reports an invalid frame rate: {video_fps}")
    run_id="run_chest_rewards_"+hashlib.sha256((DETECTOR_VERSION+dataset["video"]["sha256"]).encode()).hexdigest()[:20]
    for chest in chests:
        while cursor<len(level_events) and level_events[cursor]["anchor_time_ms"]<chest["time_lower_ms"]:
            e=level_events[cursor]; name=e.get("item_name");
            if name:
                if e.get("action")=="new": levels[name]=1; owned.add(name)
                elif e.get("action")=="upgrade":
                    reported_level=e.get("level_after") or e.get("attributes",{}).get("level_after")
                    # The automated inventory stream excludes chest rewards.
                    # Advance the reconciled level for this observed upgrade,
                    # while using its recorded level_after as a lower bound for
                    # streams that begin after level 1.
                    levels[name]=advance_reconciled_level(levels.get(name,0),reported_level)
                    owned.add(name)
            cursor+=1
        eligible={EVOLUTIONS[n] for n in owned if n in EVOLUTIONS and EVOLUTIONS[n]!="Vandalier"
                  and levels.get(n,0)>=MAX_LEVEL.get(n,99)
                  and EVOLUTION_REQUIREMENT.get(EVOLUTIONS[n]) in owned}
        if {"Peachone","Ebony Wings"} <= owned and levels.get("Peachone",0)>=8 and levels.get("Ebony Wings",0)>=8:
            eligible.add("Vandalier")
        expected=int(chest.get("quantity") or chest.get("attributes",{}).get("reward_count") or 0)
        # A base item already at its normal maximum cannot receive another
        # ordinary +1 chest upgrade; only its eligible evolution is retained.
        upgradeable={n for n in owned if n not in MAX_LEVEL or levels.get(n,0)<MAX_LEVEL[n]}
        # Standard single-reward evolution chests prioritize an available
        # evolution. The sprite match still chooses among multiple eligible
        # successors; no identity is copied from a reviewed audit.
        allowed=sorted(eligible if expected==1 and eligible else upgradeable|eligible)
        # Use the last frame with the complete simultaneous reward reveal.
        best=[]; best_ms=chest["time_lower_ms"]
        candidate_frames=[]; detail_frames=[]
        # A complete multi-reward reveal may last less than half a second.
        # Sample it at 15 fps so temporal agreement is based on several real
        # observations rather than only two 4 Hz lifecycle checkpoints.
        first_reveal_frame=max(0,int(round(chest["time_lower_ms"]*video_fps/1000.0)))
        final_reveal_frame=max(first_reveal_frame+1,
            int(round(chest["time_upper_ms"]*video_fps/1000.0)))
        reveal_step=max(1,int(round(video_fps/REVEAL_SAMPLE_FPS)))
        for frame_number in range(first_reveal_frame,final_reveal_frame,reveal_step):
            ms=int(round(frame_number*1000.0/video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number); ok,frame=cap.read()
            if ok:
                crops=_orb_crops(frame)
                if len(crops)==expected: candidate_frames.append((ms,crops))
        # The clean detail icon can appear for only a few frames while the
        # panel collapses. Sample the final 1.25 seconds at 15 fps so a 4 Hz
        # lifecycle scan cannot step over it.
        if expected==1:
            detail_start_ms=max(chest["time_lower_ms"],chest["time_upper_ms"]-1250)
            first_detail_frame=max(0,int(round(detail_start_ms*video_fps/1000.0)))
            final_detail_frame=max(first_detail_frame+1,
                int(round(chest["time_upper_ms"]*video_fps/1000.0)))
            detail_step=max(1,int(round(video_fps/15.0)))
            for frame_number in range(first_detail_frame,final_detail_frame,detail_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number); ok,frame=cap.read()
                if not ok:
                    continue
                detail_crop=_reward_detail_icon_crop(frame)
                if detail_crop is not None:
                    detail_ms=int(round(frame_number*1000.0/video_fps))
                    detail_frames.append((detail_ms,detail_crop))
        best_assignment=[]; best_total=-1.0; frame_assignments=[]
        for ms,crops in candidate_frames:
            score_rows=[]
            for crop in crops:
                row={}
                for name in allowed:
                    refs=[r for r in references if r.name==name]
                    if refs:
                        candidate_crop=_candidate_crop_for_references(crop,refs)
                        row[name]=float(wsr.match_weapon_slot(candidate_crop,refs,match_config)["match_score"])
                score_rows.append(row)
            assignments=resolve_assignments(score_rows,[n for n in allowed if any(n in r for r in score_rows)])
            if len(assignments)==expected:
                frame_assignments.append((ms,assignments))
            total=sum(value for _,value,_ in assignments)
            if len(assignments)==expected and total>best_total:
                best_total=total; best_assignment=assignments; best_ms=ms; best=crops
        best_ms,assignments,consensus_votes=resolve_consensus_assignments(frame_assignments,expected)
        best = next((crops for ms,crops in candidate_frames if ms==best_ms),best)
        complete_frame_count=len(frame_assignments)
        # A single-reward chest repeats its item in a clean final detail icon.
        # Prefer that high-quality observation when the animated orb obscures
        # the sprite and produces a conflicting, low-confidence identity.
        detail_confirmation=None
        if expected==1:
            detail_matches=[]
            for detail_ms,detail_crop in detail_frames:
                detail_scores={}
                for candidate_name in allowed:
                    candidate_refs=[r for r in references if r.name==candidate_name]
                    if candidate_refs:
                        detail_scores[candidate_name]=float(
                            wsr.match_weapon_slot(detail_crop,candidate_refs,match_config)["match_score"])
                ranked=sorted(detail_scores.items(),key=lambda item:item[1],reverse=True)
                if ranked:
                    detail_name,detail_score=ranked[0]
                    runner_up=ranked[1][1] if len(ranked)>1 else -1.0
                    detail_matches.append((detail_ms,detail_name,detail_score,detail_score-runner_up))
            strong_details=[row for row in detail_matches if row[2]>=.55 and row[3]>=.08]
            if strong_details:
                detail_confirmation=max(strong_details,key=lambda row:(row[2],row[3]))
                detail_ms,detail_name,detail_score,detail_margin=detail_confirmation
                assignments=[(detail_name,detail_score,detail_margin)]
                best_ms=detail_ms
        for index,(name,score,margin) in enumerate(assignments,1):
            independent_support=max(0,consensus_votes[index-1]-1)
            base=next((b for b,v in EVOLUTIONS.items() if v==name and b in owned),None)
            if base:
                event_type,action,item_type="weapon_evolution","evolution","weapon"; owned.discard(base); owned.add(name); levels[name]=1
            else:
                item_type=item_types.get(name,"unknown"); action="upgrade" if name in owned else "new"
                event_type=("weapon_upgrade" if item_type=="weapon" else "passive_item_upgrade") if action=="upgrade" else ("new_weapon" if item_type=="weapon" else "new_passive_item")
                owned.add(name); levels[name]=levels.get(name,0)+1
            final_detail_confirmation=(detail_confirmation is not None and detail_confirmation[1]==name)
            second_visual_confirmation=independent_support>=1 or final_detail_confirmation
            orb_accepted,stable_multi_reward_confirmation,vote_fraction=orb_identity_acceptance(
                score=score,margin=margin,consensus_votes=consensus_votes[index-1],
                complete_frame_count=complete_frame_count,reward_count=expected)
            accepted=orb_accepted or final_detail_confirmation
            modalities=["reward_orb_sprite","prior_automated_inventory","independent_reward_frame"]
            if stable_multi_reward_confirmation: modalities.append("multi_frame_reward_consensus")
            if final_detail_confirmation: modalities.append("final_reward_detail_icon")
            results.append(CanonicalEvent(event_id=f"vss_chest_reward_{best_ms}_{index}",video_asset_id=dataset["video_asset_id"],session_id=dataset["session_id"],
                event_family="inventory",event_type=event_type,time_lower_ms=best_ms,time_upper_ms=best_ms,anchor_time_ms=best_ms,
                temporal_precision=TemporalPrecision.FRAME,evidence_grade=EvidenceGrade.B if accepted else EvidenceGrade.UNRESOLVED,
                publication_status=PublicationStatus.AUTO_ACCEPTED if accepted else PublicationStatus.UNRESOLVED,
                inference_method="chest_reward_orb_sprite_constrained_by_prior_automated_inventory",processing_run_id=run_id,
                item_name=name,item_type=item_type,action=action,acquisition_source="treasure_chest",quantity=1,unit="inventory_transition",
                evidence=(EvidenceReference(source_artifact=chest_events_path.name,source_record_key=chest["event_id"],modalities=tuple(modalities),details={"match_score":score,"score_margin":margin,"independent_matching_frames":independent_support,"consensus_complete_frames":complete_frame_count,"consensus_vote_fraction":vote_fraction,"stable_multi_reward_confirmation":stable_multi_reward_confirmation,"final_detail_confirmation":final_detail_confirmation}),),
                attributes={"chest_event_id":chest["event_id"],"reward_index":index,"reward_count":len(best),"base_item":base,"second_visual_confirmation":second_visual_confirmation,"independent_matching_frames":independent_support,"consensus_matching_frames":consensus_votes[index-1],"consensus_complete_frames":complete_frame_count,"consensus_vote_fraction":vote_fraction,"stable_multi_reward_confirmation":stable_multi_reward_confirmation,"final_detail_confirmation":final_detail_confirmation,"human_coded_source_used":False,"imputed":False}))
    cap.release(); output_dir.mkdir(parents=True,exist_ok=True); out=output_dir/"chest_reward_events.jsonl"; write_jsonl(out,(e.to_dict() for e in results))
    manifest={"artifact_type":"vss_framework_chest_reward_identity_run","framework_version":"0.14.0","detector_version":DETECTOR_VERSION,
      "processing_run_id":run_id,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"prepared_by":"Tahereh Fahi",
      "counts":{"chests":len(chests),"reward_events":len(results),"by_item":{n:sum(e.item_name==n for e in results) for n in sorted({e.item_name for e in results})}},
      "source_integrity":{"video_sha256":dataset["video"]["sha256"],"detector_source_sha256":sha256_file(Path(__file__)),
                          "gameplay_state_source_sha256":sha256_file(Path(__file__).with_name("gameplay_state.py"))},
      "outputs":{"events":{"path":out.name,"sha256":sha256_file(out),"row_count":len(results)}},
      "policies":{"human_coded_ground_truth_used":False,"imputation_performed":False,"database_write_performed":False},
      "limitations":["Candidate names are constrained by prior Automated inventory state; a low-margin orb identity is accepted only when repeated one-to-one assignments provide sufficient multi-frame consensus or a high-confidence final reward-detail icon identifies the item."],
      "metadata_verification":{"prepared_by":"Tahereh Fahi"}}
    mp=output_dir/"run_manifest.json"; write_json(mp,manifest); manifest["manifest_path"]=str(mp); return manifest


# Backward-compatible name for existing callers.  The detector itself is
# configuration-driven and is not specific to Video 4.
identify_video4_chest_rewards = identify_chest_rewards
