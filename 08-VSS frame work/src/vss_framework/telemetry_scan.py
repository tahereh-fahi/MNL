"""Direct Video 4 Coin Counter and player health-bar observations."""

from __future__ import annotations

from .resources import load_runtime_config, resolve_path, source_reference, detector_path, asset_path

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hashing import sha256_file
from .io import write_json, write_jsonl
from .models import EvidenceGrade, SignalObservation, TemporalPrecision, Visibility
from .video import OpenCVVideoReader
from .gold_fever import gold_fever_overlay_features, mark_gold_fever_rows
from .gameplay_state import gameplay_pause_evidence


DETECTOR_VERSION = "0.5.0"
MAX_PLAUSIBLE_COIN_INCREASE_PER_SAMPLE = 2_000
MAX_GOLD_FEVER_COIN_INCREASE_PER_SAMPLE = 500
MIN_GOLD_FEVER_COIN_CONFIDENCE = 0.95


def is_plausible_coin_increase(candidate: int, confirmed: int) -> bool:
    """Bound a single sampled counter increase without imputing a value."""

    return candidate - confirmed <= MAX_PLAUSIBLE_COIN_INCREASE_PER_SAMPLE


def is_leading_digit_ocr_substitution(candidate: int, confirmed: int) -> bool:
    """Return True when only the first digit changed in an otherwise stable counter."""

    candidate_text = str(candidate)
    confirmed_text = str(confirmed)
    return (
        len(candidate_text) == len(confirmed_text)
        and len(candidate_text) > 1
        and candidate_text[0] != confirmed_text[0]
        and candidate_text[1:] == confirmed_text[1:]
    )


def has_attached_digit_ocr_artifact(candidate: int, confirmed: int) -> bool:
    """Return True when OCR attached one non-counter digit beside the counter."""

    candidate_text = str(candidate)
    confirmed_text = str(confirmed)
    return (
        len(candidate_text) == len(confirmed_text) + 1
        and (candidate_text.startswith(confirmed_text) or candidate_text.endswith(confirmed_text))
    )


def is_gold_fever_monotonic_coin_candidate(candidate: int, confirmed: int,
                                            confidence: float | None) -> bool:
    """Validate one rapid Gold Fever total-counter step without accepting OCR noise.

    Gold Fever changes the *upper-right cumulative counter* too quickly for the
    usual exact-repeat rule. A candidate is only eligible for the separate
    two-sample monotonic confirmation path when it is a high-confidence,
    bounded increase and does not resemble a known digit-attachment error.
    """

    return (
        confidence is not None
        and confidence >= MIN_GOLD_FEVER_COIN_CONFIDENCE
        and candidate > confirmed
        and candidate - confirmed <= MAX_GOLD_FEVER_COIN_INCREASE_PER_SAMPLE
        and not has_attached_digit_ocr_artifact(candidate, confirmed)
        and not is_leading_digit_ocr_substitution(candidate, confirmed)
    )


def select_coin_candidate(results: Sequence[Any], crop_height: int) -> tuple[int | None, float | None, str | None]:
    """Select the bottom-row numeric OCR candidate inside the Coin HUD crop."""

    candidates = []
    for bbox, text, confidence in results:
        digits = "".join(character for character in str(text) if character.isdigit())
        if not digits:
            continue
        center_y = sum(float(point[1]) for point in bbox) / len(bbox)
        if center_y < crop_height * 0.28:
            continue
        candidates.append((float(confidence), int(digits), str(text)))
    if not candidates:
        return None, None, None
    confidence, value, raw = max(candidates)
    return value, confidence, raw


def _load_module(path: Path, name: str) -> Any:
    from .detectors import gems
    return gems


def measure_health_bar_width(frame: Any, config: Any) -> tuple[float | None, float]:
    """Return the player's red health-fill width normalized to 1440p."""

    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = height / 1440.0
    center_x, center_y = width / 2.0, height / 2.0
    search_x = round(config.health_bar_search_x_1440p * scale)
    search_y = round(config.health_bar_search_y_1440p * scale)
    x0, x1 = max(0, round(center_x) - search_x), min(width, round(center_x) + search_x)
    y0, y1 = max(0, round(center_y) - search_y), min(height, round(center_y) + search_y)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    red = (((hue <= 8) | (hue >= 172)) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 70)).astype(np.uint8) * 255
    count, _, stats, centers = cv2.connectedComponentsWithStats(red, 8)
    expected_y = center_y + config.health_bar_expected_y_offset_1440p * scale
    candidates = []
    for component in range(1, count):
        _, _, component_width, component_height, area = map(int, stats[component])
        bar_x = x0 + float(centers[component, 0])
        bar_y = y0 + float(centers[component, 1])
        normalized_component_width = component_width / max(scale, 1e-6)
        is_standard_width = normalized_component_width >= config.health_bar_min_width_1440p
        if not 6.0 <= normalized_component_width <= config.health_bar_max_width_1440p:
            continue
        if not config.health_bar_min_height_1440p * scale <= component_height <= config.health_bar_max_height_1440p * scale:
            continue
        aspect_ratio = component_width / max(1, component_height)
        if is_standard_width and aspect_ratio < config.health_bar_min_aspect_ratio:
            continue
        if abs(bar_x - center_x) > config.health_bar_max_x_error_1440p * scale:
            continue
        if abs(bar_y - expected_y) > config.health_bar_max_y_error_1440p * scale:
            continue
        rectangularity = area / max(1, component_width * component_height)
        # A short fill has less aspect-ratio evidence than a full bar. Accept it
        # only in the tighter player-bar geometry with strong rectangularity.
        if not is_standard_width and (
            aspect_ratio < 1.2
            or rectangularity < 0.60
            or abs(bar_x - center_x) > 65.0 * scale
            or abs(bar_y - expected_y) > 14.0 * scale
        ):
            continue
        error = (abs(bar_x - center_x) + 2 * abs(bar_y - expected_y)) / max(scale, 1e-6)
        confidence = float(np.clip(0.55 + 0.35 * rectangularity - 0.002 * error, 0, 1))
        candidates.append((error - 12 * rectangularity, component_width / scale, confidence))
    if not candidates:
        return None, 0.0
    _, normalized_width, confidence = min(candidates)
    return float(normalized_width), confidence


def scan_video4_telemetry(
    *,
    workspace_root: Path,
    config_path: Path,
    output_dir: Path,
    start_second: float,
    max_seconds: float | None,
    sample_fps: float = 1.0,
) -> dict[str, Any]:
    import cv2
    import easyocr

    config = load_runtime_config(config_path)
    dataset = config["dataset"]
    video_path = (workspace_root / dataset["video"]["path"]).resolve()
    xp_script = resolve_path(config["detectors"]["gem_xp"]["script"], workspace_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if sha256_file(video_path) != dataset["video"]["sha256"]:
        raise ValueError("Video SHA-256 does not match the pinned Video 4 asset")
    xp_module = _load_module(xp_script, "vss_telemetry_xp_source")
    hp_config = xp_module.Config()
    reader = easyocr.Reader(["en"], gpu=False, download_enabled=False)
    observations = []
    last_observed_coin: int | None = None
    pending_coin: int | None = None
    pending_coin_ms: int | None = None
    pending_gold_fever_coin: int | None = None
    pending_gold_fever_coin_ms: int | None = None
    gameplay_segment_id = 0
    was_paused = False
    start_ms = round(start_second * 1000)
    end_ms = None if max_seconds is None else round(max_seconds * 1000)
    if start_ms < 0 or (end_ms is not None and end_ms <= start_ms):
        raise ValueError("Telemetry interval must have 0 <= start < end")
    with OpenCVVideoReader(video_path, dataset["video_asset_id"]) as video:
        effective_end = video.metadata.duration_ms if end_ms is None else min(end_ms, video.metadata.duration_ms)
        for packet in video.iter_packets(start_ms=start_ms, end_ms=effective_end, sample_fps=sample_fps):
            frame = packet.image
            pause = gameplay_pause_evidence(frame)
            paused = bool(pause["blocked"])
            if paused and not was_paused:
                gameplay_segment_id += 1
            was_paused = paused
            if paused:
                pending_coin = pending_coin_ms = None
                pending_gold_fever_coin = pending_gold_fever_coin_ms = None
                for code, unit in (("coin_counter", "gold_coins"),
                                   ("player_health_bar_width", "pixels_at_1440p")):
                    observations.append(SignalObservation(
                        observation_id=f"vss_{code}_{packet.frame_number}",
                        video_asset_id=dataset["video_asset_id"], session_id=dataset["session_id"],
                        detector_name="video4_direct_telemetry", detector_version=DETECTOR_VERSION,
                        observable_code=code, time_lower_ms=packet.media_time_ms,
                        time_upper_ms=packet.media_time_ms, temporal_precision=TemporalPrecision.FRAME,
                        visibility=Visibility.UNKNOWN, evidence_grade=EvidenceGrade.UNRESOLVED,
                        observed=False, source_artifact="VSS frame work/src/vss_framework/telemetry_scan.py",
                        source_record_key=f"frame:{packet.frame_number}", numeric_value=None,
                        unit=unit, frame_number=packet.frame_number,
                        attributes={"confidence": None, "raw_ocr_text": None,
                                    "rejection_reason": pause["reason"], "excluded_from_gameplay": True,
                                    "screen_state": pause["phase"], "gameplay_segment_id": gameplay_segment_id,
                                    "imputed": False, "health_value_calibrated": False if code.startswith("player_") else None},
                    ))
                continue
            height, width = frame.shape[:2]
            gold_fever_features = gold_fever_overlay_features(frame)
            mark_gold_fever_rows([gold_fever_features])
            gold_fever_active = bool(gold_fever_features["gold_fever_like"])
            # The Gold value is immediately left of the coin icon in the far
            # right HUD. Keep the crop tight so enemy sprites, damage numbers,
            # the Kill counter, and the level header cannot become extra OCR
            # digits (for example, visible 674 must not become 7,674).
            x0, x1 = int(0.890 * width), int(0.985 * width)
            y0, y1 = int(0.032 * height), int(0.082 * height)
            coin_crop = frame[y0:y1, x0:x1]
            coin_results = reader.readtext(coin_crop, allowlist="0123456789", detail=1)
            coin, coin_confidence, coin_raw = select_coin_candidate(coin_results, coin_crop.shape[0])
            coin_rejection_reason = None
            if coin is not None:
                if gold_fever_active and last_observed_coin is not None and coin > last_observed_coin:
                    # The permanent counter increases every sampled frame in
                    # Gold Fever. Do not require an exact repeated number;
                    # instead require two nearby eligible values progressing in
                    # the same direction. The first stays unpublished until
                    # the second corroborates the monotonic rise.
                    if not is_gold_fever_monotonic_coin_candidate(
                        coin, last_observed_coin, coin_confidence,
                    ):
                        coin_rejection_reason = "gold_fever_counter_candidate_rejected"
                        coin = None
                    elif (
                        pending_gold_fever_coin is not None
                        and pending_gold_fever_coin_ms is not None
                        and packet.media_time_ms - pending_gold_fever_coin_ms <= 2_000
                        and coin >= pending_gold_fever_coin
                        and coin - pending_gold_fever_coin <= MAX_GOLD_FEVER_COIN_INCREASE_PER_SAMPLE
                    ):
                        last_observed_coin = coin
                        pending_gold_fever_coin = None
                        pending_gold_fever_coin_ms = None
                        coin_rejection_reason = "gold_fever_monotonic_confirmation"
                    else:
                        pending_gold_fever_coin = coin
                        pending_gold_fever_coin_ms = packet.media_time_ms
                        coin_rejection_reason = "gold_fever_counter_awaiting_monotonic_confirmation"
                        coin = None
                elif (
                    last_observed_coin is not None
                    and not is_plausible_coin_increase(coin, last_observed_coin)
                ):
                    # At the native 2 fps sampling rate, legitimate counter
                    # changes (including bags and chest payouts) have remained
                    # below this bound. Much larger jumps are joined HUD/sprite
                    # digits and must not poison the monotonic counter state.
                    coin_rejection_reason = "implausible_counter_jump_rejected"
                    coin = None
                if coin is not None and last_observed_coin is not None and coin > last_observed_coin and (
                    has_attached_digit_ocr_artifact(coin, last_observed_coin)
                    or is_leading_digit_ocr_substitution(coin, last_observed_coin)
                ):
                    # A sprite/damage digit immediately left of the HUD can be
                    # joined to an otherwise unchanged Gold value (237→1237,
                    # 504→1504). Retain the confirmed suffix, but preserve the
                    # raw OCR text and correction reason in evidence.
                    coin = last_observed_coin
                    coin_rejection_reason = "leading_digit_ocr_artifact_removed_from_confirmed_counter"
                if coin is None:
                    pass
                elif last_observed_coin is None or coin > last_observed_coin:
                    if pending_coin == coin and pending_coin_ms is not None and packet.media_time_ms - pending_coin_ms <= 2_000:
                        last_observed_coin = coin
                        pending_coin = None
                        pending_coin_ms = None
                    else:
                        pending_coin = coin
                        pending_coin_ms = packet.media_time_ms
                        coin_rejection_reason = "counter_change_awaiting_repeat_confirmation"
                        coin = None
                elif coin < last_observed_coin:
                    coin_rejection_reason = "counter_decrease_rejected"
                    coin = None
                else:
                    pending_coin = None
                    pending_coin_ms = None
                    pending_gold_fever_coin = None
                    pending_gold_fever_coin_ms = None
            elif not gold_fever_active:
                pending_gold_fever_coin = None
                pending_gold_fever_coin_ms = None
            hp_width, hp_confidence = measure_health_bar_width(frame, hp_config)
            for code, value, confidence, unit, raw in (
                ("coin_counter", coin, coin_confidence, "gold_coins", coin_raw),
                ("player_health_bar_width", hp_width, hp_confidence, "pixels_at_1440p", None),
            ):
                observed = value is not None
                observations.append(SignalObservation(
                    observation_id=f"vss_{code}_{packet.frame_number}",
                    video_asset_id=dataset["video_asset_id"],
                    session_id=dataset["session_id"],
                    detector_name="video4_direct_telemetry",
                    detector_version=DETECTOR_VERSION,
                    observable_code=code,
                    time_lower_ms=packet.media_time_ms,
                    time_upper_ms=packet.media_time_ms,
                    temporal_precision=TemporalPrecision.FRAME,
                    visibility=Visibility.VISIBLE if observed else Visibility.UNKNOWN,
                    evidence_grade=EvidenceGrade.B if observed else EvidenceGrade.UNRESOLVED,
                    observed=observed,
                    source_artifact=(
                        config["detectors"]["gem_xp"]["script"]
                        if code.startswith("player_")
                        else "VSS frame work/src/vss_framework/telemetry_scan.py"
                    ),
                    source_record_key=f"frame:{packet.frame_number}",
                    numeric_value=value,
                    unit=unit,
                    frame_number=packet.frame_number,
                    attributes={
                        "confidence": confidence,
                        "raw_ocr_text": raw,
                        "rejection_reason": coin_rejection_reason if code == "coin_counter" else None,
                        "excluded_from_gameplay": False,
                        "screen_state": pause["phase"],
                        "gameplay_segment_id": gameplay_segment_id,
                        "imputed": False,
                        "health_value_calibrated": False if code.startswith("player_") else None,
                    },
                ))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "telemetry_observations.jsonl"
    write_jsonl(output_path, (item.to_dict() for item in observations))
    counts = Counter(item.observable_code for item in observations if item.observed)
    identity = f"{DETECTOR_VERSION}|{dataset['video']['sha256']}|{start_second}|{max_seconds}|{sample_fps}"
    manifest = {
        "artifact_type": "vss_framework_direct_telemetry_run",
        "framework_version": "0.8.0",
        "processing_run_id": f"run_telemetry_{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "Tahereh Fahi",
        "configuration": {"start_second": start_second, "end_second": max_seconds, "sample_fps": sample_fps},
        "source_integrity": {
            "video_sha256": dataset["video"]["sha256"],
            "health_bar_geometry_source": config["detectors"]["gem_xp"]["script"],
            "health_bar_geometry_source_sha256": sha256_file(xp_script),
            "detector_source_sha256": sha256_file(Path(__file__)),
            "gameplay_state_source_sha256": sha256_file(Path(__file__).with_name("gameplay_state.py")),
        },
        "counts": {"observations": len(observations), "observed_by_code": dict(sorted(counts.items())),
                   "level_up_pause_excluded_observations": sum(bool(item.attributes.get("excluded_from_gameplay")) for item in observations)},
        "outputs": {"telemetry_observations": {"path": output_path.name, "sha256": sha256_file(output_path), "row_count": len(observations)}},
        "policies": {"human_coded_ground_truth_used": False, "imputation_performed": False, "database_write_performed": False},
        "limitations": ["Health is a raw red-fill width, not a calibrated HP percentage.", "Missing OCR or health-bar detections remain missing.",
                        "Level-up transitions and menus are explicitly excluded; counter confirmation does not span those samples."],
        "metadata_verification": {"prepared_by": "Tahereh Fahi"},
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
