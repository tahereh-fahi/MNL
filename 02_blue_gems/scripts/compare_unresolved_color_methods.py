#!/usr/bin/env python3
"""Compare candidate blue/green/red assignments for unresolved XP events.

The script never changes the detector's official event table. It learns from
single-gem, non-saturated events whose colors were resolved visually, evaluates
each candidate method with out-of-fold predictions, and writes an auditable set
of alternative predictions for the currently unresolved events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COLORS = ["blue", "green", "red"]
# Crown upgrades were read from Video 4's level-up Growth panel. The activation
# frame is the first gameplay frame after the selected level-up menu closes.
VIDEO4_CROWN_ACTIVATION_FRAMES = (5530, 6026, 7350, 10449, 14759)
VIDEO4_BASE_GROWTH_PERCENT = 15.0
VIDEO4_XP_PIXEL_CORRECTION = 0.994
DISPLAY_COLORS = {
    "blue": "#2589d8",
    "green": "#2ca25f",
    "red": "#d73027",
    "unresolved": "#d89b17",
}
VISUAL_REFERENCE_EVIDENCE = {
    "visual_trajectory",
    "visual_disappearance",
    "visual_disappearance_weak",
}
VERSION_NAMES = {
    "current": "Current conservative",
    "physics": "XP physics bands",
    "knn": "Nearest resolved examples",
    "multimodal": "XP + weak visual model",
    "consensus": "High-confidence consensus",
}


def apply_video4_crown_growth(events: pd.DataFrame) -> pd.DataFrame:
    """Add audited Crown/Growth features and recompute base XP for Video 4."""
    adjusted = events.copy()
    frame_b = pd.to_numeric(adjusted["frame_b"], errors="coerce").fillna(-1)
    level = pd.to_numeric(adjusted["hud_level"], errors="coerce").fillna(1).astype(int)

    crown_level = sum(
        frame_b.ge(activation_frame).astype(int)
        for activation_frame in VIDEO4_CROWN_ACTIVATION_FRAMES
    )
    imelda_growth = 10.0 * np.minimum(level // 5, 3)
    crown_growth = 8.0 * crown_level
    level_spike_growth = np.where(level.isin([20, 40]), 100.0, 0.0)
    total_growth = (
        VIDEO4_BASE_GROWTH_PERCENT
        + imelda_growth
        + crown_growth
        + level_spike_growth
    )
    multiplier = 1.0 + total_growth / 100.0
    required = pd.to_numeric(adjusted["xp_required_for_level"], errors="coerce")
    delta_fraction = pd.to_numeric(adjusted["xp_delta_fraction"], errors="coerce")
    crown_adjusted_gain = delta_fraction * required / multiplier

    adjusted["crown_level_before"] = crown_level.astype(int)
    adjusted["crown_growth_percent"] = crown_growth
    adjusted["imelda_growth_percent"] = imelda_growth
    adjusted["base_growth_powerup_percent"] = VIDEO4_BASE_GROWTH_PERCENT
    adjusted["level_spike_growth_percent"] = level_spike_growth
    adjusted["total_growth_percent"] = total_growth
    adjusted["growth_multiplier_crown_adjusted"] = multiplier
    adjusted["estimated_base_xp_gain_crown_adjusted"] = crown_adjusted_gain
    adjusted["xp_pixel_correction_crown_adjusted"] = VIDEO4_XP_PIXEL_CORRECTION
    adjusted["level_normalized_xp_gain_pre_crown_model"] = pd.to_numeric(
        adjusted["level_normalized_xp_gain"], errors="coerce"
    )
    adjusted["level_normalized_xp_gain"] = (
        crown_adjusted_gain / VIDEO4_XP_PIXEL_CORRECTION
    ).round(4)
    return adjusted


def parsed_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    default_root = (
        project
        / "results"
        / "video4_Imelda_100"
        / "collected_gems_cv_final"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=default_root / "collected_gems_video4_imelda_100_xp_ab_events.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "color_classification_comparison",
    )
    return parser.parse_args()


def event_label(frame: pd.DataFrame) -> pd.Series:
    counts = frame[
        ["collected_blue_gems", "collected_green_gems", "collected_red_gems"]
    ]
    return (
        counts.idxmax(axis=1)
        .str.removeprefix("collected_")
        .str.removesuffix("_gems")
    )


def visual_reference(events: pd.DataFrame) -> pd.DataFrame:
    reference = events.loc[
        events["collected_gems_total"].eq(1)
        & events["unresolved_collected_gems"].eq(0)
        & events["xp_bar_saturated"].eq(0)
        & events["color_evidence"].isin(VISUAL_REFERENCE_EVIDENCE)
    ].copy()
    reference["reference_color"] = event_label(reference)
    return reference


def color_occurrences(value: object, color: str) -> int:
    if pd.isna(value):
        return 0
    return sum(token == color for token in str(value).split("|"))


def feature_frame(events: pd.DataFrame, *, nearest_only: bool) -> pd.DataFrame:
    gain = pd.to_numeric(events["level_normalized_xp_gain"], errors="coerce")
    percent = pd.to_numeric(events["xp_bar_increase_percent"], errors="coerce")
    features = pd.DataFrame(index=events.index)
    features["hud_level"] = pd.to_numeric(events["hud_level"], errors="coerce")
    features["log_xp_increase_percent"] = np.log10(percent.clip(lower=0.001))
    if nearest_only:
        return features

    features["log_level_normalized_gain"] = np.log10(gain.clip(lower=0.001))
    for column in (
        "crown_level_before",
        "crown_growth_percent",
        "total_growth_percent",
    ):
        if column in events:
            features[column] = pd.to_numeric(events[column], errors="coerce")
    features["log_jump_step_ratio"] = np.log10(
        pd.to_numeric(events["xp_jump_step_ratio"], errors="coerce").clip(lower=0.001)
    )
    for column in (
        "hud_level_ocr_confidence",
        "count_trajectory_track_count",
        "count_trajectory_max_score",
        "count_trajectory_mean_score",
        "unmatched_candidate_blue",
        "unmatched_candidate_green",
        "unmatched_candidate_red",
        "disappeared_blue",
        "disappeared_green",
        "disappeared_red",
        "detections_a_blue",
        "detections_a_green",
        "detections_a_red",
        "detections_b_blue",
        "detections_b_green",
        "detections_b_red",
    ):
        features[column] = pd.to_numeric(events[column], errors="coerce")
    for color in COLORS:
        features[f"weak_track_{color}_count"] = events[
            "count_trajectory_colors"
        ].map(lambda value, name=color: color_occurrences(value, name))
    return features.replace([np.inf, -np.inf], np.nan)


def physics_prediction(events: pd.DataFrame) -> np.ndarray:
    gain = pd.to_numeric(events["level_normalized_xp_gain"], errors="coerce")
    predictions = np.full(len(events), "", dtype=object)
    usable = gain.notna() & gain.gt(0) & events["xp_bar_saturated"].eq(0)
    predictions[usable & gain.le(2.25)] = "blue"
    predictions[usable & gain.gt(2.25) & gain.le(9.50)] = "green"
    predictions[usable & gain.gt(9.50)] = "red"
    return predictions


def knn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=31, weights="distance", p=2),
            ),
        ]
    )


def multimodal_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=9,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=20260802,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(COLORS)), dtype=float)
    for source_index, label in enumerate(model_classes):
        aligned[:, COLORS.index(str(label))] = probabilities[:, source_index]
    return aligned


def red_gate(events: pd.DataFrame) -> np.ndarray:
    gain = pd.to_numeric(events["level_normalized_xp_gain"], errors="coerce")
    weak_red = events["count_trajectory_colors"].map(
        lambda value: color_occurrences(value, "red")
    )
    score = pd.to_numeric(events["count_trajectory_max_score"], errors="coerce")
    return (
        events["xp_bar_saturated"].eq(0)
        & gain.gt(9.50)
        & weak_red.gt(0)
        & score.ge(0.76)
    ).to_numpy()


def level_up_visual_override(
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve saturated level-up pickups only from one strong visual track."""
    track_count = pd.to_numeric(
        events["count_trajectory_track_count"], errors="coerce"
    ).fillna(0)
    color = events["count_trajectory_colors"].fillna("").astype(str)
    max_score = pd.to_numeric(
        events["count_trajectory_max_score"], errors="coerce"
    ).fillna(0.0)
    mean_score = pd.to_numeric(
        events["count_trajectory_mean_score"], errors="coerce"
    ).fillna(0.0)
    disappearance = pd.to_numeric(
        events["count_trajectory_disappearance_frames"], errors="coerce"
    )
    end_frame = pd.to_numeric(
        events["count_trajectory_end_frames"], errors="coerce"
    )
    frame_a = pd.to_numeric(events["frame_a"], errors="coerce")
    frame_b = pd.to_numeric(events["frame_b"], errors="coerce")

    timing_supported = (
        disappearance.ge(frame_a - 1)
        & disappearance.le(frame_b)
        & end_frame.ge(frame_a - 4)
        & end_frame.le(disappearance)
    )
    eligible = (
        events["xp_bar_saturated"].eq(1)
        & track_count.eq(1)
        & color.isin(COLORS)
        & max_score.ge(0.90)
        & mean_score.ge(0.90)
        & timing_supported
    )
    return eligible.to_numpy(), color.to_numpy(dtype=object), max_score.to_numpy()


def saturated_likely_color(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give unresolved saturated pickups a visual-only likely color.

    This is intentionally separate from an exact count assignment. Trajectory scores vote
    by color; if no trajectory exists, unmatched A/B candidates provide the fallback vote.
    """
    likely = np.full(len(events), "", dtype=object)
    confidence = np.zeros(len(events), dtype=float)
    reason = np.full(len(events), "", dtype=object)

    for position, (_, row) in enumerate(events.iterrows()):
        if int(row.get("xp_bar_saturated", 0)) != 1:
            continue
        colors = [
            token for token in str(row.get("count_trajectory_colors", "")).split("|")
            if token in COLORS
        ]
        raw_scores = [
            token for token in str(row.get("count_trajectory_scores", "")).split("|")
            if token and token.lower() != "nan"
        ]
        scores = []
        for token in raw_scores:
            try:
                scores.append(float(token))
            except ValueError:
                scores.append(0.0)
        votes = {color: 0.0 for color in COLORS}
        for index, color in enumerate(colors):
            votes[color] += scores[index] if index < len(scores) else 1.0
        vote_source = "trajectory_score_vote"

        if sum(votes.values()) == 0:
            for color in COLORS:
                value = pd.to_numeric(
                    pd.Series([row.get(f"unmatched_candidate_{color}", 0)]),
                    errors="coerce",
                ).fillna(0).iloc[0]
                votes[color] = float(value)
            vote_source = "unmatched_candidate_vote"

        total = sum(votes.values())
        if total <= 0:
            # A pickup is confirmed by the level change; use the weak visual classifier as
            # a last-resort color ranking, never as an exact-count resolution.
            vote_source = "weak_visual_fallback"
            likely[position] = "blue"
            confidence[position] = 0.0
            reason[position] = vote_source
            continue
        winner = max(COLORS, key=lambda color: votes[color])
        likely[position] = winner
        confidence[position] = votes[winner] / total
        reason[position] = vote_source
    return likely, confidence, reason


def multimodal_predict(
    model: Pipeline,
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = align_probabilities(
        model.predict_proba(feature_frame(events, nearest_only=False)),
        model.named_steps["classifier"].classes_,
    )
    predictions = np.asarray(COLORS, dtype=object)[probabilities.argmax(axis=1)]
    gates = red_gate(events)
    predictions[gates] = "red"
    red_score = pd.to_numeric(
        events["count_trajectory_max_score"], errors="coerce"
    ).fillna(0.0)
    probabilities[gates] = 0.0
    probabilities[gates, COLORS.index("red")] = np.clip(
        0.65 + 0.35 * red_score[gates].to_numpy(), 0.0, 0.99
    )
    saturated = events["xp_bar_saturated"].eq(1).to_numpy()
    predictions[saturated] = ""
    probabilities[saturated] = 0.0

    visual_override, visual_color, visual_score = level_up_visual_override(events)
    for color_index, color in enumerate(COLORS):
        selected = visual_override & (visual_color == color)
        if not selected.any():
            continue
        score = np.clip(visual_score[selected], 0.0, 0.99)
        probabilities[selected] = ((1.0 - score) / (len(COLORS) - 1))[:, None]
        probabilities[selected, color_index] = score
        predictions[selected] = color
    return predictions, probabilities


def out_of_fold_predictions(
    reference: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    labels = reference["reference_color"].to_numpy()
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260802)
    results = {
        "knn": {
            "prediction": np.full(len(reference), "", dtype=object),
            "probability": np.zeros((len(reference), len(COLORS))),
        },
        "multimodal": {
            "prediction": np.full(len(reference), "", dtype=object),
            "probability": np.zeros((len(reference), len(COLORS))),
        },
    }
    nearest_features = feature_frame(reference, nearest_only=True)
    full_features = feature_frame(reference, nearest_only=False)
    for train_indices, test_indices in splitter.split(nearest_features, labels):
        knn = knn_pipeline()
        knn.fit(nearest_features.iloc[train_indices], labels[train_indices])
        knn_probabilities = align_probabilities(
            knn.predict_proba(nearest_features.iloc[test_indices]),
            knn.named_steps["classifier"].classes_,
        )
        results["knn"]["probability"][test_indices] = knn_probabilities
        results["knn"]["prediction"][test_indices] = np.asarray(
            COLORS, dtype=object
        )[knn_probabilities.argmax(axis=1)]

        multimodal = multimodal_pipeline()
        non_red_train = train_indices[labels[train_indices] != "red"]
        multimodal.fit(full_features.iloc[non_red_train], labels[non_red_train])
        fold_predictions, fold_probabilities = multimodal_predict(
            multimodal,
            reference.iloc[test_indices],
        )
        results["multimodal"]["prediction"][test_indices] = fold_predictions
        results["multimodal"]["probability"][test_indices] = fold_probabilities
    return results


def choose_consensus_threshold(
    truth: np.ndarray,
    physics: np.ndarray,
    knn: np.ndarray,
    multimodal: np.ndarray,
    multimodal_probability: np.ndarray,
    reference: pd.DataFrame,
) -> float:
    confidence = multimodal_probability.max(axis=1)
    agreement = (physics == knn) & (physics == multimodal) & (physics != "")
    eligibility = (
        agreement
        & reference["hud_level_ocr_accepted"].eq(1).to_numpy()
        & reference["xp_event_temporally_isolated"].eq(1).to_numpy()
        & reference["xp_bar_saturated"].eq(0).to_numpy()
    )
    candidates: list[tuple[int, float]] = []
    for threshold in np.arange(0.50, 0.991, 0.01):
        selected = eligibility & (confidence >= threshold)
        count = int(selected.sum())
        if count < 20:
            continue
        precision = float((multimodal[selected] == truth[selected]).mean())
        if precision >= 0.95:
            candidates.append((count, float(threshold)))
    return max(candidates, default=(0, 0.90))[1]


def consensus_prediction(
    events: pd.DataFrame,
    physics: np.ndarray,
    knn: np.ndarray,
    multimodal: np.ndarray,
    multimodal_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    confidence = multimodal_probability.max(axis=1)
    selected = (
        (physics == knn)
        & (physics == multimodal)
        & (physics != "")
        & (confidence >= threshold)
        & events["hud_level_ocr_accepted"].eq(1).to_numpy()
        & events["xp_event_temporally_isolated"].eq(1).to_numpy()
        & events["xp_bar_saturated"].eq(0).to_numpy()
    )
    predictions = np.full(len(events), "", dtype=object)
    predictions[selected] = multimodal[selected]
    return predictions, confidence


def metric_record(
    version: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    assigned = prediction != ""
    assigned_count = int(assigned.sum())
    record: dict[str, object] = {
        "version": version,
        "version_name": VERSION_NAMES[version],
        "reference_events": len(truth),
        "reference_assigned": assigned_count,
        "reference_coverage": round(assigned_count / len(truth), 4),
        "reference_accuracy_assigned": math.nan,
        "reference_balanced_accuracy_assigned": math.nan,
        "reference_macro_f1_assigned": math.nan,
    }
    for color in COLORS:
        record[f"{color}_precision"] = math.nan
        record[f"{color}_recall"] = math.nan
        record[f"{color}_f1"] = math.nan
        record[f"{color}_support"] = int((truth == color).sum())
    if not assigned_count:
        return record
    selected_truth = truth[assigned]
    selected_prediction = prediction[assigned]
    record["reference_accuracy_assigned"] = round(
        float(accuracy_score(selected_truth, selected_prediction)), 4
    )
    record["reference_balanced_accuracy_assigned"] = round(
        float(balanced_accuracy_score(selected_truth, selected_prediction)), 4
    )
    record["reference_macro_f1_assigned"] = round(
        float(
            f1_score(
                selected_truth,
                selected_prediction,
                labels=COLORS,
                average="macro",
                zero_division=0,
            )
        ),
        4,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        selected_truth,
        selected_prediction,
        labels=COLORS,
        zero_division=0,
    )
    for index, color in enumerate(COLORS):
        record[f"{color}_precision"] = round(float(precision[index]), 4)
        record[f"{color}_recall"] = round(float(recall[index]), 4)
        record[f"{color}_f1"] = round(float(f1[index]), 4)
        record[f"{color}_support"] = int(support[index])
    return record


def add_unresolved_outcome(
    record: dict[str, object],
    prediction: np.ndarray,
    unresolved_count: int,
) -> None:
    assigned = prediction != ""
    record["unresolved_input"] = unresolved_count
    record["unresolved_assigned"] = int(assigned.sum())
    for color in COLORS:
        record[f"unresolved_predicted_{color}"] = int((prediction == color).sum())
    record["unresolved_remaining"] = int(unresolved_count - assigned.sum())


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(
        output_dir / f"{name}.jpg",
        dpi=200,
        bbox_inches="tight",
        pil_kwargs={"quality": 94},
    )
    plt.close(fig)


def plot_outcomes(comparison: pd.DataFrame, output_dir: Path) -> None:
    display = comparison.set_index("version_name")
    columns = [
        "unresolved_predicted_blue",
        "unresolved_predicted_green",
        "unresolved_predicted_red",
        "unresolved_remaining",
    ]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    left = np.zeros(len(display))
    for column, label, color in zip(
        columns,
        ["Predicted blue", "Predicted green", "Predicted red", "Still unresolved"],
        [
            DISPLAY_COLORS["blue"],
            DISPLAY_COLORS["green"],
            DISPLAY_COLORS["red"],
            DISPLAY_COLORS["unresolved"],
        ],
    ):
        values = display[column].to_numpy()
        bars = ax.barh(display.index, values, left=left, label=label, color=color)
        for bar, value in zip(bars, values):
            if value >= 20:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if column != "unresolved_remaining" else "#29323c",
                    fontweight="bold",
                )
        left += values
    ax.set(
        xlabel="Number of pickups",
        ylabel="",
        xlim=(0, int(display[columns].sum(axis=1).max()) * 1.01),
    )
    ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Alternative outcomes for the same 849 unresolved pickups",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(fig, output_dir, "unresolved_assignment_outcomes")


def plot_validation(comparison: pd.DataFrame, output_dir: Path) -> None:
    evaluated = comparison.loc[comparison["reference_assigned"].gt(0)].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    label_offsets = {
        "physics": (-10, -14),
        "knn": (-10, 8),
        "multimodal": (-10, 8),
        "consensus": (8, 5),
    }
    for _, row in evaluated.iterrows():
        x = 100 * float(row["reference_coverage"])
        y = 100 * float(row["reference_accuracy_assigned"])
        ax.scatter(x, y, s=90, color="#334e68")
        offset = label_offsets[str(row["version"])]
        ax.annotate(
            row["version_name"],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha="right" if offset[0] < 0 else "left",
        )
    ax.axhline(95, color="#d73027", linestyle="--", linewidth=1, label="95% target")
    ax.set(
        title="Held-out visual-reference accuracy versus coverage",
        xlabel="Reference events assigned (%)",
        ylabel="Accuracy among assigned events (%)",
        xlim=(0, 102),
        ylim=(50, 101),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output_dir, "validation_accuracy_vs_coverage")


def plot_version_scatter(
    reference: pd.DataFrame,
    unresolved: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    versions = ["current", "physics", "knn", "multimodal", "consensus"]
    fig, axes = plt.subplots(3, 2, figsize=(17, 15), sharex=True, sharey=True)
    axes = axes.ravel()
    reference_colors = reference["reference_color"].map(DISPLAY_COLORS)
    for axis, version in zip(axes, versions):
        axis.scatter(
            reference["hud_level"],
            reference["xp_bar_increase_percent"],
            c=reference_colors,
            s=8,
            alpha=0.16,
            linewidths=0,
        )
        candidate = predictions[version]
        colors = [
            DISPLAY_COLORS.get(value, DISPLAY_COLORS["unresolved"])
            for value in candidate
        ]
        axis.scatter(
            unresolved["hud_level"],
            unresolved["xp_bar_increase_percent"],
            c=colors,
            s=18,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.2,
        )
        counts = {color: int((candidate == color).sum()) for color in COLORS}
        remaining = int((candidate == "").sum())
        axis.set_title(
            f"{VERSION_NAMES[version]}\n"
            f"B {counts['blue']} | G {counts['green']} | R {counts['red']} | U {remaining}",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_yscale("log")
        axis.grid(alpha=0.15)
        axis.set_xlabel("Visible HUD level")
        axis.set_ylabel("XP-bar increase (%)")
    axes[-1].axis("off")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=DISPLAY_COLORS[color],
            markeredgecolor="none",
            label=color.title(),
        )
        for color in ["blue", "green", "red", "unresolved"]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    fig.suptitle(
        "How each version colors the same unresolved XP jumps",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, output_dir, "unresolved_color_versions_by_level")


def plot_confusions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    versions = ["physics", "knn", "multimodal", "consensus"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    labels = COLORS + ["unresolved"]
    for axis, version in zip(axes.ravel(), versions):
        predicted = np.where(predictions[version] == "", "unresolved", predictions[version])
        matrix = confusion_matrix(truth, predicted, labels=labels)
        shown = matrix[:3, :]
        image = axis.imshow(shown, cmap="Blues")
        for row in range(shown.shape[0]):
            for column in range(shown.shape[1]):
                axis.text(
                    column,
                    row,
                    str(int(shown[row, column])),
                    ha="center",
                    va="center",
                    color="white" if shown[row, column] > shown.max() / 2 else "#18212b",
                )
        axis.set(
            title=VERSION_NAMES[version],
            xticks=range(len(labels)),
            xticklabels=[label.title() for label in labels],
            yticks=range(3),
            yticklabels=[label.title() for label in COLORS],
            xlabel="Predicted",
            ylabel="Visual reference",
        )
        axis.tick_params(axis="x", rotation=25)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Out-of-fold confusion matrices", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_dir, "out_of_fold_confusion_matrices")


def main() -> None:
    args = parsed_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    reference = visual_reference(events)
    unresolved = events.loc[events["unresolved_collected_gems"].gt(0)].copy()
    truth = reference["reference_color"].to_numpy()

    oof = out_of_fold_predictions(reference)
    reference_physics = physics_prediction(reference)
    threshold = choose_consensus_threshold(
        truth,
        reference_physics,
        oof["knn"]["prediction"],
        oof["multimodal"]["prediction"],
        oof["multimodal"]["probability"],
        reference,
    )
    reference_consensus, _ = consensus_prediction(
        reference,
        reference_physics,
        oof["knn"]["prediction"],
        oof["multimodal"]["prediction"],
        oof["multimodal"]["probability"],
        threshold,
    )
    reference_predictions = {
        "current": np.full(len(reference), "", dtype=object),
        "physics": reference_physics,
        "knn": oof["knn"]["prediction"],
        "multimodal": oof["multimodal"]["prediction"],
        "consensus": reference_consensus,
    }

    nearest_model = knn_pipeline()
    nearest_model.fit(
        feature_frame(reference, nearest_only=True),
        truth,
    )
    unresolved_knn_probability = align_probabilities(
        nearest_model.predict_proba(feature_frame(unresolved, nearest_only=True)),
        nearest_model.named_steps["classifier"].classes_,
    )
    unresolved_knn = np.asarray(COLORS, dtype=object)[
        unresolved_knn_probability.argmax(axis=1)
    ]
    unresolved_knn[unresolved["xp_bar_saturated"].eq(1).to_numpy()] = ""

    multimodal_model = multimodal_pipeline()
    non_red_reference = reference["reference_color"].ne("red")
    multimodal_model.fit(
        feature_frame(reference.loc[non_red_reference], nearest_only=False),
        reference.loc[non_red_reference, "reference_color"],
    )
    unresolved_multimodal, unresolved_multimodal_probability = multimodal_predict(
        multimodal_model,
        unresolved,
    )
    unresolved_physics = physics_prediction(unresolved)
    unresolved_consensus, unresolved_consensus_confidence = consensus_prediction(
        unresolved,
        unresolved_physics,
        unresolved_knn,
        unresolved_multimodal,
        unresolved_multimodal_probability,
        threshold,
    )
    unresolved_predictions = {
        "current": np.full(len(unresolved), "", dtype=object),
        "physics": unresolved_physics,
        "knn": unresolved_knn,
        "multimodal": unresolved_multimodal,
        "consensus": unresolved_consensus,
    }

    records = []
    for version in VERSION_NAMES:
        record = metric_record(version, truth, reference_predictions[version])
        add_unresolved_outcome(
            record,
            unresolved_predictions[version],
            len(unresolved),
        )
        records.append(record)
    comparison = pd.DataFrame(records)
    comparison.to_csv(args.output_dir / "method_comparison.csv", index=False)

    audit_columns = [
        "event_id",
        "event_key",
        "frame_a",
        "frame_b",
        "video_time_stamp",
        "hud_level",
        "hud_level_source",
        "hud_level_ocr_confidence",
        "hud_level_ocr_accepted",
        "xp_bar_increase_percent",
        "level_normalized_xp_gain",
        "xp_bar_saturated",
        "xp_event_temporally_isolated",
        "count_trajectory_colors",
        "count_trajectory_track_count",
        "count_trajectory_max_score",
        "percentage_assist_gate_reason",
    ]
    audit = unresolved[audit_columns].copy()
    audit["physics_prediction"] = unresolved_physics
    audit["knn_prediction"] = unresolved_knn
    audit["knn_confidence"] = unresolved_knn_probability.max(axis=1).round(4)
    for index, color in enumerate(COLORS):
        audit[f"knn_probability_{color}"] = unresolved_knn_probability[:, index].round(4)
    audit["multimodal_prediction"] = unresolved_multimodal
    audit["multimodal_confidence"] = unresolved_multimodal_probability.max(axis=1).round(4)
    for index, color in enumerate(COLORS):
        audit[f"multimodal_probability_{color}"] = unresolved_multimodal_probability[:, index].round(4)
    audit["consensus_prediction"] = unresolved_consensus
    audit["consensus_confidence"] = unresolved_consensus_confidence.round(4)
    audit["consensus_threshold"] = round(threshold, 4)
    audit["consensus_status"] = np.where(
        unresolved_consensus == "", "remain_unresolved", "high_confidence_assignment"
    )
    audit.to_csv(args.output_dir / "unresolved_event_predictions.csv", index=False)

    class_balance = (
        reference["reference_color"]
        .value_counts()
        .reindex(COLORS, fill_value=0)
        .rename_axis("color")
        .reset_index(name="visual_reference_events")
    )
    class_balance.to_csv(args.output_dir / "visual_reference_class_balance.csv", index=False)

    plot_outcomes(comparison, args.output_dir)
    plot_validation(comparison, args.output_dir)
    plot_version_scatter(reference, unresolved, unresolved_predictions, args.output_dir)
    plot_confusions(truth, reference_predictions, args.output_dir)

    report = {
        "method": "exploratory_unresolved_color_comparison",
        "official_detector_outputs_modified": False,
        "human_coded_sheet_used": False,
        "visual_reference_events": len(reference),
        "visual_reference_class_counts": {
            color: int((truth == color).sum()) for color in COLORS
        },
        "unresolved_input_events": len(unresolved),
        "cross_validation": "3-fold stratified out-of-fold",
        "consensus_confidence_threshold": round(threshold, 4),
        "warnings": [
            "Visual detector labels are pseudo-ground truth, not independent human labels.",
            "Only three clean red reference events exist; red metrics are not statistically stable.",
            "Saturated XP jumps remain unresolved because their visible percentage is right-censored.",
            "Alternative assignments are exploratory and do not overwrite official counts.",
        ],
        "recommended_decision_rule": (
            "Prefer high-confidence consensus when precision matters; use multimodal predictions "
            "as a review queue rather than automatic truth for the remaining events."
        ),
    }
    (args.output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(comparison.to_string(index=False))
    print(f"\nWrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
