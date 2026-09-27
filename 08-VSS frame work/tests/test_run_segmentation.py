import csv

from vss_framework.run_segmentation import segment_runs


def test_detects_two_runs_and_preserves_media_time(tmp_path):
    source = tmp_path / "hud.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Video Second", "Time Stamp", "timer_observed"])
        writer.writeheader()
        writer.writerows([
            {"Video Second": "9", "Time Stamp": "00:00", "timer_observed": "True"},
            {"Video Second": "10", "Time Stamp": "00:01", "timer_observed": "True"},
            {"Video Second": "170", "Time Stamp": "02:21", "timer_observed": "True"},
            {"Video Second": "191", "Time Stamp": "00:00", "timer_observed": "True"},
            {"Video Second": "565", "Time Stamp": "05:14", "timer_observed": "True"},
        ])
    result = segment_runs(source, video_asset_id="video3", output_path=tmp_path / "segments.json")
    assert [(x["mediaStartMs"], x["mediaEndMs"]) for x in result["segments"]] == [
        (9000, 171000), (191000, 566000)
    ]
