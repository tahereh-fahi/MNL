"""Regression replay from real Video 4 artifacts, not an accuracy evaluation."""
from dataclasses import fields
from pathlib import Path
import unittest

import pandas as pd
from vss_framework.detectors.inventory import (
    MenuObservation, MenuSegment, build_inventory_events,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


class PregameInventoryRegression(unittest.TestCase):
    def test_real_pregame_candidate_does_not_increment_level(self):
        historical = WORKSPACE / (
            "08-VSS-external-reconstructed/runs/"
            "video4_inventory_full_20260912_162323/detector/worker")
        xp_path = ROOT / "runs/video4_xp_levels_20260912_203947/xp_level_events.csv"
        if not (historical / "level_up_menu_audit.csv").is_file() or not xp_path.is_file():
            self.skipTest("Real Video 4 regression artifacts are not installed")
        audit = pd.read_csv(historical / "level_up_menu_audit.csv").head(2)
        original = pd.read_csv(historical / "inventory_events.csv")
        initial = original.loc[original.event_source.eq("initial_state")].iloc[0]
        names = [field.name for field in fields(MenuObservation)]
        menus = [MenuSegment(row.menu_start_second, row.menu_end_second,
                 MenuObservation(**{name: getattr(row, name) for name in names}))
                 for row in audit.itertuples(index=False)]
        # Adapt the recorded initial detection to the builder's input schema.
        weapons = pd.DataFrame([{
            "weapon": initial.item_after, "occupied": True, "slot": initial.slot,
            "suggested_weapon": initial.suggested_item,
            "confidence": initial.confidence, "needs_review": initial.needs_review,
        }])
        kwargs = dict(video_path=Path("video4_Imelda_100.mp4"),
                      xp_events=pd.read_csv(xp_path), weapon_rows=weapons,
                      passive_rows=pd.DataFrame(), initial_second=float(initial.video_second),
                      initial_frame=int(initial.frame_number))
        with_pregame = build_inventory_events(menu_segments=menus, **kwargs)
        gameplay_only = build_inventory_events(menu_segments=menus[1:], **kwargs)
        pd.testing.assert_frame_equal(with_pregame, gameplay_only)
        self.assertEqual(with_pregame.character_level.tolist(), [1, 2])
        self.assertEqual(with_pregame.item_after.tolist(), ["Magic Wand", "King Bible"])
        self.assertEqual(len(with_pregame), 2)


if __name__ == "__main__":
    unittest.main()
