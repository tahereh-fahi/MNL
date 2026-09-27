"""Direct video detector interfaces and implementations."""

from .base import FrameDetector
from .screen_state import ScreenStateDetector
from .xp_bar import XPBarDetector
from .legacy_gem_xp import build_legacy_gem_command, run_legacy_gem_xp_detector
from .legacy_hud import build_legacy_hud_command, run_legacy_hud_detector
from .legacy_inventory import build_legacy_inventory_command, run_legacy_inventory_detector
from .instant_reward import classify_instant_reward_icon

__all__ = [
    "FrameDetector",
    "ScreenStateDetector",
    "XPBarDetector",
    "build_legacy_gem_command",
    "run_legacy_gem_xp_detector",
    "build_legacy_hud_command",
    "run_legacy_hud_detector",
    "build_legacy_inventory_command",
    "run_legacy_inventory_detector",
    "classify_instant_reward_icon",
]
