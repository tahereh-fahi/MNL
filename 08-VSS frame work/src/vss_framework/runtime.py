"""Distribution identity helpers. Author: Tahereh Fahi."""
from importlib import metadata


def distribution_version(name):
    """Keep legacy receipt keys while supporting the headless OpenCV wheel."""
    candidates = ("opencv-python-headless", "opencv-python") if name == "opencv-python" else (name,)
    for candidate in candidates:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            pass
    raise metadata.PackageNotFoundError(name)
