import importlib
import os
from pathlib import Path
from types import ModuleType


def _default_xlerobot_root() -> Path:
    return Path(__file__).resolve().parents[4] / "submodules" / "xlerobot"


def xlerobot_root() -> Path:
    override = os.environ.get("LEROBOT_XLEROBOT_PATH")
    return Path(override).expanduser().resolve() if override else _default_xlerobot_root()


def load_upstream_xlerobot() -> ModuleType:
    """Expose XLeRobot's source tree as ``lerobot.robots.xlerobot`` without copying it."""
    robots_path = xlerobot_root() / "software" / "src" / "robots"
    package_file = robots_path / "xlerobot" / "__init__.py"
    if not package_file.is_file():
        raise ImportError(
            "XLeRobot submodule is unavailable. Run: "
            "git submodule update --init submodules/xlerobot, or set LEROBOT_XLEROBOT_PATH."
        )

    import lerobot.robots as lerobot_robots

    path = str(robots_path)
    if path not in lerobot_robots.__path__:
        lerobot_robots.__path__.append(path)
    return importlib.import_module("lerobot.robots.xlerobot")


upstream = load_upstream_xlerobot()
XLerobot = upstream.XLerobot
XLerobotConfig = upstream.XLerobotConfig
