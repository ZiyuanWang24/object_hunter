"""Write the living-room MJCF and launch MuJoCo's native interactive viewer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from living_room import ROOM_XML


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    xml_path = repo / "assets" / "living_room.xml"
    xml_path.write_text(ROOM_XML.strip() + "\n", encoding="utf-8")
    print(f"Wrote living-room model: {xml_path}")
    raise SystemExit(subprocess.call([sys.executable, "-m", "mujoco.viewer", "--mjcf", str(xml_path)], cwd=repo))


if __name__ == "__main__":
    main()
