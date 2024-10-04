from pathlib import Path
from typing import Final

ROOT_DIR: Final[Path] = Path(__file__).parent.parent
LOG_DIR: Final[Path] = ROOT_DIR / "logs"
