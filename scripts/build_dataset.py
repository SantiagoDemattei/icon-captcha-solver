import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.dataset.build import build_dataset

if __name__ == "__main__":
    build_dataset()
