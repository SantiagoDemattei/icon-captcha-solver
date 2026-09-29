import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.evaluation import evaluate_solver

if __name__ == "__main__":
    print("\n".join(evaluate_solver().lines()))
