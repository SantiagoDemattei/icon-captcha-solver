import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.evaluation import evaluate_classifier
from icon_solver.model import IconModel
from icon_solver.solver import DEFAULT_MODEL

if __name__ == "__main__":
    print("\n".join(evaluate_classifier(IconModel.load(DEFAULT_MODEL)).lines()))
