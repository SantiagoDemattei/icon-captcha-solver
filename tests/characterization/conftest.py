import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute  # noqa: E402

from icon_solver.paths import ROOT  # noqa: E402

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"
REQUIRED = [compute.RAW_DIR, compute.EXAMPLES_DIR, compute.PROCESSED_DIR / compute.PROTOTYPES_FILE, compute.ICON_MODEL_PATH]


@pytest.fixture(scope="session", autouse=True)
def local_data():
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        pytest.skip(f"caracterizacion sin datos locales: falta {', '.join(missing)}")


@pytest.fixture(scope="session")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="session")
def solver():
    from icon_solver.solver import IconSolver

    return IconSolver()


@pytest.fixture(scope="session")
def icon_model():
    return compute.icon_model()


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict:
    return compute.build(tmp_path_factory.mktemp("build") / "processed")
