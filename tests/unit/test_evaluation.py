import numpy as np

from icon_solver.evaluation import HIT_RADIUS, accuracy, balanced_accuracy, is_hit
from icon_solver.segmentation import Region

TRUTH = {"x": 100, "y": 100}
REGION = Region(np.zeros((1, 1, 2), np.int32), (120.0, 100.0), 300.0)


def test_hit_within_radius_of_the_human_click():
    assert is_hit({"x": 100 + HIT_RADIUS, "y": 100}, TRUTH, None)
    assert is_hit({"x": 105, "y": 106}, TRUTH, None)
    assert not is_hit({"x": 100 + HIT_RADIUS + 1, "y": 100}, TRUTH, None)


def test_hit_within_radius_of_the_truth_region_centroid():
    assert is_hit({"x": 125, "y": 100}, TRUTH, REGION)
    assert not is_hit({"x": 110, "y": 100}, TRUTH, REGION)
    assert not is_hit({"x": 125, "y": 100}, TRUTH, None)


def test_accuracy_and_balanced_accuracy_with_background_column():
    # Filas: clase verdadera; columnas: prediccion, con la ultima columna = fondo.
    confusion = np.array(
        [
            [3, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ]
    )
    assert accuracy(confusion) == 4 / 6
    assert balanced_accuracy(confusion) == (3 / 4 + 1 / 2) / 2
    assert accuracy(np.zeros((2, 3), int)) == 0.0
    assert balanced_accuracy(np.zeros((2, 3), int)) == 0.0
