import numpy as np

from icon_solver.legend import PROTOTYPE_SIZE, IconPrototypes, load_prototypes, save_prototypes


def _mask(seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random((PROTOTYPE_SIZE, PROTOTYPE_SIZE)) > 0.5).astype(np.uint8)


def test_prototypes_round_trip(tmp_path):
    prototypes = IconPrototypes()
    for seed in range(3):
        prototypes.match_or_add(_mask(seed))
    path = tmp_path / "prototypes.npz"

    save_prototypes(prototypes, path)
    loaded = load_prototypes(path)

    assert loaded.classes == prototypes.classes == [0, 1, 2]
    assert all(np.array_equal(a, b) for a, b in zip(loaded.masks, prototypes.masks))
    assert loaded.num_classes == 3


def test_nearest_prototype_or_new_class():
    prototypes = IconPrototypes()
    first = _mask(0)
    assert prototypes.match_or_add(first) == 0

    close = first.copy()
    close[0, :10] ^= 1
    assert prototypes.match_or_add(close) == 0
    assert prototypes.classify(close) == (0, 10)

    assert prototypes.match_or_add(_mask(1)) == 1
    assert len(prototypes.masks) == 2
