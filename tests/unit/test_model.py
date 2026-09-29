import numpy as np
import pytest
import torch

from icon_solver.legend import PROTOTYPE_SIZE, IconPrototypes
from icon_solver.model import IconClassifier, IconModel


def _prototypes(num_classes: int) -> IconPrototypes:
    masks = [np.full((PROTOTYPE_SIZE, PROTOTYPE_SIZE), c % 2, dtype=np.uint8) for c in range(num_classes)]
    return IconPrototypes(masks, list(range(num_classes)))


def test_save_and_load_with_weights_only(tmp_path):
    model = IconModel(IconClassifier(4), _prototypes(3), num_classes=3)
    path = tmp_path / "icon_model.pt"
    model.save(path)

    torch.load(path, weights_only=True)
    loaded = IconModel.load(path, torch.device("cpu"))

    assert (loaded.num_classes, loaded.num_outputs) == (3, 4)
    assert loaded.prototypes.classes == [0, 1, 2]
    assert all(np.array_equal(a, b) for a, b in zip(loaded.prototypes.masks, model.prototypes.masks))
    for a, b in zip(model.classifier.state_dict().values(), loaded.classifier.state_dict().values()):
        assert torch.equal(a, b)


def test_prototypes_with_other_class_count_fail():
    with pytest.raises(ValueError, match="prototipos"):
        IconModel(IconClassifier(4), _prototypes(4), num_classes=3)


def test_classifier_outputs_must_match_classes():
    with pytest.raises(ValueError, match="salidas"):
        IconModel(IconClassifier(6), _prototypes(3), num_classes=3)


def test_loading_a_mismatched_artifact_fails(tmp_path):
    path = tmp_path / "icon_model.pt"
    IconModel(IconClassifier(4), _prototypes(3), num_classes=3).save(path)
    data = torch.load(path, weights_only=True)
    data["num_classes"] = 2
    torch.save(data, path)

    with pytest.raises(ValueError):
        IconModel.load(path, torch.device("cpu"))
