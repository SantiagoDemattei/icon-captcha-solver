import json

import numpy as np
import pytest

from icon_solver.dataset.build import META_FILE, PROTOTYPES_FILE
from icon_solver.legend import PROTOTYPE_SIZE, IconPrototypes, save_prototypes
from icon_solver.training import train


def test_training_with_classes_other_than_the_prototypes_fails(tmp_path):
    masks = [np.full((PROTOTYPE_SIZE, PROTOTYPE_SIZE), c, dtype=np.uint8) for c in range(2)]
    save_prototypes(IconPrototypes(masks, [0, 1]), tmp_path / PROTOTYPES_FILE)
    (tmp_path / META_FILE).write_text(json.dumps({"num_classes": 3}))

    with pytest.raises(ValueError, match="prototipos"):
        train(epochs=1, processed_dir=tmp_path, out_path=tmp_path / "icon_model.pt")
    assert not (tmp_path / "icon_model.pt").exists()
