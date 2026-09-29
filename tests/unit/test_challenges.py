import io
import json

import numpy as np
from PIL import Image

from icon_solver.challenges import HUMAN, MODEL, Challenge, ChallengeStore, is_validation, split_manifest


def _image_bytes(mode: str, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 6, len(mode)), 200, dtype=np.uint8), mode).save(buffer, fmt)
    return buffer.getvalue()


LEGEND = _image_bytes("RGBA", "PNG")
BACKGROUND = _image_bytes("RGB", "WEBP")
POINTS = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def _challenge(challenge_id: str, labeler: str | None) -> Challenge:
    return Challenge(challenge_id, LEGEND, BACKGROUND, POINTS, labeler)


def _val_ids(n: int) -> list[str]:
    return [f"c{i}" for i in range(200) if is_validation(f"c{i}", HUMAN)][:n]


def test_human_round_trip(tmp_path):
    store = ChallengeStore(tmp_path)
    directory = store.save(_challenge("a", HUMAN))

    assert sorted(p.name for p in directory.iterdir()) == ["background.webp", "legend.png", "points.json"]
    loaded = store.load("a")
    assert (loaded.legend_bytes, loaded.background_bytes, loaded.points, loaded.labeler) == (LEGEND, BACKGROUND, POINTS, HUMAN)
    assert loaded.legend.shape == (4, 6, 4)
    assert loaded.background.shape == (4, 6, 3)


def test_model_labeled_round_trip(tmp_path):
    store = ChallengeStore(tmp_path)
    directory = store.save(_challenge("b", MODEL), {"verified": True})

    assert json.loads((directory / "model_label.json").read_text()) == {"verified": True}
    assert store.load("b").labeler == MODEL


def test_rejected_round_trip(tmp_path):
    store = ChallengeStore(tmp_path)
    directory = store.save(_challenge("c", None), {"verified": False})

    assert json.loads((directory / "result.json").read_text()) == {"verified": False}
    loaded = store.load("c")
    assert loaded.labeler is None
    assert loaded.points == POINTS


def test_validation_rule_is_the_sha1_hash_and_never_model_labeled():
    assert _val_ids(3), "la regla deberia mandar alguno de 200 ids a validacion"
    for challenge_id in _val_ids(3):
        assert is_validation(challenge_id, HUMAN)
        assert not is_validation(challenge_id, MODEL)
    assert 0.15 < sum(is_validation(f"c{i}", HUMAN) for i in range(2000)) / 2000 < 0.25


def test_store_split_never_puts_model_labeled_in_validation(tmp_path):
    store = ChallengeStore(tmp_path)
    human_val, model_val = _val_ids(2)
    store.save(_challenge(human_val, HUMAN))
    store.save(_challenge(model_val, MODEL), {"verified": True})

    assert [c.id for c in store.iter("val")] == [human_val]
    assert [c.id for c in store.iter("train")] == [model_val]
    assert len(list(store)) == 2


def test_split_manifest_drops_negatives_of_validation_challenges():
    val_id = _val_ids(1)[0]
    manifest = [
        {"example": val_id, "labeler": HUMAN, "source": "real"},
        {"example": val_id, "labeler": HUMAN, "source": "negative"},
        {"example": val_id, "labeler": MODEL, "source": "real"},
    ]
    assert split_manifest(manifest) == ([2], [0])
