import hashlib
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

HUMAN = "human"
MODEL = "model"
VAL_SPLIT = 0.2
LEGEND_FILE = "legend.png"
BACKGROUND_FILE = "background.webp"
POINTS_FILE = "points.json"
MODEL_LABEL_FILE = "model_label.json"
RESULT_FILE = "result.json"

Split = Literal["all", "train", "val"]


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def decode_bgra(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def is_validation(challenge_id: str, labeler: str | None, val_split: float = VAL_SPLIT) -> bool:
    # Un challenge etiquetado por el modelo existe solo porque el modelo lo resolvio bien: en
    # validacion la llenaria de casos faciles y dejaria de detectar errores. Validacion = humanos.
    # Por hash del id y no por orden: sumar challenges nuevos no mueve los viejos de lado.
    if labeler == MODEL:
        return False
    digest = int(hashlib.sha1(challenge_id.encode()).hexdigest(), 16)
    return (digest % 10_000) < val_split * 10_000


@dataclass(frozen=True, eq=False)
class Challenge:
    id: str
    legend_bytes: bytes
    background_bytes: bytes
    points: list[dict] | None = None
    labeler: str | None = None

    @cached_property
    def legend(self) -> np.ndarray:
        return decode_bgra(self.legend_bytes)

    @cached_property
    def background(self) -> np.ndarray:
        return decode_bgr(self.background_bytes)


class ChallengeStore:
    # Formato en disco: una carpeta por challenge con legend.png, background.webp y points.json.
    # Los etiquetados por el modelo llevan ademas model_label.json con el veredicto del server; los
    # rechazados (en otra raiz, sin etiqueta) llevan result.json y en points.json el intento.
    def __init__(self, root: Path, val_split: float = VAL_SPLIT):
        self.root = root
        self.val_split = val_split

    def load(self, challenge_id: str) -> Challenge:
        directory = self.root / challenge_id
        points_path = directory / POINTS_FILE
        labeler: str | None
        if (directory / MODEL_LABEL_FILE).exists():
            labeler = MODEL
        elif (directory / RESULT_FILE).exists():
            labeler = None
        else:
            labeler = HUMAN
        return Challenge(
            id=challenge_id,
            legend_bytes=(directory / LEGEND_FILE).read_bytes(),
            background_bytes=(directory / BACKGROUND_FILE).read_bytes(),
            points=json.loads(points_path.read_text()) if points_path.exists() else None,
            labeler=labeler,
        )

    def __iter__(self) -> Iterator[Challenge]:
        return self.iter()

    def iter(self, split: Split = "all") -> Iterator[Challenge]:
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            challenge = self.load(directory.name)
            if split == "all" or (split == "val") == is_validation(challenge.id, challenge.labeler, self.val_split):
                yield challenge

    def save(self, challenge: Challenge, record: dict | None = None) -> Path:
        directory = self.root / challenge.id
        directory.mkdir(parents=True)
        (directory / LEGEND_FILE).write_bytes(challenge.legend_bytes)
        (directory / BACKGROUND_FILE).write_bytes(challenge.background_bytes)
        if challenge.points is not None:
            (directory / POINTS_FILE).write_text(json.dumps(challenge.points))
        if challenge.labeler == MODEL:
            (directory / MODEL_LABEL_FILE).write_text(json.dumps(record, indent=2))
        elif challenge.labeler is None:
            (directory / RESULT_FILE).write_text(json.dumps(record, indent=2))
        return directory


def split_manifest(manifest: list[dict], val_split: float = VAL_SPLIT) -> tuple[list[int], list[int]]:
    # Los iconos de un mismo challenge quedan siempre del mismo lado. Los negativos de un challenge
    # de validacion no van a ningun lado: entrenar con ellos filtraria ese fondo al modelo antes de
    # evaluar el solver sobre el.
    train_indices, val_indices = [], []
    for i, entry in enumerate(manifest):
        if not is_validation(entry["example"], entry.get("labeler", HUMAN), val_split):
            train_indices.append(i)
        elif entry.get("source", "real") == "real":
            val_indices.append(i)
    return train_indices, val_indices
