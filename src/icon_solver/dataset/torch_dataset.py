import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from icon_solver.shapes import read_mask, to_input, to_polar

from .build import MANIFEST_FILE


def _random_rotation(polar: np.ndarray) -> np.ndarray:
    # En polar, rotar el icono original es un shift circular exacto del eje angular (fila): sin
    # interpolacion ni aliasing.
    return np.roll(polar, random.randrange(polar.shape[0]), axis=0)


class IconMaskDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        augment: bool,
        indices: list[int] | None = None,
        num_outputs: int = 1,
    ):
        manifest = json.loads((processed_dir / MANIFEST_FILE).read_text())
        self.processed_dir = processed_dir
        self.entries = [manifest[i] for i in indices] if indices is not None else manifest
        self.augment = augment
        self.num_outputs = num_outputs

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        entry = self.entries[idx]
        polar = to_polar(read_mask(self.processed_dir / entry["crop"]))
        if self.augment:
            polar = _random_rotation(polar)
        # Un negativo (class_id -1) no tiene clase conocida: solo se sabe que no es ninguna de
        # las clases pedidas en su challenge, y eso es lo que codifica esta mascara.
        exclude = torch.zeros(self.num_outputs, dtype=torch.bool)
        exclude[entry.get("exclude", [])] = True
        return to_input(polar[None])[0], entry["class_id"], exclude
