from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from icon_solver.shapes import legend_alpha, square_pad

PROTOTYPE_SIZE = 32
NEW_ICON_DISTANCE = 100


def prototype_mask(legend_bgra: np.ndarray, contour: np.ndarray) -> np.ndarray:
    # Sin estirar: pad a cuadrado antes de reducir. Estirado, el aspect ratio se pierde y
    # iconos distintos (el trofeo y la bombilla, el casco y la "i") quedaban a menos distancia
    # que dos renders del mismo icono.
    square = square_pad(legend_alpha(legend_bgra, contour))
    return (cv2.resize(square, (PROTOTYPE_SIZE, PROTOTYPE_SIZE), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)


@dataclass
class IconPrototypes:
    # Varios prototipos por clase: el mismo icono aparece renderizado a unos pocos tamaños
    # distintos, y dos renders del mismo icono pueden diferir mas que dos iconos parecidos
    # entre si. Por eso se asigna por prototipo mas cercano y no por umbral contra una sola
    # plantilla (medido: 100% sobre los 966 iconos de leyenda; distancia maxima a su
    # prototipo 57, minima entre iconos distintos 137).
    masks: list[np.ndarray] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return max(self.classes) + 1 if self.classes else 0

    def classify(self, mask: np.ndarray) -> tuple[int, int]:
        if not self.masks:
            return -1, PROTOTYPE_SIZE * PROTOTYPE_SIZE
        distances = np.logical_xor(np.stack(self.masks), mask).sum(axis=(1, 2))
        nearest = int(np.argmin(distances))
        return self.classes[nearest], int(distances[nearest])

    def match_or_add(self, mask: np.ndarray, new_icon_distance: int = NEW_ICON_DISTANCE) -> int:
        class_id, distance = self.classify(mask)
        if distance < new_icon_distance:
            return class_id
        self.masks.append(mask)
        self.classes.append(self.num_classes)
        return self.classes[-1]


def save_prototypes(prototypes: IconPrototypes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, masks=np.stack(prototypes.masks), classes=np.array(prototypes.classes))


def load_prototypes(path: Path) -> IconPrototypes:
    data = np.load(path)
    return IconPrototypes(list(data["masks"]), [int(c) for c in data["classes"]])
