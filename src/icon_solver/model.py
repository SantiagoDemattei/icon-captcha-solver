from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from icon_solver.legend import IconPrototypes, prototype_mask
from icon_solver.segmentation import Region, legend_icons
from icon_solver.shapes import encode_regions

FORMAT_VERSION = 1


class _PolarConv(nn.Module):
    # En la entrada polar el eje de filas es el angulo: la fila 0 y la ultima son vecinas. Con
    # padding circular en ese eje (y ceros en el radial) la conv es equivariante a la rotacion
    # exacta, y el pooling global del final la vuelve invariante por construccion en vez de
    # tener que aprenderlo solo por augmentacion.
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 1, 1), mode="circular")
        x = F.pad(x, (1, 1, 0, 0))
        return self.conv(x)


class IconClassifier(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            _PolarConv(1, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            _PolarConv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            _PolarConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_outputs)

    @property
    def num_outputs(self) -> int:
        return self.classifier.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(model(x), dim=1)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IconModel:
    # Todo lo que la inferencia necesita en un solo artefacto: si los prototipos vivieran aparte y
    # el registro de clases creciera despues de entrenar, la clase nueva leeria en silencio la
    # salida "fondo" del clasificador.
    def __init__(self, classifier: IconClassifier, prototypes: IconPrototypes, num_classes: int):
        if prototypes.num_classes != num_classes:
            raise ValueError(f"los prototipos tienen {prototypes.num_classes} clases y el modelo {num_classes}")
        if classifier.num_outputs not in (num_classes, num_classes + 1):
            raise ValueError(f"el clasificador tiene {classifier.num_outputs} salidas para {num_classes} clases")
        self.classifier = classifier
        self.prototypes = prototypes
        self.num_classes = num_classes

    @property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    @property
    def num_outputs(self) -> int:
        return self.classifier.num_outputs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "num_classes": self.num_classes,
                "num_outputs": self.num_outputs,
                "state_dict": self.classifier.state_dict(),
                "prototype_masks": torch.from_numpy(np.stack(self.prototypes.masks)),
                "prototype_classes": torch.tensor(self.prototypes.classes, dtype=torch.int64),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "IconModel":
        device = device or default_device()
        data = torch.load(path, map_location=device, weights_only=True)
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: formato de modelo {data.get('format_version')!r}, se esperaba {FORMAT_VERSION}")
        classifier = IconClassifier(data["num_outputs"])
        classifier.load_state_dict(data["state_dict"])
        classifier.to(device).eval()
        prototypes = IconPrototypes(list(data["prototype_masks"].cpu().numpy()), data["prototype_classes"].tolist())
        return cls(classifier, prototypes, data["num_classes"])

    def legend_classes(self, legend_bgra: np.ndarray, icons: Sequence[Region] | None = None) -> list[int]:
        icons = legend_icons(legend_bgra) if icons is None else icons
        return [self.prototypes.classify(prototype_mask(legend_bgra, icon.contour))[0] for icon in icons]

    def class_probs(self, regions: Sequence[Region]) -> np.ndarray:
        x = encode_regions(regions)
        with torch.no_grad():
            return predict_proba(self.classifier, x.to(self.device)).cpu().numpy()
