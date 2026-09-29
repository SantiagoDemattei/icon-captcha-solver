import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from icon_solver.challenges import decode_bgr, decode_bgra
from icon_solver.model import IconModel
from icon_solver.paths import MODELS_DIR
from icon_solver.segmentation import DEFAULT_CONFIG, Region, SegmentationConfig, background_candidates, legend_icons
from icon_solver.shapes import legend_silhouettes, region_silhouettes, silhouette_similarity

DEFAULT_MODEL = MODELS_DIR / "icon_model.pt"
TOP_K = 8
TEMPLATE_WEIGHT = 1.0
SAME_ICON_DISTANCE = 8


@dataclass(frozen=True)
class SolveTrace:
    # Vacio si ninguna combinacion del shortlist tiene candidatos distintos para todos los iconos.
    points: list[dict]
    legend_icons: list[Region]
    legend_classes: list[int]
    candidates: list[Region]
    # Una fila por candidato, una columna por icono de la leyenda.
    scores: np.ndarray


class IconSolver:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        template_weight: float = TEMPLATE_WEIGHT,
        device: torch.device | None = None,
        config: SegmentationConfig = DEFAULT_CONFIG,
    ):
        self.model = IconModel.load(model_path, device)
        self.template_weight = template_weight
        self.config = config

    def solve(self, background_bgr: np.ndarray, legend_bgra: np.ndarray) -> list[dict]:
        trace = self.trace(background_bgr, legend_bgra)
        if len(trace.points) != len(trace.legend_icons):
            raise ValueError(f"no hay {len(trace.legend_icons)} candidatos distintos para los iconos de la leyenda")
        return trace.points

    def trace(self, background_bgr: np.ndarray, legend_bgra: np.ndarray) -> SolveTrace:
        icons = legend_icons(legend_bgra, self.config)
        classes = self.model.legend_classes(legend_bgra, icons)
        regions = background_candidates(background_bgr, self.config)
        scores = np.log(self.model.class_probs(regions)[:, classes] + 1e-9)
        if self.template_weight > 0:
            # Señal complementaria que no depende de lo aprendido: la silueta exacta del icono
            # pedido, sacada de la propia leyenda, contra cada candidato.
            similarity = silhouette_similarity(legend_silhouettes(legend_bgra, icons), region_silhouettes(regions))
            scores = scores + self.template_weight * np.log(similarity.T + 1e-3)
        points = [click_point(regions[i]) for i in _assign(regions, scores)]
        return SolveTrace(points, icons, classes, regions, scores)


def _assign(regions: list[Region], scores: np.ndarray) -> tuple[int, ...]:
    n_icons = scores.shape[1]

    # Cada icono de la leyenda es un blob distinto del fondo (una leyenda puede repetir el mismo
    # icono, y entonces son dos instancias distintas): asignacion uno-a-uno que maximiza la
    # probabilidad conjunta, por fuerza bruta sobre los mejores candidatos de cada icono.
    shortlist = sorted({int(i) for k in range(n_icons) for i in np.argsort(-scores[:, k])[:TOP_K]})
    centers = {i: np.array(regions[i].centroid) for i in shortlist}
    best: tuple[int, ...] = ()
    best_score = -np.inf
    for combo in itertools.permutations(shortlist, n_icons):
        # La siembra en grilla puede dar dos candidatos casi iguales del mismo icono (con
        # tolerancias distintas): distintos como candidatos, pero el mismo click.
        if any(np.hypot(*(centers[a] - centers[b])) < SAME_ICON_DISTANCE for a, b in itertools.combinations(combo, 2)):
            continue
        score = sum(scores[i, k] for k, i in enumerate(combo))
        if score > best_score:
            best, best_score = combo, score
    return best


def click_point(region: Region) -> dict:
    # El centroide aunque caiga fuera de la forma (Leo, el telefono): el server valida por distancia
    # al centro del icono con un umbral chico, no por caer adentro -- los humanos clickean ahi y se
    # acepta, y mover el click al interior de un icono concavo lo aleja del centro.
    cx, cy = region.centroid
    return {"x": int(round(cx)), "y": int(round(cy))}


_default_solver: IconSolver | None = None


def solve_icon(background_bytes: bytes, legend_bytes: bytes) -> list[dict]:
    global _default_solver
    if _default_solver is None:
        _default_solver = IconSolver()
    return _default_solver.solve(decode_bgr(background_bytes), decode_bgra(legend_bytes))
