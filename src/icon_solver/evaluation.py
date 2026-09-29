import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from icon_solver.challenges import VAL_SPLIT, ChallengeStore, split_manifest
from icon_solver.dataset.build import MANIFEST_FILE
from icon_solver.dataset.torch_dataset import IconMaskDataset
from icon_solver.model import IconModel, predict_proba
from icon_solver.paths import PROCESSED_DIR, RAW_DIR
from icon_solver.segmentation import Region, truth_region
from icon_solver.solver import IconSolver, click_point

HIT_RADIUS = 8


def is_hit(predicted: dict, truth: dict, truth_region: Region | None) -> bool:
    # El server valida por cercania al centro del icono con un umbral chico: los clicks humanos
    # aceptados caen a <=8px del centroide de su region en el 95% de los casos. "Dentro de la forma"
    # no alcanza en iconos grandes o concavos, donde buena parte de la forma queda lejos del centro.
    if np.hypot(predicted["x"] - truth["x"], predicted["y"] - truth["y"]) <= HIT_RADIUS:
        return True
    if truth_region is None:
        return False
    cx, cy = truth_region.centroid
    return bool(np.hypot(predicted["x"] - cx, predicted["y"] - cy) <= HIT_RADIUS)


@dataclass(frozen=True)
class ChallengeResult:
    challenge_id: str
    predicted: list[dict]
    truth: list[dict]
    hits: list[bool]
    # Por punto de click: si algun candidato del solver habria acertado ese icono.
    candidate_found: list[bool]

    @property
    def solved(self) -> bool:
        return len(self.predicted) == len(self.truth) and all(self.hits)


@dataclass(frozen=True)
class SolverReport:
    results: list[ChallengeResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def solved_count(self) -> int:
        return sum(r.solved for r in self.results)

    @property
    def icon_total(self) -> int:
        return sum(len(r.truth) for r in self.results)

    @property
    def icon_hits(self) -> int:
        return sum(sum(r.hits) for r in self.results)

    @property
    def candidates_found(self) -> int:
        return sum(sum(r.candidate_found) for r in self.results)

    @property
    def solved(self) -> float:
        return self.solved_count / self.total

    @property
    def icons(self) -> float:
        return self.icon_hits / self.icon_total

    @property
    def candidates(self) -> float:
        return self.candidates_found / self.icon_total

    def lines(self) -> list[str]:
        return [
            f"challenges de validacion: {self.total}",
            f"resueltos completos (todo o nada): {self.solved_count}/{self.total} = {self.solved:.1%}",
            f"iconos acertados: {self.icon_hits}/{self.icon_total} = {self.icons:.1%}",
            f"icono correcto presente entre los candidatos: {self.candidates_found}/{self.icon_total} = {self.candidates:.1%}",
        ]


def evaluate_solver(store: ChallengeStore | None = None, solver: IconSolver | None = None) -> SolverReport:
    # Offline y end-to-end sobre los challenges de validacion: un challenge cuenta como resuelto
    # solo si todos sus puntos caen en el icono correcto, igual que /icon/verify (todo o nada, sin
    # credito parcial).
    store = store or ChallengeStore(RAW_DIR)
    solver = solver or IconSolver()
    results = []
    for challenge in store.iter("val"):
        truth = challenge.points or []
        trace = solver.trace(challenge.background, challenge.legend)
        regions = [truth_region(challenge.background, t, solver.config) for t in truth]
        complete = len(trace.points) == len(truth)
        hits = [complete and is_hit(p, t, r) for p, t, r in zip(trace.points, truth, regions)]
        clicks = [click_point(c) for c in trace.candidates]
        found = [any(is_hit(c, t, r) for c in clicks) for t, r in zip(truth, regions)]
        results.append(ChallengeResult(challenge.id, trace.points, truth, hits, found))
    return SolverReport(results)


def confusion_matrix(
    classifier: nn.Module, batches: Iterable, num_classes: int, num_outputs: int, device: torch.device
) -> np.ndarray:
    # Con clase "fondo" hay una columna de prediccion mas que filas de verdad (ningun icono real es
    # fondo): la ultima columna son los iconos que el modelo descarto como fondo.
    classifier.eval()
    confusion = np.zeros((num_classes, num_outputs), dtype=int)
    with torch.no_grad():
        for x, y, _ in batches:
            preds = predict_proba(classifier, x.to(device)).argmax(1).cpu().numpy()
            for true, pred in zip(y.numpy(), preds):
                confusion[true, pred] += 1
    return confusion


def accuracy(confusion: np.ndarray) -> float:
    return float(np.diag(confusion).sum() / max(1, confusion.sum()))


def balanced_accuracy(confusion: np.ndarray) -> float:
    per_class_total = confusion.sum(axis=1)
    present = per_class_total > 0
    if not present.any():
        return 0.0
    return float((np.diag(confusion)[present] / per_class_total[present]).mean())


@dataclass(frozen=True)
class ClassifierReport:
    confusion: np.ndarray

    @property
    def accuracy(self) -> float:
        return accuracy(self.confusion)

    @property
    def balanced_accuracy(self) -> float:
        return balanced_accuracy(self.confusion)

    def lines(self) -> list[str]:
        per_class_total = self.confusion.sum(axis=1)
        rows_n, cols_n = self.confusion.shape
        lines = [
            f"ejemplos de validacion: {per_class_total.sum()} | accuracy global: {self.accuracy:.1%} "
            f"| accuracy balanceada: {self.balanced_accuracy:.1%}\n",
            "true\\pred " + " ".join(f"{j:3d}" for j in range(cols_n)),
            *(f"{i:9d} " + " ".join(f"{self.confusion[i, j]:3d}" for j in range(cols_n)) for i in range(rows_n)),
            "",
        ]
        for c in range(rows_n):
            total = per_class_total[c]
            if total == 0:
                lines.append(f"clase {c}: sin ejemplos en validacion")
            else:
                correct = self.confusion[c, c]
                lines.append(f"clase {c}: {correct}/{total} correcto ({correct / total:.1%})")
        return lines


def evaluate_classifier(
    model: IconModel, processed_dir: Path = PROCESSED_DIR, val_split: float = VAL_SPLIT, batch_size: int = 16
) -> ClassifierReport:
    manifest = json.loads((processed_dir / MANIFEST_FILE).read_text())
    _, val_indices = split_manifest(manifest, val_split)
    loader = DataLoader(IconMaskDataset(processed_dir, augment=False, indices=val_indices), batch_size=batch_size)
    return ClassifierReport(confusion_matrix(model.classifier, loader, model.num_classes, model.num_outputs, model.device))
