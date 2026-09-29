import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from icon_solver import segmentation
from icon_solver.challenges import Challenge, ChallengeStore
from icon_solver.legend import IconPrototypes, load_prototypes, prototype_mask, save_prototypes
from icon_solver.paths import PROCESSED_DIR, RAW_DIR
from icon_solver.segmentation import DEFAULT_CONFIG, Region, SegmentationConfig
from icon_solver.shapes import legend_template, region_crop

NEGATIVES_PER_EXAMPLE = 30
PROTOTYPES_FILE = "canonical_templates.npz"
MANIFEST_FILE = "manifest.json"
META_FILE = "meta.json"
LEGEND_TEMPLATES_DIR = "legend_templates"
CROPS_DIR = "crops"


@dataclass
class ChallengeLabels:
    icon_classes: list[int]
    legend_templates: list[np.ndarray]
    truth_regions: list[Region | None]
    # Un crop por punto de click; None si no hubo region valida y el crop se descarta.
    crops: list[np.ndarray | None]
    negatives: list[np.ndarray]

    @property
    def excluded_classes(self) -> list[int]:
        return sorted(set(self.icon_classes))


def derive_labels(
    challenge: Challenge,
    prototypes: IconPrototypes,
    config: SegmentationConfig = DEFAULT_CONFIG,
    negatives_limit: int = NEGATIVES_PER_EXAMPLE,
) -> ChallengeLabels | None:
    # Sin E/S: decide que etiquetas salen de un challenge etiquetado. El registro de clases crece
    # si aparece un icono nuevo en la leyenda, en el orden de la leyenda.
    points = challenge.points or []
    icons = segmentation.legend_icons(challenge.legend, config)
    if len(icons) != len(points):
        return None

    classes, templates, truth_regions, crops = [], [], [], []
    for icon, point in zip(icons, points):
        classes.append(prototypes.match_or_add(prototype_mask(challenge.legend, icon.contour)))
        templates.append(legend_template(challenge.legend, icon.contour))
        # Sin region de tamaño de icono alrededor del click (fondo de bajo contraste): mejor perder
        # el crop que entrenar con un pedazo de fondo.
        region = segmentation.truth_region(challenge.background, point, config)
        truth_regions.append(region)
        crops.append(None if region is None else region_crop(region.contour))

    negatives = []
    if negatives_limit > 0:
        regions = _negatives(challenge.background, points, truth_regions, negatives_limit, challenge.id, config)
        negatives = [region_crop(r.contour) for r in regions]
    return ChallengeLabels(classes, templates, truth_regions, crops, negatives)


def _negatives(
    background: np.ndarray,
    points: list[dict],
    truth_regions: list[Region | None],
    limit: int,
    seed: str,
    config: SegmentationConfig,
) -> list[Region]:
    # El captcha garantiza que los decoys son iconos de la libreria distintos de los pedidos: todo
    # candidato que no toca un icono marcado es con seguridad "no es ninguno de los pedidos" (no
    # sabemos cual es, pero eso alcanza como señal negativa). Se descarta cualquier candidato que
    # contenga un click o cuyo centro caiga dentro de un icono marcado (agujeros, pedazos).
    negatives = []
    for region in segmentation.background_candidates(background, config):
        touches_truth = any(
            cv2.pointPolygonTest(region.contour, (float(p["x"]), float(p["y"])), False) >= 0 for p in points
        ) or any(r is not None and cv2.pointPolygonTest(r.contour, region.centroid, False) >= 0 for r in truth_regions)
        if not touches_truth:
            negatives.append(region)
    random.Random(seed).shuffle(negatives)
    return negatives[:limit]


def build_dataset(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    negatives_per_example: int = NEGATIVES_PER_EXAMPLE,
) -> None:
    # Los class_id son el indice en el registro de prototipos: se parte del ya persistido para que
    # un rebuild con challenges nuevos no reordene las clases (solo agrega al final).
    prototypes_path = out_dir / PROTOTYPES_FILE
    prototypes = load_prototypes(prototypes_path) if prototypes_path.exists() else IconPrototypes()
    crops_dir = out_dir / CROPS_DIR
    if crops_dir.exists():
        # Se regenera entero desde data/raw en cada corrida; sin esto, crops de una
        # asignacion de class_id vieja quedan huerfanos mezclados con los nuevos.
        shutil.rmtree(crops_dir)

    class_templates: dict[int, np.ndarray] = {}
    manifest: list[dict] = []
    skipped = 0
    for challenge in ChallengeStore(raw_dir):
        labels = derive_labels(challenge, prototypes, negatives_limit=negatives_per_example)
        if labels is None:
            continue
        for class_id, template in zip(labels.icon_classes, labels.legend_templates):
            class_templates.setdefault(class_id, template)

        for icon_index, (class_id, crop) in enumerate(zip(labels.icon_classes, labels.crops)):
            if crop is None:
                skipped += 1
                continue
            # Con el indice: una leyenda puede repetir el mismo icono, y sin el los dos crops
            # comparten nombre y el segundo pisa al primero.
            path = _write_crop(crops_dir, str(class_id), f"{challenge.id}_{icon_index}.png", crop)
            manifest.append(
                {"crop": path, "class_id": class_id, "source": "real", "example": challenge.id, "labeler": challenge.labeler}
            )

        for k, crop in enumerate(labels.negatives):
            path = _write_crop(crops_dir, "neg", f"{challenge.id}_{k}.png", crop)
            manifest.append(
                {
                    "crop": path,
                    "class_id": -1,
                    "exclude": labels.excluded_classes,
                    "source": "negative",
                    "example": challenge.id,
                    "labeler": challenge.labeler,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = out_dir / LEGEND_TEMPLATES_DIR
    templates_dir.mkdir(parents=True, exist_ok=True)
    for class_id, template in class_templates.items():
        cv2.imwrite(str(templates_dir / f"{class_id}.png"), template)

    real_count = sum(1 for e in manifest if e["source"] == "real")
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    (out_dir / META_FILE).write_text(json.dumps({"num_classes": prototypes.num_classes}, indent=2))
    save_prototypes(prototypes, prototypes_path)
    print(
        f"{real_count} crops reales ({skipped} descartados sin region valida) + {len(manifest) - real_count} negativos, "
        f"{prototypes.num_classes} clases -> {out_dir}"
    )


def _write_crop(crops_dir: Path, subdir: str, name: str, crop: np.ndarray) -> str:
    directory = crops_dir / subdir
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / name), crop)
    return f"{CROPS_DIR}/{subdir}/{name}"
