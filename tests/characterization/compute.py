import contextlib
import hashlib
import io
import json
import shutil
import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import torch

from icon_solver import segmentation as seg
from icon_solver.challenges import Challenge, ChallengeStore, split_manifest
from icon_solver.dataset.build import PROTOTYPES_FILE, build_dataset, derive_labels
from icon_solver.evaluation import evaluate_classifier, evaluate_solver
from icon_solver.legend import load_prototypes
from icon_solver.model import IconModel
from icon_solver.paths import EXAMPLES_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR
from icon_solver.shapes import encode_regions, legend_silhouettes, region_silhouettes, silhouette_similarity

ICON_MODEL_PATH = MODELS_DIR / "icon_model.pt"
VAL_SPLIT = 0.2
TRAIN_SAMPLE = 10
ENCODING_SAMPLE = 5


def sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}{array.shape}".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def raw_store() -> ChallengeStore:
    return ChallengeStore(RAW_DIR, VAL_SPLIT)


@cache
def challenges() -> dict[str, Challenge]:
    out = {f"raw/{c.id}": c for c in raw_store()}
    out.update({f"examples/{c.id}": c for c in ChallengeStore(EXAMPLES_DIR)})
    return out


def val_ids() -> list[str]:
    return [c.id for c in raw_store().iter("val")]


def segmentation_sample_ids() -> list[str]:
    train = [c.id for c in raw_store().iter("train")]
    return sorted(val_ids()) + train[:TRAIN_SAMPLE]


def encoding_sample_ids() -> list[str]:
    return val_ids()[:ENCODING_SAMPLE]


def region_summary(region: seg.Region | None) -> list | None:
    if region is None:
        return None
    cx, cy = region.centroid
    return [float(cx), float(cy), float(region.area), sha(region.contour.astype(np.int32))]


def solver_points(solver) -> tuple[dict, float]:
    points, val, val_seconds = {}, set(val_ids()), []
    for key, challenge in challenges().items():
        start = time.perf_counter()
        points[key] = solver.solve(challenge.background, challenge.legend)
        if key.startswith("raw/") and challenge.id in val:
            val_seconds.append(time.perf_counter() - start)
    return points, float(np.mean(val_seconds))


def icon_model() -> IconModel:
    return IconModel.load(ICON_MODEL_PATH, torch.device("cpu"))


def legend_classes(model: IconModel) -> dict:
    return {key: model.legend_classes(challenge.legend) for key, challenge in challenges().items()}


def segmentation(ids: list[str]) -> dict:
    out = {}
    for example_id in ids:
        challenge = challenges()[f"raw/{example_id}"]
        candidates = [region_summary(r) for r in seg.background_candidates(challenge.background)]
        out[example_id] = {
            "candidates": {"count": len(candidates), "sha256": sha_json(candidates)},
            "truth": [region_summary(seg.truth_region(challenge.background, p)) for p in challenge.points or []],
        }
    return out


def encoding(model: IconModel, ids: list[str]) -> dict:
    out = {}
    for example_id in ids:
        challenge = challenges()[f"raw/{example_id}"]
        regions = seg.background_candidates(challenge.background)
        x = encode_regions(regions)
        probs = model.class_probs(regions)
        silhouettes = legend_silhouettes(challenge.legend, seg.legend_icons(challenge.legend))
        out[example_id] = {
            "input": sha(x.numpy()),
            "probs": sha(probs),
            "similarity": sha(silhouette_similarity(silhouettes, region_silhouettes(regions))),
        }
    return out


def _pixels_sha(path: Path) -> str:
    return sha(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))


def build(out_dir: Path) -> dict:
    # Sin las plantillas persistidas los ids de clase se reasignan por orden de iteracion.
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROCESSED_DIR / PROTOTYPES_FILE, out_dir / PROTOTYPES_FILE)
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        build_dataset(raw_dir=RAW_DIR, out_dir=out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())

    crops_by_example: dict[str, list] = {}
    for entry in manifest:
        crops_by_example.setdefault(entry["example"], []).append([entry["crop"], _pixels_sha(out_dir / entry["crop"])])
    templates = np.load(out_dir / PROTOTYPES_FILE)
    return {
        "stdout": stdout.getvalue().replace(str(out_dir), "<out_dir>"),
        "num_entries": len(manifest),
        "manifest_sha256": sha_json(manifest),
        "crops_sha256_by_example": {k: sha_json(v) for k, v in sorted(crops_by_example.items())},
        "meta": json.loads((out_dir / "meta.json").read_text()),
        "legend_templates": {p.name: _pixels_sha(p) for p in sorted((out_dir / "legend_templates").iterdir())},
        "canonical_templates": {"masks": sha(templates["masks"]), "classes": sha(templates["classes"])},
        "split": split(manifest),
    }


def split(manifest: list[dict]) -> dict:
    train, val = split_manifest(manifest, VAL_SPLIT)
    return {"train_sha256": sha_json(train), "train_count": len(train), "val": val}


def solver_metrics(solver) -> dict:
    start = time.perf_counter()
    report = evaluate_solver(raw_store(), solver)
    return {
        "result": {"solved": report.solved, "icons": report.icons, "candidates": report.candidates},
        "stdout": "\n".join(report.lines()) + "\n",
        "seconds": time.perf_counter() - start,
    }


def classifier_metrics() -> dict:
    report = evaluate_classifier(icon_model(), PROCESSED_DIR, VAL_SPLIT)
    return {"stdout": "\n".join(report.lines()) + "\n"}


def derived_labels() -> dict:
    # Mismo manifest y crops que build_dataset, pero sacados de la funcion pura sin escribir nada.
    prototypes = load_prototypes(PROCESSED_DIR / PROTOTYPES_FILE)
    manifest, crops_by_example, skipped = [], {}, 0
    for challenge in raw_store():
        labels = derive_labels(challenge, prototypes)
        if labels is None:
            continue
        crops = crops_by_example.setdefault(challenge.id, [])
        for index, (class_id, crop) in enumerate(zip(labels.icon_classes, labels.crops)):
            if crop is None:
                skipped += 1
                continue
            path = f"crops/{class_id}/{challenge.id}_{index}.png"
            manifest.append({"crop": path, "class_id": class_id, "source": "real", "example": challenge.id, "labeler": challenge.labeler})
            crops.append([path, sha(crop)])
        for index, crop in enumerate(labels.negatives):
            path = f"crops/neg/{challenge.id}_{index}.png"
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
            crops.append([path, sha(crop)])
    return {
        "manifest_sha256": sha_json(manifest),
        "crops_sha256_by_example": {k: sha_json(v) for k, v in sorted(crops_by_example.items()) if v},
        "skipped": skipped,
        "num_classes": prototypes.num_classes,
    }
