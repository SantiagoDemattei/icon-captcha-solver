import json
from pathlib import Path

import cv2
import numpy as np

from icon_solver.paths import PROCESSED_DIR
from icon_solver.shapes import CROP_SIZE, read_mask

from .build import LEGEND_TEMPLATES_DIR, MANIFEST_FILE

ANGLE_STEP = 5
SCALES = (0.85, 1.0, 1.15)


def _rotated_variants(template_mask: np.ndarray, size: int) -> list[np.ndarray]:
    resized = cv2.resize(template_mask * 255, (size, size), interpolation=cv2.INTER_NEAREST)
    variants = []
    for scale in SCALES:
        for angle in range(0, 360, ANGLE_STEP):
            m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
            warped = cv2.warpAffine(resized, m, (size, size), borderValue=0)
            variants.append((warped > 127).astype(np.uint8))
    return variants


def _best_match(mask: np.ndarray, variants: list[np.ndarray]) -> tuple[float, np.ndarray]:
    # Ninguna divergencia supera mask.size: la primera variante siempre reemplaza al inicial.
    best_variant, best_score = variants[0], mask.size + 1
    for variant in variants:
        score = np.logical_xor(mask, variant).sum()
        if score < best_score:
            best_score, best_variant = score, variant
    return best_score / mask.size, best_variant


def audit_crops(
    processed_dir: Path = PROCESSED_DIR, worst_n: int = 12, bad_threshold: float = 0.35
) -> None:
    # Compara cada crop real contra la mejor rotacion+escala posible de la plantilla limpia
    # de su propia clase verdadera -- si el modelo geometrico (solo rotacion + escala
    # uniforme, confirmado por el usuario) fuera exacto, esta divergencia deberia ser baja
    # (ruido de antialiasing) para casi todos los crops. Una divergencia alta sistematica
    # apunta a un problema de segmentacion (blob fusionado o blob equivocado), no a que
    # falte augmentar mas o a que el modelo sea chico.
    manifest = json.loads((processed_dir / MANIFEST_FILE).read_text())
    templates_dir = processed_dir / LEGEND_TEMPLATES_DIR

    variants_by_class: dict[int, list[np.ndarray]] = {}
    scored = []
    best_variants: dict[str, np.ndarray] = {}
    for entry in manifest:
        class_id = entry["class_id"]
        # Los negativos no tienen clase conocida: no hay plantilla contra la cual compararlos.
        if class_id < 0:
            continue
        if class_id not in variants_by_class:
            template = read_mask(templates_dir / f"{class_id}.png")
            variants_by_class[class_id] = _rotated_variants((template > 127).astype(np.uint8), CROP_SIZE)

        crop = read_mask(processed_dir / entry["crop"])
        mask = (crop > 127).astype(np.uint8)
        divergence, best_variant = _best_match(mask, variants_by_class[class_id])
        scored.append((divergence, class_id, entry["crop"]))
        best_variants[entry["crop"]] = best_variant

    scored.sort(reverse=True)
    divergences = np.array([d for d, _, _ in scored])
    mean_div = divergences.mean()

    print(f"{len(scored)} crops auditados")
    print(f"divergencia media vs mejor rotacion propia: {mean_div:.1%} (mediana {np.median(divergences):.1%})")
    print(f"crops limpios (<15%): {(divergences < 0.15).mean():.1%}")
    print(f"crops muy corruptos (>{bad_threshold:.0%}): {(divergences > bad_threshold).mean():.1%}")

    print("\ndivergencia media por clase:")
    for class_id in sorted(variants_by_class):
        class_divs = [d for d, c, _ in scored if c == class_id]
        flag = "  <-- revisar" if np.mean(class_divs) > mean_div + 0.05 else ""
        print(f"  clase {class_id}: {np.mean(class_divs):.1%} sobre {len(class_divs)} crops{flag}")

    print(f"\npeores {worst_n} crops (comparar contra data/raw/<uuid>/background.webp + points.json):")
    for divergence, class_id, crop_path in scored[:worst_n]:
        example_id = Path(crop_path).stem.split("_")[0]
        print(f"  clase {class_id} divergencia={divergence:.1%} -> data/raw/{example_id}/")

    _save_worst_grid(processed_dir, scored[:worst_n], best_variants)


def _save_worst_grid(processed_dir: Path, worst: list[tuple[float, int, str]], best_variants: dict) -> None:
    rows = []
    gap = np.full((CROP_SIZE, 6), 128, dtype=np.uint8)
    for _, _, crop_path in worst:
        crop = read_mask(processed_dir / crop_path)
        template_match = best_variants[crop_path] * 255
        rows.append(np.hstack([crop, gap, template_match]))
    sep = np.full((4, rows[0].shape[1]), 64, dtype=np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep, row])
    out_path = processed_dir / "audit_worst.png"
    cv2.imwrite(str(out_path), grid)
    print(f"\ncomparacion visual (crop real | mejor rotacion de la plantilla) -> {out_path}")
