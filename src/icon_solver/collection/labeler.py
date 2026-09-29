import itertools

import cv2
import numpy as np
from curl_cffi import requests

from icon_solver.challenges import decode_bgr, decode_bgra
from icon_solver.segmentation import Region, legend_icons, region_from_mask

from .http import BotDeflectorTarget, IconChallenge, Verdict, verify_icon


def _blob(region: Region) -> dict:
    # El ranking por Hu-moments compara ademas el aspect ratio, que el resto del pipeline no usa.
    _, _, w, h = cv2.boundingRect(region.contour)
    return {
        "centroid": region.centroid,
        "contour": region.contour,
        "area": region.area,
        "aspect_ratio": w / h if h > 0 else 0.0,
    }


def _background_color(image_bgr: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(image_bgr.reshape(-1, image_bgr.shape[-1]), axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def segment_blobs(image_bgr: np.ndarray, min_area: float = 40, color_tolerance: int = 16) -> list[dict]:
    bg = _background_color(image_bgr)
    diff = np.abs(image_bgr.astype(int) - bg.astype(int)).sum(axis=-1)
    fg_mask = diff > 30

    quantized = (image_bgr.astype(int) // color_tolerance * color_tolerance)
    fg_colors = quantized[fg_mask]
    unique_colors, counts = np.unique(fg_colors, axis=0, return_counts=True)
    # Un color con menos de min_area pixeles en TODA la imagen no puede producir ningun
    # componente que pase el filtro de abajo -- saltearlo evita un connectedComponents()
    # sobre la imagen entera por cada resto de antialiasing de un puñado de pixeles (cientos
    # de colores "unicos" por imagen, la mayoria irrelevantes).
    unique_colors = unique_colors[counts >= min_area]

    blobs = []
    for color in unique_colors:
        color_mask = (np.all(quantized == color, axis=-1) & fg_mask).astype(np.uint8) * 255
        n_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
        for label in range(1, n_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            if cv2.countNonZero(component_mask) < min_area:
                continue
            region = region_from_mask(component_mask)
            blob = _blob(region) if region is not None else None
            if blob is not None and blob["area"] >= min_area:
                blob["color_bgr"] = tuple(int(c) for c in color)
                blobs.append(blob)
    return blobs


def rank_assignments(
    legend_icons: list[dict],
    candidates: list[dict],
    hu_weight: float = 1.0,
    area_weight: float = 1.0,
    aspect_weight: float = 1.0,
) -> list[tuple[float, tuple[int, ...]]]:
    # Hu-moments solo no alcanza (ver CLAUDE.md): el warp de los iconos en el fondo cambia
    # el contorno lo suficiente como para que un decoy puntue mejor que el match real. El
    # tamano y aspect ratio relativos -- icono vs el resto de la leyenda, blob vs el resto
    # de los candidatos de ese fondo -- son señales baratas e independientes de la escala
    # absoluta de cada imagen que Hu-moments no captura.
    icon_areas = np.array([icon["area"] for icon in legend_icons], dtype=float)
    icon_area_frac = icon_areas / icon_areas.sum()
    cand_areas = np.array([c["area"] for c in candidates], dtype=float)
    cand_area_frac = cand_areas / cand_areas.sum()

    scores = np.zeros((len(legend_icons), len(candidates)))
    for i, icon in enumerate(legend_icons):
        for j, cand in enumerate(candidates):
            hu = cv2.matchShapes(icon["contour"], cand["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
            area_cost = abs(icon_area_frac[i] - cand_area_frac[j])
            icon_ar, cand_ar = icon["aspect_ratio"], cand["aspect_ratio"]
            aspect_cost = abs(np.log(icon_ar / cand_ar)) if icon_ar > 0 and cand_ar > 0 else 1.0
            scores[i, j] = hu_weight * hu + area_weight * area_cost + aspect_weight * aspect_cost

    ranked = []
    for perm in itertools.permutations(range(len(candidates)), len(legend_icons)):
        cost = sum(scores[i, j] for i, j in enumerate(perm))
        ranked.append((cost, perm))
    ranked.sort(key=lambda entry: entry[0])
    return ranked


def label_challenge(
    session: requests.Session,
    challenge: IconChallenge,
    flow_token: str,
    fingerprint: str,
    target: BotDeflectorTarget,
) -> tuple[list[dict], Verdict] | None:
    # Cada challengeToken es de un solo uso, todo-o-nada, sin feedback por punto: un intento
    # equivocado no permite reintentar sobre la misma imagen (confirmado en vivo -- el widget
    # simplemente pide una imagen nueva). Por eso acá se prueba una sola vez el mejor candidato
    # (rank 0 de Hu-moments), no se itera sobre assignments como si /icon/verify fuera un oraculo
    # reutilizable. El dataset se arma cosechando los aciertos de ese primer intento (bootstrapping).
    legend = decode_bgra(challenge.legend_bytes)
    background = decode_bgr(challenge.background_bytes)

    icons = [_blob(icon) for icon in legend_icons(legend)]
    blobs = segment_blobs(background)
    ranked = rank_assignments(icons, blobs)
    if not ranked:
        return None

    _, perm = ranked[0]
    points = [{"x": round(blobs[j]["centroid"][0]), "y": round(blobs[j]["centroid"][1])} for j in perm]
    verdict = verify_icon(session, challenge, flow_token, target, points, fingerprint, solve_time_ms=1500)
    return (points, verdict) if verdict.verified else None
