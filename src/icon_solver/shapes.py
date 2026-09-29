from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from icon_solver.segmentation import Region

CROP_SIZE = 48
LEGEND_TEMPLATE_SIZE = 64
LEGEND_ALPHA_THRESHOLD = 10
SILHOUETTE_ANGLES = 64
SILHOUETTE_RADII = 24
SILHOUETTE_UPSCALE = 4
SILHOUETTE_RADIUS_FACTOR = 1.3


def square_pad(mask: np.ndarray) -> np.ndarray:
    # Sin esto, resize() estira el bounding box no cuadrado directo a un cuadrado, y ese
    # estiramiento depende del angulo de rotacion del blob -- reintroduce por la puerta de
    # atras la deformacion no uniforme que el usuario confirmo que no existe en el captcha real.
    h, w = mask.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=mask.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = mask
    return canvas


def legend_alpha(legend_bgra: np.ndarray, contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    return (legend_bgra[y : y + h, x : x + w, 3] > LEGEND_ALPHA_THRESHOLD).astype(np.uint8) * 255


def region_crop(contour: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)
    return cv2.resize(square_pad(mask), (size, size), interpolation=cv2.INTER_AREA)


def legend_template(legend_bgra: np.ndarray, contour: np.ndarray, size: int = LEGEND_TEMPLATE_SIZE) -> np.ndarray:
    return cv2.resize(square_pad(legend_alpha(legend_bgra, contour)), (size, size), interpolation=cv2.INTER_AREA)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"no se pudo leer la mascara {path}")
    return mask


def to_polar(mask: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    # Mapas explicitos + remap en vez de cv2.warpPolar: warpPolar no es determinista (el mismo
    # input da pixeles distintos entre llamadas, incluso con un solo hilo), y eso hacia que la
    # evaluacion del mismo checkpoint variara varios puntos entre corridas.
    m = cv2.moments(mask)
    cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else (mask.shape[1] / 2, mask.shape[0] / 2)
    theta = np.arange(size) * 2 * np.pi / size
    rho = np.arange(size) * (size / 2) / size
    map_x = (cx + rho[None, :] * np.cos(theta[:, None])).astype(np.float32)
    map_y = (cy + rho[None, :] * np.sin(theta[:, None])).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_input(polar_masks: np.ndarray) -> torch.Tensor:
    # Filas = angulo, columnas = radio: rotar la forma es un roll exacto de las filas.
    return torch.from_numpy(polar_masks).float().unsqueeze(1) / 255.0


def encode_crops(crops: np.ndarray) -> torch.Tensor:
    return to_input(np.stack([to_polar(crop) for crop in crops]))


def encode_regions(regions: Sequence[Region]) -> torch.Tensor:
    return encode_crops(np.stack([region_crop(r.contour) for r in regions]))


def _filled_region_mask(contour: np.ndarray, pad: int = 2) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(mask, [contour - [x - pad, y - pad]], -1, 255, cv2.FILLED)
    return mask


def _filled_legend_mask(legend_bgra: np.ndarray, contour: np.ndarray, pad: int = 2) -> np.ndarray:
    # Los candidatos del fondo se rellenan por contorno externo (sin agujeros): la silueta de la
    # leyenda se rellena igual para comparar la misma forma.
    alpha = np.pad(legend_alpha(legend_bgra, contour), pad)
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(alpha)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def _silhouette(mask: np.ndarray) -> np.ndarray:
    # Polar centrado en el centroide con radio proporcional a sqrt(area): la escala y la posicion
    # quedan normalizadas sin depender del bounding box (que cambia con la rotacion), y la
    # rotacion queda como un shift circular del eje angular.
    mask = cv2.resize(mask, None, fx=SILHOUETTE_UPSCALE, fy=SILHOUETTE_UPSCALE, interpolation=cv2.INTER_LINEAR)
    m = cv2.moments(mask)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    rmax = SILHOUETTE_RADIUS_FACTOR * np.sqrt(m["m00"] / 255)
    theta = np.arange(SILHOUETTE_ANGLES) * 2 * np.pi / SILHOUETTE_ANGLES
    rho = (np.arange(SILHOUETTE_RADII) + 0.5) * rmax / SILHOUETTE_RADII
    map_x = (cx + rho[None] * np.cos(theta[:, None])).astype(np.float32)
    map_y = (cy + rho[None] * np.sin(theta[:, None])).astype(np.float32)
    polar = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return polar.astype(np.float32) / 255


def legend_silhouettes(legend_bgra: np.ndarray, icons: Sequence[Region]) -> np.ndarray:
    return np.stack([_silhouette(_filled_legend_mask(legend_bgra, icon.contour)) for icon in icons])


def region_silhouettes(regions: Sequence[Region]) -> np.ndarray:
    return np.stack([_silhouette(_filled_region_mask(r.contour)) for r in regions])


def silhouette_similarity(legend: np.ndarray, regions: np.ndarray) -> np.ndarray:
    # IoU suave entre cada silueta de leyenda y cada region, maximizado sobre todas las rotaciones a
    # la vez: la correlacion circular del eje angular sale de una FFT.
    ft = np.fft.fft(legend, axis=1)[:, None]
    fc = np.fft.fft(regions, axis=1)[None]
    inter = np.real(np.fft.ifft(ft * np.conj(fc), axis=2)).sum(-1).max(-1)
    union = legend.sum((1, 2))[:, None] + regions.sum((1, 2))[None] - inter
    return inter / union
