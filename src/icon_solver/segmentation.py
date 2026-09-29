from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, eq=False)
class Region:
    contour: np.ndarray
    centroid: tuple[float, float]
    area: float


@dataclass(frozen=True)
class SegmentationConfig:
    # Las mismas tolerancias y areas definen la region verdadera de un click (entrenamiento) y los
    # candidatos del fondo (inferencia): si difieren, el modelo se evalua sobre otro dominio.
    tolerances: tuple[int, ...] = (40, 30, 20, 12)
    max_area: float = 4000
    min_area: float = 40
    stride: int = 3
    seed_radius: int = 2
    hole_ratio: float = 0.3
    accept_ratio: float = 1.0
    center_ratio: float = 0.5
    search_radius: int = 10
    search_step: int = 2
    legend_min_area: float = 5


DEFAULT_CONFIG = SegmentationConfig()


def region_from_mask(mask: np.ndarray) -> Region | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return Region(contour, (m["m10"] / m["m00"], m["m01"] / m["m00"]), cv2.contourArea(contour))


def _seed_pixel(image_bgr: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int]:
    # El click puede caer justo sobre el borde antialiaseado o una linea decoy que cruza el
    # icono: se siembra desde el pixel vecino cuyo color es el mas repetido en la ventana,
    # que es casi siempre el relleno del icono.
    h, w = image_bgr.shape[:2]
    y0, y1, x0, x1 = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
    window = image_bgr[y0:y1, x0:x1].reshape(-1, 3)
    colors, inverse, counts = np.unique(window, axis=0, return_inverse=True, return_counts=True)
    candidates = np.flatnonzero(inverse.ravel() == np.argmax(counts))
    ys, xs = np.divmod(candidates, x1 - x0)
    nearest = np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2)
    return int(xs[nearest] + x0), int(ys[nearest] + y0)


def _flood_mask(image_bgr: np.ndarray, seed: tuple[int, int], tolerance: int) -> np.ndarray:
    # Rango fijo respecto del color semilla, no del vecino: sin FIXED_RANGE el relleno se
    # escapa por gradientes suaves hacia formas vecinas de color parecido, que es justo la
    # fusion que se busca evitar sin erosionar iconos chicos.
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    diff = (tolerance,) * 3
    cv2.floodFill(image_bgr.copy(), mask, seed, 0, diff, diff, flags)
    return mask[1:-1, 1:-1]


def _region_at(
    image_bgr: np.ndarray, seed: tuple[int, int], config: SegmentationConfig
) -> tuple[np.ndarray, Region] | None:
    # Si el relleno se escapa a una forma vecina de color parecido el area se dispara muy por
    # encima de cualquier icono real: se reintenta con tolerancias mas estrictas.
    for tolerance in config.tolerances:
        mask = _flood_mask(image_bgr, seed, tolerance)
        region = region_from_mask(mask)
        if region is not None and region.area <= config.max_area:
            return mask, region
    return None


def _enclosing_seed(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    ring = cv2.dilate(mask, np.ones((5, 5), np.uint8)) & ~mask
    ys, xs = np.nonzero(ring)
    if len(ys) == 0:
        return None
    colors, inverse, counts = np.unique(image_bgr[ys, xs], axis=0, return_inverse=True, return_counts=True)
    dominant = np.flatnonzero(inverse.ravel() == np.argmax(counts))[0]
    return int(xs[dominant]), int(ys[dominant])


def _is_centered(region: Region, x: int, y: int, ratio: float) -> bool:
    return np.hypot(region.centroid[0] - x, region.centroid[1] - y) < ratio * np.sqrt(region.area / np.pi)


def truth_region(image_bgr: np.ndarray, point: dict, config: SegmentationConfig = DEFAULT_CONFIG) -> Region | None:
    x, y = int(round(point["x"])), int(round(point["y"]))
    region = _clicked_region(image_bgr, x, y, config)
    # Un humano clickea con varios pixeles de error: la region clickeada se acepta si el click
    # cae dentro de un radio equivalente de su centro, o si no la busqueda termina eligiendo un
    # sub-pedazo del icono (el visor del casco, el tallo de la flor) cuyo centro queda mas cerca.
    if region is not None and _is_centered(region, x, y, config.accept_ratio):
        return region
    # En iconos de trazo curvo (Leo, el telefono) el centro geometrico, donde clickea un humano y
    # que el server acepta, cae afuera del trazo: la region sembrada ahi es la forma de fondo que
    # rodea al icono, con el centroide lejos del click. Se busca en la vecindad una region que si
    # tenga el click en su centro.
    return _centered_region_near(image_bgr, x, y, config) or region


def _centered_region_near(image_bgr: np.ndarray, x: int, y: int, config: SegmentationConfig) -> Region | None:
    h, w = image_bgr.shape[:2]
    radius, step = config.search_radius, config.search_step
    best, best_dist, seen = None, np.inf, set()
    for sy in range(max(0, y - radius), min(h, y + radius + 1), step):
        for sx in range(max(0, x - radius), min(w, x + radius + 1), step):
            found = _region_at(image_bgr, (sx, sy), config)
            if found is None or found[1].area < config.min_area:
                continue
            region = found[1]
            key = (round(region.centroid[0]), round(region.centroid[1]), round(region.area))
            if key in seen:
                continue
            seen.add(key)
            dist = np.hypot(region.centroid[0] - x, region.centroid[1] - y)
            if dist < best_dist and _is_centered(region, x, y, config.center_ratio):
                best, best_dist = region, dist
    return best


def _clicked_region(image_bgr: np.ndarray, x: int, y: int, config: SegmentationConfig) -> Region | None:
    found = _region_at(image_bgr, _seed_pixel(image_bgr, x, y, config.seed_radius), config)
    if found is None:
        return None
    mask, region = found

    # Varios iconos tienen recortes internos (las flechas del 3, el hueco del casco) y el centro
    # del icono, donde clickea un humano, cae justo en el recorte: el relleno agarra el agujero.
    # Se prueba la region que lo rodea y se la acepta solo si tiene tamaño de icono, el agujero
    # es chico respecto de ella y el click esta cerca de su centro -- un icono apoyado sobre una
    # forma de fondo mas grande no cumple lo ultimo.
    outer_seed = _enclosing_seed(image_bgr, mask)
    if outer_seed is not None:
        outer_found = _region_at(image_bgr, outer_seed, config)
        outer = outer_found[1] if outer_found is not None else None
        if (
            outer is not None
            and region.area < config.hole_ratio * outer.area
            and cv2.pointPolygonTest(outer.contour, (float(x), float(y)), False) >= 0
            and np.hypot(outer.centroid[0] - x, outer.centroid[1] - y) < np.sqrt(outer.area / np.pi)
        ):
            return outer
    return region


def background_candidates(image_bgr: np.ndarray, config: SegmentationConfig = DEFAULT_CONFIG) -> list[Region]:
    # En inferencia no hay punto de click: se siembra el mismo relleno que genero los crops de
    # entrenamiento sobre una grilla, para que el modelo vea el mismo dominio. El relleno de rango
    # fijo depende de donde se siembra: sembrado en una forma de fondo puede tragarse un icono de
    # color parecido, pero sembrado en el icono no se traga el fondo. Por eso solo se da por
    # cubierto lo que esta a tolerancia estricta del color semilla, no toda la region aceptada:
    # el icono, de color algo distinto, se sigue sembrando por su cuenta.
    # Los recortes internos de un icono salen como candidatos chicos aparte; el icono que los
    # rodea se siembra desde su propio relleno.
    h, w = image_bgr.shape[:2]
    stride = config.stride
    covered = np.zeros((h, w), dtype=bool)
    regions, seen = [], set()
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            if covered[y, x]:
                continue
            covered |= _flood_mask(image_bgr, (x, y), config.tolerances[-1]) > 0
            found = _region_at(image_bgr, (x, y), config)
            if found is None or found[1].area < config.min_area:
                continue
            region = found[1]
            key = (round(region.centroid[0]), round(region.centroid[1]), round(region.area))
            if key not in seen:
                seen.add(key)
                regions.append(region)
    return regions


def legend_icons(legend_bgra: np.ndarray, config: SegmentationConfig = DEFAULT_CONFIG) -> list[Region]:
    alpha_mask = (legend_bgra[:, :, 3] > 10).astype(np.uint8) * 255
    n_labels, labels = cv2.connectedComponents(alpha_mask, connectivity=8)

    # Un icono puede tener partes desconectadas (el arco del candado abierto queda separado del
    # cuerpo): los componentes cuyos rangos en x se solapan son el mismo icono, porque la
    # leyenda los pone uno al lado del otro sin superponerse.
    groups: list[list] = []
    for label in sorted(range(1, n_labels), key=lambda lb: np.flatnonzero((labels == lb).any(axis=0))[0]):
        cols = np.flatnonzero((labels == label).any(axis=0))
        if groups and cols[0] <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], cols[-1])
            groups[-1][2].append(label)
        else:
            groups.append([cols[0], cols[-1], [label]])

    icons = []
    for _, _, group_labels in groups:
        group_mask: np.ndarray = np.isin(labels, group_labels).astype(np.uint8) * 255
        if len(group_labels) > 1:
            group_mask = cv2.fillConvexPoly(np.zeros_like(group_mask), cv2.convexHull(cv2.findNonZero(group_mask)), 255)
        icon = region_from_mask(group_mask)
        if icon is not None and icon.area >= config.legend_min_area:
            icons.append(icon)
    icons.sort(key=lambda icon: icon.centroid[0])
    return icons
