import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.challenges import ChallengeStore  # noqa: E402
from icon_solver.paths import RAW_DIR  # noqa: E402
from icon_solver.shapes import region_crop, to_polar  # noqa: E402
from icon_solver.solver import IconSolver  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "img"
# Un challenge de validacion etiquetado por un humano: el modelo nunca lo vio al entrenar.
EXAMPLE = "e509e52867b145249975262f309f199e"
# Paleta Okabe-Ito: distinguible con daltonismo, un color por icono de la leyenda.
ICON_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
GRAY = "#8a8a8a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _legend_on_white(legend_bgra: np.ndarray) -> np.ndarray:
    alpha = legend_bgra[:, :, 3:4].astype(np.float32) / 255
    rgb = cv2.cvtColor(legend_bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _contour_xy(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([contour[:, 0], contour[:1, 0]])
    return points[:, 0], points[:, 1]


def challenge_figure(challenge, path: Path) -> None:
    # Como en el widget real: la leyenda arriba, a la misma escala que el fondo.
    legend, background = challenge.legend, challenge.background
    fig, axes = plt.subplots(
        2, 1, figsize=(5.2, 5.0), gridspec_kw={"height_ratios": [legend.shape[0] * 1.6, background.shape[0]]}
    )
    width = background.shape[1]
    canvas = np.full((legend.shape[0], width, 3), 255, np.uint8)
    canvas[:, : legend.shape[1]] = _legend_on_white(legend)
    axes[0].imshow(canvas, interpolation="nearest")
    axes[0].set_title("Leyenda: qué clickear y en qué orden", loc="left", fontsize=10)
    axes[1].imshow(_rgb(background))
    axes[1].set_title("Fondo: los mismos íconos, rotados y escalados, entre señuelos", loc="left", fontsize=10)
    for ax in axes:
        _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def pipeline_figure(challenge, trace, path: Path) -> None:
    background = _rgb(challenge.background)
    dimmed = (background * 0.45 + 255 * 0.55).astype(np.uint8)
    fig = plt.figure(figsize=(12, 2.9))
    grid = fig.add_gridspec(1, 4, wspace=0.06)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(background)
    ax.set_title("1. Fondo")
    _clean(ax)

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(dimmed)
    rng = np.random.default_rng(0)
    for region in trace.candidates:
        xs, ys = _contour_xy(region.contour)
        ax.plot(xs, ys, color=plt.cm.tab20(rng.integers(20)), linewidth=0.6)
    ax.set_title(f"2. Candidatos ({len(trace.candidates)})")
    _clean(ax)

    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(dimmed)
    for k in range(trace.scores.shape[1]):
        for rank, i in enumerate(np.argsort(-trace.scores[:, k])[:4]):
            xs, ys = _contour_xy(trace.candidates[i].contour)
            best = rank == 0
            ax.plot(xs, ys, color=ICON_COLORS[k] if best else GRAY, linewidth=2.2 if best else 0.9, linestyle="-" if best else "--")
    ax.set_title("3. Mejores por ícono")
    _clean(ax)

    ax = fig.add_subplot(grid[0, 3])
    ax.imshow(background)
    for k, point in enumerate(trace.points):
        ax.scatter(point["x"], point["y"], s=260, color=ICON_COLORS[k], edgecolors="white", linewidths=1.8, zorder=3)
        ax.text(point["x"], point["y"], str(k + 1), color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    ax.set_title("4. Clicks en orden")
    _clean(ax)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _chosen_regions(trace) -> list:
    return [
        min(trace.candidates, key=lambda r: np.hypot(r.centroid[0] - p["x"], r.centroid[1] - p["y"])) for p in trace.points
    ]


def classifier_view_figure(challenge, trace, path: Path) -> None:
    # Una fila por icono: la forma limpia de la leyenda, la region que se eligio en el fondo y su
    # codificacion polar, que es lo que recibe el clasificador.
    legend = _legend_on_white(challenge.legend)
    regions = _chosen_regions(trace)
    fig, axes = plt.subplots(len(regions), 3, figsize=(5.6, 1.9 * len(regions)))
    for k, (icon, region) in enumerate(zip(trace.legend_icons, regions)):
        x, y, w, h = cv2.boundingRect(icon.contour)
        crop = region_crop(region.contour)
        axes[k, 0].imshow(legend[max(0, y - 2) : y + h + 2, max(0, x - 2) : x + w + 2], interpolation="nearest")
        axes[k, 1].imshow(crop, cmap="gray_r", interpolation="nearest")
        axes[k, 2].imshow(to_polar(crop).T, cmap="gray_r", interpolation="nearest", origin="lower", aspect="auto")
        for j in range(3):
            _clean(axes[k, j])
        for spine in axes[k, 0].spines.values():
            spine.set_visible(True)
            spine.set_color(ICON_COLORS[k])
            spine.set_linewidth(2.5)
        axes[k, 0].set_ylabel(f"ícono {k + 1}", color=ICON_COLORS[k], fontweight="bold")
    for j, label in enumerate(["leyenda", "región elegida", "entrada polar"]):
        axes[0, j].set_title(label, fontweight="normal")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def polar_figure(challenge, trace, path: Path) -> None:
    crop = region_crop(_chosen_regions(trace)[0].contour)

    fig, axes = plt.subplots(2, 4, figsize=(9, 4.6), gridspec_kw={"height_ratios": [1, 1]})
    for k in range(4):
        rotated = np.ascontiguousarray(np.rot90(crop, k))
        axes[0, k].imshow(rotated, cmap="gray_r", interpolation="nearest")
        axes[0, k].set_title(f"rotado {90 * k}°", fontweight="normal")
        axes[1, k].imshow(to_polar(rotated).T, cmap="gray_r", interpolation="nearest", aspect="auto", origin="lower")
        _clean(axes[0, k])
        _clean(axes[1, k])
    axes[0, 0].set_ylabel("región", fontsize=10)
    axes[1, 0].set_ylabel("radio ↑", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("ángulo →", fontsize=9)
    fig.suptitle("En coordenadas polares, rotar la forma es desplazar la imagen en el eje del ángulo", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    challenge = ChallengeStore(RAW_DIR).load(EXAMPLE)
    trace = IconSolver().trace(challenge.background, challenge.legend)
    challenge_figure(challenge, OUT_DIR / "challenge.png")
    pipeline_figure(challenge, trace, OUT_DIR / "pipeline.png")
    classifier_view_figure(challenge, trace, OUT_DIR / "classifier_view.png")
    polar_figure(challenge, trace, OUT_DIR / "polar.png")
    print(f"figuras -> {OUT_DIR}")


if __name__ == "__main__":
    main()
