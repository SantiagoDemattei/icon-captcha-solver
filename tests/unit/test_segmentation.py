import cv2
import numpy as np

from icon_solver.segmentation import legend_icons


def _legend_with_open_lock() -> np.ndarray:
    legend = np.zeros((27, 86, 4), dtype=np.uint8)
    cv2.rectangle(legend, (4, 4), (20, 20), (90, 90, 90, 255), cv2.FILLED)
    # Candado abierto: el arco queda desconectado del cuerpo pero solapado en x.
    cv2.rectangle(legend, (32, 13), (48, 23), (90, 90, 90, 255), cv2.FILLED)
    cv2.ellipse(legend, (40, 9), (6, 5), 0, 180, 360, (90, 90, 90, 255), 2)
    cv2.circle(legend, (70, 13), 8, (90, 90, 90, 255), cv2.FILLED)
    return legend


def test_disconnected_parts_overlapping_in_x_are_one_icon():
    legend = _legend_with_open_lock()
    n_components = cv2.connectedComponents((legend[:, :, 3] > 10).astype(np.uint8))[0] - 1
    assert n_components == 4

    icons = legend_icons(legend)

    assert len(icons) == 3
    assert [round(icon.centroid[0]) // 30 for icon in icons] == [0, 1, 2]
