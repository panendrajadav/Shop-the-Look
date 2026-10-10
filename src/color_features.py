"""Color histograms used to compare a detected garment with catalog products.

A 28-bin histogram: 4 achromatic bins (black, dark grey, light grey, white) and 12 hues at
2 brightness levels. Two histograms are compared by intersection, which gives a continuous
0..1 color agreement instead of a single named color per image.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image

N_BINS = 28
UPPER_BOX = (0.08, 0.05, 0.92, 0.58)
LOWER_BOX = (0.10, 0.42, 0.90, 0.92)


def product_color_box(category: str) -> Optional[Tuple[float, float, float, float]]:
    """Body band of a catalog photo that holds the product, for garments shown on a model."""
    if category in ("Top / T-shirt", "Jacket / Outerwear"):
        return UPPER_BOX
    if category in ("Pants / Trousers", "Skirt"):
        return LOWER_BOX
    return None


def color_histogram(
    image: Image.Image,
    box: Optional[Tuple[float, float, float, float]] = None,
    remove_background: bool = False,
    center_only: bool = False,
) -> np.ndarray:
    """L1-normalized 28-bin color histogram of the garment pixels in an image.

    box: fractional (x1, y1, x2, y2) region to use.
    remove_background: drop pixels close to the border color (studio backdrops).
    center_only: use the central part of the crop, where the garment is.
    """
    rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
    keep = np.ones((64, 64), dtype=bool)

    if remove_background:
        border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                                 rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        if np.mean(np.linalg.norm(border - bg, axis=1) < 30) > 0.7:
            keep &= np.linalg.norm(rgb - bg, axis=2) >= 34

    region = np.zeros((64, 64), dtype=bool)
    if box is not None:
        x1, y1, x2, y2 = (int(round(v * 64)) for v in box)
        region[y1:y2, x1:x2] = True
    elif center_only:
        region[10:54, 13:51] = True
    else:
        region[:] = True
    if (keep & region).sum() >= 40:
        keep &= region
    elif keep.sum() < 40:
        keep = region

    px = rgb[keep] / 255.0
    mx, mn = px.max(axis=1), px.min(axis=1)
    delta = mx - mn
    v = mx
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hue = np.zeros(len(px), dtype=np.float32)
    nz = delta > 1e-6
    rmax, gmax = nz & (mx == r), nz & (mx == g) & (mx != r)
    bmax = nz & ~rmax & ~gmax
    hue[rmax] = ((g - b)[rmax] / delta[rmax]) % 6
    hue[gmax] = (b - r)[gmax] / delta[gmax] + 2
    hue[bmax] = (r - g)[bmax] / delta[bmax] + 4
    hue = hue * 60.0

    achromatic = (s < 0.18) | (v < 0.18)
    bins = np.empty(len(px), dtype=np.int64)
    bins[achromatic] = np.digitize(v[achromatic], [0.25, 0.55, 0.85])
    chroma = ~achromatic
    bins[chroma] = 4 + (hue[chroma] // 30).astype(np.int64) % 12 * 2 + (v[chroma] >= 0.55)

    hist = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return hist / max(1.0, hist.sum())
