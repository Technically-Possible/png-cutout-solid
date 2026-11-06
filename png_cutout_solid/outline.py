# SPDX-FileCopyrightText: 2026 Jennifer Freeman
# SPDX-License-Identifier: GPL-3.0-or-later

"""Image -> outline tracing used by PNG Cutout Solid.

Pure numpy / Python with no Blender imports, so it can be unit-tested
outside Blender (see ``tests/test_outline.py``).

Coordinates: fields are 2D arrays indexed ``[row, col]`` with row 0 at the
*bottom* (Blender's pixel order). Returned outline points are ``(x, y)`` in
field-pixel units, where the centre of field pixel ``[j, i]`` is ``(i, j)``.
"""

import math

import numpy as np


def pixels_to_field(px, source="AUTO", invert=False, max_res=1024):
    """Turn an ``(h, w, channels)`` float pixel array into a solidity field.

    Returns ``(field, source, k)``:

    * ``field``  - 2D float32 array, values >= threshold count as solid.
    * ``source`` - the mask actually used, ``"ALPHA"`` or ``"LUMINANCE"``
      (``"AUTO"`` is resolved here).
    * ``k``      - downscale factor; each field pixel covers ``k x k`` image
      pixels (1 = full resolution).
    """
    h, w, ch = px.shape
    rgb = px[..., :3] if ch >= 3 else np.repeat(px[..., :1], 3, axis=2)
    if ch == 4:
        alpha = px[..., 3]
    elif ch == 2:
        alpha = px[..., 1]
    else:
        alpha = np.ones((h, w), np.float32)

    if source == "AUTO":
        source = "ALPHA" if float(alpha.min()) < 0.99 else "LUMINANCE"

    if source == "ALPHA":
        field = alpha
    else:  # dark pixels = solid (ink on white paper)
        lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        field = (1.0 - lum) * alpha

    if invert:
        field = 1.0 - field

    # block-average downscale for huge images
    k = int(math.ceil(max(w, h) / float(max_res))) if max_res > 0 else 1
    if k > 1:
        hh, ww = (h // k) * k, (w // k) * k
        field = field[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
    k = max(k, 1)

    return np.ascontiguousarray(field, dtype=np.float32), source, k


# Directed marching-squares segments (solid always on the LEFT of travel,
# so outer outlines run counter-clockwise and holes clockwise).
# Corners: a=bottom-left(1) b=bottom-right(2) c=top-right(4) d=top-left(8)
# Edges:   B=bottom  R=right  T=top  L=left
_SEGS = {
    1: [("B", "L")], 2: [("R", "B")], 3: [("R", "L")], 4: [("T", "R")],
    6: [("T", "B")], 7: [("T", "L")], 8: [("L", "T")], 9: [("B", "T")],
    11: [("R", "T")], 12: [("L", "R")], 13: [("B", "R")], 14: [("L", "B")],
}
_SADDLE = {  # (centre solid, centre empty)
    5: ([("B", "R"), ("T", "L")], [("B", "L"), ("T", "R")]),
    10: ([("L", "B"), ("R", "T")], [("R", "B"), ("L", "T")]),
}


def trace_contours(field, thr=0.5):
    """Marching squares with sub-pixel interpolation.

    Returns a list of closed loops, each a list of ``(x, y)`` points.
    Outer outlines are counter-clockwise (positive area), holes clockwise.
    """
    f = np.pad(field, 1, mode="constant", constant_values=0.0)
    ins = f >= thr
    idx = (ins[:-1, :-1] * 1 + ins[:-1, 1:] * 2 + ins[1:, 1:] * 4 + ins[1:, :-1] * 8)
    ys, xs = np.nonzero((idx > 0) & (idx < 15))

    nxt = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        case = int(idx[y, x])
        edges = {"B": ("h", x, y), "T": ("h", x, y + 1),
                 "L": ("v", x, y), "R": ("v", x + 1, y)}
        if case in _SADDLE:
            centre = (f[y, x] + f[y, x + 1] + f[y + 1, x + 1] + f[y + 1, x]) * 0.25
            segs = _SADDLE[case][0 if centre >= thr else 1]
        else:
            segs = _SEGS[case]
        for s, e in segs:
            nxt[edges[s]] = edges[e]

    def pos(key):
        kind, x, y = key
        if kind == "h":
            f0, f1 = f[y, x], f[y, x + 1]
        else:
            f0, f1 = f[y, x], f[y + 1, x]
        d = f1 - f0
        t = 0.5 if abs(d) < 1e-9 else min(max((thr - f0) / d, 0.0), 1.0)
        # -1 removes the padding offset
        return (x - 1 + t, y - 1) if kind == "h" else (x - 1, y - 1 + t)

    loops = []
    while nxt:
        start, cur = nxt.popitem()
        loop = [start]
        while cur is not None and cur != start:
            loop.append(cur)
            cur = nxt.pop(cur, None)
        if len(loop) >= 3:
            loops.append([pos(k) for k in loop])
    return loops


def loop_area(pts):
    """Signed area (shoelace). Positive = counter-clockwise."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a * 0.5


def smooth_loop(pts, iterations):
    """Laplacian smoothing of a closed loop (softens pixel stair-steps)."""
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(iterations):
        p = 0.5 * p + 0.25 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
    return [tuple(v) for v in p]


def _rdp(p, eps):
    """Iterative Ramer-Douglas-Peucker on an open polyline (numpy Nx2)."""
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        seg = b - a
        seg_len = math.hypot(seg[0], seg[1])
        mid = p[i + 1:j]
        if seg_len < 1e-12:
            d = np.hypot(mid[:, 0] - a[0], mid[:, 1] - a[1])
        else:
            d = np.abs(seg[0] * (mid[:, 1] - a[1]) - seg[1] * (mid[:, 0] - a[0])) / seg_len
        m = int(np.argmax(d))
        if d[m] > eps:
            m += i + 1
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return p[keep]


def simplify_loop(pts, eps):
    """Reduce points on a closed loop, keeping it within ``eps`` pixels."""
    if eps <= 0 or len(pts) < 8:
        return pts
    p = np.asarray(pts, dtype=np.float64)
    far = int(np.argmax(np.hypot(p[:, 0] - p[0, 0], p[:, 1] - p[0, 1])))
    a = _rdp(p[:far + 1], eps)
    b = _rdp(np.vstack([p[far:], p[:1]]), eps)
    out = np.vstack([a[:-1], b[:-1]])
    return [tuple(v) for v in out] if len(out) >= 3 else pts


def build_outlines(field, thr, simplify, smooth, min_area):
    """Trace, filter, smooth and simplify. Returns a list of closed loops."""
    loops = []
    for lp in trace_contours(field, thr):
        if abs(loop_area(lp)) < min_area:
            continue
        if smooth > 0:
            lp = smooth_loop(lp, smooth)
        lp = simplify_loop(lp, simplify)
        if len(lp) >= 3:
            loops.append(lp)
    return loops
