"""Unit tests for the image -> outline code. Runs without Blender:  pytest"""

import ast
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1] / "png_cutout_solid"
sys.path.insert(0, str(PKG))  # import outline.py directly; __init__.py needs bpy

import outline  # noqa: E402


def disk(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(xx - cx, yy - cy) < r


def rgba(mask):
    px = np.zeros(mask.shape + (4,), np.float32)
    px[..., :3] = 1.0
    px[..., 3] = mask
    return px


# --- tracing ---------------------------------------------------------------

def test_empty_field_has_no_outlines():
    assert outline.trace_contours(np.zeros((10, 10), np.float32)) == []


def test_square_is_one_counter_clockwise_loop_of_the_right_size():
    f = np.zeros((20, 20), np.float32)
    f[5:15, 5:15] = 1.0  # 10x10 px
    loops = outline.trace_contours(f)
    assert len(loops) == 1
    area = outline.loop_area(loops[0])
    assert area > 0  # outer outlines run counter-clockwise
    assert area == pytest.approx(100.0, abs=1.0)  # minus the corner cut-offs


def test_image_filling_the_whole_frame_is_closed():
    loops = outline.trace_contours(np.ones((8, 8), np.float32))
    assert len(loops) == 1
    assert outline.loop_area(loops[0]) == pytest.approx(64.0, abs=1.0)


def test_ring_gives_outer_loop_and_clockwise_hole():
    f = (disk(60, 60, 30, 30, 25) & ~disk(60, 60, 30, 30, 10)).astype(np.float32)
    areas = sorted(outline.loop_area(lp) for lp in outline.trace_contours(f))
    assert len(areas) == 2
    assert areas[0] < 0 < areas[1]  # hole is clockwise
    assert areas[1] == pytest.approx(np.pi * 25 ** 2, rel=0.03)
    assert -areas[0] == pytest.approx(np.pi * 10 ** 2, rel=0.06)


def test_separate_islands_are_separate_loops():
    f = np.zeros((10, 30), np.float32)
    f[2:8, 2:8] = 1.0
    f[2:8, 12:18] = 1.0
    f[2:8, 22:28] = 1.0
    assert len(outline.trace_contours(f)) == 3


@pytest.mark.parametrize("centre", [0.0, 1.0])
def test_diagonal_saddle_produces_closed_loops(centre):
    f = np.array([[1, 0], [0, 1]], np.float32)  # pixels touching only at a corner
    # make the saddle's centre value land on either side of the threshold
    f = np.kron(f, np.ones((3, 3), np.float32))
    f[2:4, 2:4] = centre
    loops = outline.trace_contours(f)
    assert loops and all(len(lp) >= 3 for lp in loops)


def test_outline_points_are_subpixel_interpolated():
    f = np.zeros((5, 5), np.float32)
    f[1:4, 1:4] = 1.0
    f[1:4, 3] = 0.75  # right column partly solid
    xs = [x for x, _ in outline.trace_contours(f)[0]]
    # threshold 0.5 between 0.75 (col 3) and 0 (col 4): 1/3 of the way
    assert max(xs) == pytest.approx(3 + (0.75 - 0.5) / 0.75)


# --- filtering / simplifying ----------------------------------------------

def test_min_area_removes_specks_but_keeps_shapes():
    f = np.zeros((40, 40), np.float32)
    f[5:30, 5:30] = 1.0
    f[35, 35] = 1.0  # one stray pixel
    assert len(outline.build_outlines(f, 0.5, 0.0, 0, 0.0)) == 2
    assert len(outline.build_outlines(f, 0.5, 0.0, 0, 4.0)) == 1


def test_simplify_reduces_points_and_keeps_shape():
    f = disk(200, 200, 100, 100, 80).astype(np.float32)
    raw = outline.build_outlines(f, 0.5, 0.0, 0, 0.0)[0]
    simple = outline.build_outlines(f, 0.5, 1.0, 0, 0.0)[0]
    assert len(simple) < len(raw) / 3
    assert outline.loop_area(simple) == pytest.approx(outline.loop_area(raw), rel=0.01)


def test_smoothing_keeps_loop_closed_and_similar():
    f = disk(50, 50, 25, 25, 18).astype(np.float32)
    raw = outline.build_outlines(f, 0.5, 0.0, 0, 0.0)[0]
    smooth = outline.build_outlines(f, 0.5, 0.0, 5, 0.0)[0]
    assert len(smooth) == len(raw)
    assert outline.loop_area(smooth) == pytest.approx(outline.loop_area(raw), rel=0.05)


# --- pixels -> field ------------------------------------------------------

def test_auto_uses_alpha_when_image_has_transparency():
    px = rgba(disk(20, 20, 10, 10, 5))
    field, used, k = outline.pixels_to_field(px, "AUTO")
    assert used == "ALPHA" and k == 1
    assert field[10, 10] == 1.0 and field[0, 0] == 0.0


def test_auto_uses_dark_pixels_for_opaque_images():
    px = np.ones((20, 20, 4), np.float32)  # white, fully opaque
    px[5:15, 5:15, :3] = 0.0                # black square
    field, used, _ = outline.pixels_to_field(px, "AUTO")
    assert used == "LUMINANCE"
    assert field[10, 10] == pytest.approx(1.0) and field[0, 0] == pytest.approx(0.0)


def test_invert_swaps_solid_and_empty():
    px = rgba(disk(20, 20, 10, 10, 5))
    field, _, _ = outline.pixels_to_field(px, "ALPHA", invert=True)
    assert field[10, 10] == 0.0 and field[0, 0] == 1.0


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_fewer_channels_are_supported(channels):
    px = np.zeros((10, 10, channels), np.float32)
    field, _, _ = outline.pixels_to_field(px, "AUTO")
    assert field.shape == (10, 10)


def test_large_images_are_downsampled():
    px = rgba(np.ones((300, 500), bool))
    field, _, k = outline.pixels_to_field(px, "ALPHA", max_res=100)
    assert k == 5
    assert field.shape == (60, 100)


# --- packaging -------------------------------------------------------------

def test_manifest_and_bl_info_versions_match():
    manifest = tomllib.loads((PKG / "blender_manifest.toml").read_text("utf-8"))
    tree = ast.parse((PKG / "__init__.py").read_text("utf-8"))
    bl_info = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and n.targets[0].id == "bl_info")
    assert ".".join(map(str, bl_info["version"])) == manifest["version"]
    assert bl_info["name"] == manifest["name"]
