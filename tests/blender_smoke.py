"""End-to-end test inside Blender (no UI needed):

    blender --background --factory-startup --python tests/blender_smoke.py

(Deliberately not named test_*.py: it needs Blender, so pytest must not collect it.)

Exits with code 1 if anything fails.
"""

import math
import os
import sys
import tempfile
import traceback

import bmesh
import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import png_cutout_solid  # noqa: E402

TMP = tempfile.mkdtemp(prefix="png_cutout_test_")
op = bpy.ops.import_image.png_cutout_solid
failures = []


def save_png(name, px):
    h, w = px.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    path = os.path.join(TMP, name + ".png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return path


def ring_and_square(w, h, scale=1.0):
    """Transparent background, a ring (has a hole) and a separate square."""
    yy, xx = np.mgrid[0:h, 0:w] / scale
    r = np.hypot(xx - 60, yy - 60)
    solid = ((r < 50) & (r > 25)) | ((xx > 130) & (xx < 180) & (yy > 30) & (yy < 90))
    px = np.zeros((h, w, 4))
    px[..., 0] = 1.0
    px[..., 3] = solid
    return px


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  " + detail))
    if not cond:
        failures.append(name)


def non_manifold_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def run(label, **kw):
    print("[%s]" % label)
    before = set(bpy.data.objects)
    try:
        result = op(**kw)
    except RuntimeError as e:  # raised whenever the operator reports an error
        print("  (reported: %s)" % str(e).strip())
        result = {"REPORTED_ERROR"}
    new = [o for o in bpy.data.objects if o not in before]
    return result, new


def main():
    png_cutout_solid.register()
    ring = save_png("ring", ring_and_square(200, 120))
    big = save_png("ring_big", ring_and_square(1000, 600, scale=5.0))
    empty = save_png("empty", np.zeros((50, 50, 4)))
    ink = np.ones((60, 60, 4))
    ink[15:45, 15:45, :3] = 0.0  # black square on opaque white
    ink = save_png("ink", ink)

    res, objs = run("default", filepath=ring)
    o = objs[0]
    check("finished", res == {"FINISHED"})
    check("one object named after the file", len(objs) == 1 and o.name == "ring", o.name)
    check("is a mesh", o.type == "MESH")
    check("size", math.isclose(o.dimensions.x, 1.69, abs_tol=0.01), str(o.dimensions))
    check("thickness", math.isclose(o.dimensions.z, 0.05, abs_tol=1e-4), str(o.dimensions))
    check("watertight", non_manifold_edges(o) == 0, str(non_manifold_edges(o)))
    check("has UVs", len(o.data.uv_layers) == 1)
    tex = [n for n in o.active_material.node_tree.nodes if n.type == "TEX_IMAGE"]
    check("material shows the image", tex and tex[0].image.name == "ring.png")
    me = o.data
    vco = me.vertices[me.loops[0].vertex_index].co  # image is 2.0 x 1.2 units
    u, v = me.uv_layers[0].data[0].uv
    check("uv maps to image", math.isclose(u, vco.x / 2.0 + 0.5, abs_tol=1e-4)
          and math.isclose(v, vco.y / 1.2 + 0.5, abs_tol=1e-4), "%s %s" % (tuple(vco), (u, v)))
    default_dims = o.dimensions.copy()

    res, objs = run("again: next name is .001", filepath=ring)
    check("second import named ring.001", objs[0].name == "ring.001", objs[0].name)

    res, objs = run("bevel keeps size and thickness", filepath=ring, bevel_depth=0.01)
    o = objs[0]
    check("same size as without bevel", (o.dimensions - default_dims).length < 1e-3,
          "%s vs %s" % (o.dimensions, default_dims))
    check("watertight", non_manifold_edges(o) == 0)

    res, objs = run("bevel larger than half the thickness", filepath=ring, bevel_depth=1.0)
    check("thickness unchanged", math.isclose(objs[0].dimensions.z, 0.05, abs_tol=1e-4),
          str(objs[0].dimensions))

    res, objs = run("thickness 0 = flat plane", filepath=ring, thickness=0.0)
    check("flat", objs[0].dimensions.z < 1e-6, str(objs[0].dimensions))

    res, objs = run("keep as curve", filepath=ring, convert_to_mesh=False)
    o = objs[0]
    check("is a curve", o.type == "CURVE")
    check("curve size", (o.dimensions - default_dims).length < 0.02, str(o.dimensions))
    nodes = o.active_material.node_tree.nodes
    check("curve material uses object mapping", any(n.type == "MAPPING" for n in nodes))

    res, objs = run("upright", filepath=ring, upright=True)
    check("rotated to stand up", math.isclose(objs[0].rotation_euler.x, math.pi / 2, abs_tol=1e-6))

    res, objs = run("no material", filepath=ring, add_material=False)
    check("no material slots", len(objs[0].data.materials) == 0)

    res, objs = run("downsampled big image", filepath=big, max_resolution=200)
    check("same proportions as small image",
          (objs[0].dimensions - default_dims).length < 0.03, str(objs[0].dimensions))

    res, objs = run("opaque image uses dark pixels", filepath=ink)
    check("one object", len(objs) == 1)
    check("square is half the image", objs and math.isclose(objs[0].dimensions.x, 1.0, abs_tol=0.04),
          objs and str(objs[0].dimensions))

    n_images = len(bpy.data.images)
    res, objs = run("empty image fails cleanly", filepath=empty)
    check("error reported", res == {"REPORTED_ERROR"})
    check("no objects left behind", objs == [])
    check("no unused image left behind", len(bpy.data.images) == n_images)

    res, objs = run("missing file", filepath=os.path.join(TMP, "nope.png"))
    check("error reported, nothing created", res == {"REPORTED_ERROR"} and objs == [])

    res, objs = run("several files at once", directory=TMP,
                    files=[{"name": "ring.png"}, {"name": "empty.png"}, {"name": "ink.png"}])
    check("bad file reported", res == {"REPORTED_ERROR"})
    check("good files still imported", len(objs) == 2, str(objs))
    check("new objects are selected", objs and all(o.select_get() for o in objs))

    png_cutout_solid.unregister()


try:
    main()
except Exception:
    traceback.print_exc()
    failures.append("exception")

print("\n%s (Blender %s)" % ("ALL OK" if not failures else "FAILED: %s" % ", ".join(failures),
                             bpy.app.version_string))
sys.exit(1 if failures else 0)
