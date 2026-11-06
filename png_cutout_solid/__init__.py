# SPDX-FileCopyrightText: 2026 Jennifer Freeman
# SPDX-License-Identifier: GPL-3.0-or-later

"""PNG Cutout Solid - turn an image into a solid, extruded shape cut to its outline.

Blender 4.2+ reads metadata from ``blender_manifest.toml``; ``bl_info`` below
is only used when installed as a legacy add-on in Blender 2.93 - 4.1.
"""

bl_info = {
    "name": "PNG Cutout Solid",
    "author": "Jennifer Freeman",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "File > Import > PNG as Cutout Solid  |  3D View > Sidebar (N) > Cutout",
    "description": "Turn a PNG into a solid, extruded shape cut to the image's outline "
                   "(transparent areas and holes are removed)",
    "category": "Import-Export",
}

if "outline" in locals():  # support "Reload Scripts"
    import importlib
    outline = importlib.reload(outline)  # noqa: F821
else:
    from . import outline

import math
import os
import traceback

import bmesh
import bpy
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, IntProperty, StringProperty)
from bpy_extras.io_utils import ImportHelper


class CutoutError(Exception):
    """A problem with one input file, reported to the user without a traceback."""


def image_pixels(img):
    """Image pixels as an ``(h, w, channels)`` float32 array, bottom row first."""
    w, h = img.size
    ch = img.channels
    px = np.empty(w * h * ch, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, ch)


def make_material(img, name, object_mapping=None):
    """Material that shows ``img`` on the faces.

    Meshes use their UVs. Curves have no UVs, so for them pass
    ``object_mapping=(scale_x, scale_y)`` to project object coordinates onto
    the image instead.
    """
    mat = bpy.data.materials.new(name)
    if not getattr(mat, "use_nodes", True):  # always on (and deprecated) in Blender 5.0+
        mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            out = nodes.new("ShaderNodeOutputMaterial")
            out.location = (300, 300)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-400, 300)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if object_mapping is not None:
        tc = nodes.new("ShaderNodeTexCoord")
        tc.location = (-900, 300)
        mp = nodes.new("ShaderNodeMapping")
        mp.location = (-700, 300)
        mp.inputs["Scale"].default_value = (object_mapping[0], object_mapping[1], 1.0)
        mp.inputs["Location"].default_value = (0.5, 0.5, 0.0)
        links.new(tc.outputs["Object"], mp.inputs["Vector"])
        links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    return mat


def create_cutout(context, filepath, op):
    """Import one image. Returns ``(object, mask_source_used, outline_count)``."""
    if not os.path.isfile(filepath):
        raise CutoutError("File not found: %s" % filepath)
    existing = {i.as_pointer() for i in bpy.data.images}
    try:
        img = bpy.data.images.load(filepath, check_existing=True)
    except RuntimeError:
        raise CutoutError("Could not read image: %s" % os.path.basename(filepath))
    try:
        return _build(context, img, filepath, op)
    finally:
        # don't leave an unused image behind (failed import, or no material)
        if img.as_pointer() not in existing and img.users == 0:
            bpy.data.images.remove(img)


def _build(context, img, filepath, op):
    w, h = img.size
    if w == 0 or h == 0:
        raise CutoutError("Could not read image: %s" % os.path.basename(filepath))

    field, used, k = outline.pixels_to_field(image_pixels(img), op.mask_source,
                                             op.invert, op.max_resolution)
    loops = outline.build_outlines(field, op.threshold, op.simplify,
                                   op.smooth_iterations, op.min_area)
    if not loops:
        raise CutoutError("No solid area found in '%s' (mask: %s). Try a different "
                          "Threshold, Invert, or Mask setting."
                          % (os.path.basename(filepath), used))

    s = op.size / float(max(w, h))  # world units per image pixel
    name = os.path.splitext(os.path.basename(filepath))[0]

    # The bevel is carved out of the shape, so Size and Thickness stay exact.
    bevel = min(op.bevel_depth, op.thickness * 0.5)

    # --- filled 2D curve: Blender fills it and cuts the holes for us
    cu = bpy.data.curves.new(name + "_outline", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = op.thickness * 0.5 - bevel
    cu.bevel_depth = bevel
    cu.bevel_resolution = op.bevel_resolution
    cu.offset = -bevel
    for lp in loops:
        sp = cu.splines.new("POLY")
        sp.points.add(len(lp) - 1)
        co = []
        for x, y in lp:  # field px -> image px -> world, centred on the image
            co.extend((((x + 0.5) * k - w * 0.5) * s, ((y + 0.5) * k - h * 0.5) * s, 0.0, 1.0))
        sp.points.foreach_set("co", co)
        sp.use_cyclic_u = True

    obj = bpy.data.objects.new(name, cu)
    context.collection.objects.link(obj)

    if op.convert_to_mesh:
        dg = context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        # remove the curve first so the mesh object gets the clean name
        bpy.data.objects.remove(obj)
        bpy.data.curves.remove(cu)
        me.name = name

        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=s * 1e-3)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        # planar UVs that line up with the original image
        uv = bm.loops.layers.uv.verify()
        sx, sy = w * s, h * s
        for face in bm.faces:
            for loop in face.loops:
                loop[uv].uv = (loop.vert.co.x / sx + 0.5, loop.vert.co.y / sy + 0.5)
        bm.to_mesh(me)
        bm.free()

        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)

    if op.add_material:
        mapping = None if op.convert_to_mesh else (1.0 / (w * s), 1.0 / (h * s))
        obj.data.materials.append(make_material(img, name + "_mat", mapping))

    if op.upright:
        obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
    obj.location = context.scene.cursor.location
    return obj, used, len(loops)


_MASK_ITEMS = [
    ("AUTO", "Auto", "Use transparency if the image has any, otherwise dark pixels"),
    ("ALPHA", "Transparency", "Transparent pixels are cut away"),
    ("LUMINANCE", "Dark = Solid", "Dark pixels are kept, light/white pixels are cut away"),
]


class IMPORT_IMAGE_OT_png_cutout_solid(bpy.types.Operator, ImportHelper):
    """Create a solid, extruded shape cut to the outline of an image's opaque area"""
    bl_idname = "import_image.png_cutout_solid"
    bl_label = "PNG as Cutout Solid"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.tif;*.tiff;*.webp",
                                options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    mask_source: EnumProperty(name="Mask", items=_MASK_ITEMS, default="AUTO",
                              description="What decides which parts of the image are solid")
    threshold: FloatProperty(name="Threshold", default=0.5, min=0.001, max=0.999,
                             description="Cut-off between solid and empty")
    invert: BoolProperty(name="Invert", default=False,
                         description="Swap solid and empty areas")
    size: FloatProperty(name="Size", default=2.0, min=0.0001, unit="LENGTH",
                        description="Length of the image's longest side")
    thickness: FloatProperty(name="Thickness", default=0.05, min=0.0, unit="LENGTH",
                             description="Total thickness (0 = flat plane)")
    bevel_depth: FloatProperty(name="Edge Bevel", default=0.0, min=0.0, unit="LENGTH",
                               description="Round the edges (0 = sharp). "
                                           "Limited to half the thickness")
    bevel_resolution: IntProperty(name="Bevel Segments", default=2, min=0, max=16,
                                  description="Smoothness of the rounded edge")
    simplify: FloatProperty(name="Simplify (px)", default=0.5, min=0.0, max=20.0,
                            description="Outline tolerance in pixels (0 = keep every point)")
    smooth_iterations: IntProperty(name="Smooth", default=0, min=0, max=50,
                                   description="Smooth jagged/pixel edges before simplifying")
    min_area: FloatProperty(name="Remove Specks (px²)", default=4.0, min=0.0,
                            description="Ignore islands and holes smaller than this many pixels")
    max_resolution: IntProperty(name="Max Resolution", default=1024, min=16, max=16384,
                                description="Larger images are downsampled to this before "
                                            "tracing (higher = more detail, slower)")
    convert_to_mesh: BoolProperty(name="Convert to Mesh", default=True,
                                  description="Off = keep an editable curve object")
    add_material: BoolProperty(name="Image Material", default=True,
                               description="Put the image on the faces")
    upright: BoolProperty(name="Stand Upright", default=False,
                          description="Stand it up facing -Y (front view) instead of lying flat")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        box = layout.box()
        box.label(text="Cut-out")
        box.prop(self, "mask_source")
        box.prop(self, "threshold")
        box.prop(self, "invert")
        box = layout.box()
        box.label(text="Shape")
        box.prop(self, "size")
        box.prop(self, "thickness")
        box.prop(self, "bevel_depth")
        box.prop(self, "bevel_resolution")
        box = layout.box()
        box.label(text="Outline Quality")
        box.prop(self, "simplify")
        box.prop(self, "smooth_iterations")
        box.prop(self, "min_area")
        box.prop(self, "max_resolution")
        box = layout.box()
        box.label(text="Output")
        box.prop(self, "convert_to_mesh")
        box.prop(self, "add_material")
        box.prop(self, "upright")

    def execute(self, context):
        paths = []
        if self.files and self.directory:
            paths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        if not paths:
            paths = [self.filepath]

        if context.mode != "OBJECT" and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        made = []
        for i, path in enumerate(paths):
            try:
                obj, used, n = create_cutout(context, path, self)
            except CutoutError as e:  # keep going with the other files
                self.report({"ERROR"}, str(e))
                continue
            except Exception as e:
                traceback.print_exc()
                self.report({"ERROR"}, "%s: unexpected error: %s (details in the system console)"
                            % (os.path.basename(path), e))
                continue
            obj.location.x += i * self.size * 1.1
            made.append(obj)
            self.report({"INFO"}, "%s: %d outline(s), mask = %s" % (obj.name, n, used))
        if not made:
            return {"CANCELLED"}

        for o in context.selected_objects:
            o.select_set(False)
        for o in made:
            o.select_set(True)
        context.view_layer.objects.active = made[-1]
        return {"FINISHED"}


class VIEW3D_PT_png_cutout_solid(bpy.types.Panel):
    bl_label = "PNG Cutout Solid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Cutout"

    def draw(self, context):
        col = self.layout.column()
        col.operator(IMPORT_IMAGE_OT_png_cutout_solid.bl_idname, text="Import PNG as Solid",
                     icon="IMAGE_DATA")
        col.label(text="Tweak settings in the file browser")
        col.label(text="or the Adjust Last Operation panel.")


def _menu_import(self, context):
    self.layout.operator(IMPORT_IMAGE_OT_png_cutout_solid.bl_idname,
                         text="PNG as Cutout Solid", icon="IMAGE_DATA")


_classes = (IMPORT_IMAGE_OT_png_cutout_solid, VIEW3D_PT_png_cutout_solid)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
