# PNG Cutout Solid

A Blender add-on that turns an image into a **solid 3D shape cut to the image's outline**.
The transparent background is cut away, holes are cut out (the inside of an "O", a donut),
and what's left is extruded to the thickness you choose, with the image on its faces.

Use it for logos, stickers, signs, game props, cookie-cutter style shapes, or 3D-printable
badges and keychains.

- Works with transparent PNGs, or with dark-on-white images (scans, JPGs, line art)
- Keeps holes and separate pieces
- Mesh output is watertight (good for 3D printing), with UVs lined up to the image
- Optional rounded edges that don't change the size or thickness
- Import several images in one go
- Or keep the result as an editable curve instead of a mesh

## Requirements

Blender **2.93 or newer**. Tested on Blender 3.5, 4.4 and 5.2.

## Installation

1. Download `png_cutout_solid-x.y.z.zip` from the [latest release](../../releases/latest).
   **Don't unzip it.**
2. Install it in Blender:
   - **Blender 4.2 and newer:** drag the zip into the Blender window, *or* go to
     **Edit → Preferences → Get Extensions**, click the **⌄** menu in the top-right corner, choose
     **Install from Disk…** and pick the zip.
   - **Blender 2.93 – 4.1:** go to **Edit → Preferences → Add-ons → Install…**, pick the zip, then tick
     the checkbox next to **Import-Export: PNG Cutout Solid**.

## Usage

1. **File → Import → PNG as Cutout Solid**
   (or the **Cutout** tab in the 3D Viewport sidebar: press **N**).
2. Pick one or more images. The settings are in the right-hand panel of the file browser.
3. The new object appears at the 3D cursor. To change settings afterwards, open the
   **Adjust Last Operation** panel (bottom-left of the viewport, or press **F9**).

Supported formats: PNG, JPG, TGA, BMP, TIFF and WebP (WebP needs Blender 3.4+).

### Settings

| Setting | What it does |
| --- | --- |
| **Mask** | What counts as solid. *Auto* uses transparency if the image has any, otherwise dark pixels. *Transparency*: opaque pixels are kept. *Dark = Solid*: dark pixels are kept and white is cut away. |
| **Threshold** | Cut-off between solid and empty (0–1). Lower keeps more of soft or semi-transparent edges. |
| **Invert** | Swap solid and empty, e.g. for a white logo on a black background. |
| **Size** | Length of the image's longest side, in scene units. This is the whole image including any transparent margin, not just the visible shape. |
| **Thickness** | Total thickness. `0` makes a flat cut-out plane. |
| **Edge Bevel** / **Bevel Segments** | Rounds the edges. The overall size and thickness stay the same; the bevel is limited to half the thickness. |
| **Simplify (px)** | How far (in pixels) the outline may stray from the image to save vertices. `0` keeps every point. |
| **Smooth** | Smooths jagged, pixel-stepped edges before simplifying. Try 2–5 for low-resolution images. |
| **Remove Specks (px²)** | Ignores tiny islands and holes smaller than this many pixels (stray dots, noise). |
| **Max Resolution** | Images larger than this are scaled down before the outline is traced. Raise it to keep more detail (slower). |
| **Convert to Mesh** | Off = keep a 2D curve object, so you can edit the outline or change the extrude/bevel later. |
| **Image Material** | Creates a material with the image on the faces. |
| **Stand Upright** | Stands the shape up facing the front view (−Y) instead of lying flat. |

Use the **preset menu** at the top of the settings to save your favourite combinations.

### Tips

- **Nothing was imported / "No solid area found":** the image probably has no transparency and is
  mostly light. Try **Invert**, or set **Mask** explicitly.
- **Edges look like stairs:** raise **Smooth**, or use a higher-resolution image.
- **Too many vertices:** raise **Simplify**. Lots of little bits: raise **Remove Specks**.
- **Thin details missing on a large image:** raise **Max Resolution**.

## Development

The tracing code (`png_cutout_solid/outline.py`) is plain Python + numpy, so it can be tested
without Blender:

```sh
pip install numpy pytest
pytest
```

End-to-end test inside Blender (creates test images, imports them, checks the results):

```sh
blender --background --factory-startup --python tests/blender_smoke.py
```

Build the installable zip (Python 3.11+) into `dist/`:

```sh
python build.py
```

### Releasing

1. Bump the version in **both** `png_cutout_solid/blender_manifest.toml` and `bl_info` in
   `png_cutout_solid/__init__.py` (a test checks they match).
2. Commit, then tag and push: `git tag v1.0.1 && git push origin v1.0.1`.
3. GitHub Actions runs the tests, builds the zip and publishes a release with it attached.

## License

[GPL-3.0-or-later](LICENSE), like Blender itself.
