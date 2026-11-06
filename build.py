"""Build the installable add-on zip:  python build.py  ->  dist/png_cutout_solid-<version>.zip

The zip holds a single ``png_cutout_solid/`` folder, which installs both as a
Blender 4.2+ extension and as a legacy add-on in Blender 2.93 - 4.1.
"""

import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "png_cutout_solid"


def main():
    version = tomllib.loads((PKG / "blender_manifest.toml").read_text("utf-8"))["version"]
    out = ROOT / "dist" / ("png_cutout_solid-%s.zip" % version)
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(PKG.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                zf.write(f, f.relative_to(ROOT).as_posix())
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
