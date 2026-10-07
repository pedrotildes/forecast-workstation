# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Forecast Workstation (macOS .app / Windows one-folder)."""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = [("../frontend", "frontend")]
binaries = []
hiddenimports = ["app.main", "app.workers", "encodings.idna"]

# ecCodes' native library ships in the separate `eccodeslib` wheel (GRIB definitions
# are embedded in it). Bundle the whole tree so findlibs locates it at runtime.
try:
    import eccodeslib  # noqa: E402

    datas += [(os.path.dirname(eccodeslib.__file__), "eccodeslib")]
except ImportError:
    print("[spec] eccodeslib not found — GRIB decoding will fail in the bundle")


def add_all(pkg):
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({pkg}) skipped: {exc}")


for pkg in ["eccodes", "gribapi", "findlibs", "metpy", "pint", "cartopy", "pyproj",
            "shapely", "h5py", "webview"]:
    add_all(pkg)

datas += collect_data_files("certifi")
hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["httpx", "httpcore", "anyio", "h11", "sniffio", "scipy.ndimage",
                  "scipy.interpolate", "matplotlib.backends.backend_agg"]

a = Analysis(
    ["desktop.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PySide2", "PySide6", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ForecastWorkstation",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ForecastWorkstation")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Forecast Workstation.app",
        icon=None,
        bundle_identifier="com.pedrotildes.forecast",
        info_plist={
            "CFBundleShortVersionString": "1.1.0",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.weather",
        },
    )
