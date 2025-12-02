# -*- mode: python ; coding: utf-8 -*-
import glob
import pathlib
from PyInstaller.utils.hooks import collect_all, collect_data_files

# =============================================
# Dynamic Paths
# =============================================
current_path = pathlib.Path().absolute()
print('Current Path', current_path)
code_path = pathlib.Path(current_path, 'code')

datas = []

# =============================================
# 1. Include Assets Folder (code\src\assets)
# =============================================
dirs = ['code\\src\\assets']
for _dir in dirs:
    dir_path = pathlib.Path(pathlib.Path(), _dir)
    all_files = glob.glob(f"{dir_path}/**/*", recursive=True)
    for file in all_files:
        rel_path = str(pathlib.Path(file).relative_to(dir_path.parent.parent.parent).as_posix())
        print('Adding file', rel_path)
        src_path = str(dir_path).replace("code\\", "")
        datas.append((rel_path, src_path))

# =============================================
# 2. Include .ini and .sql files
# =============================================
dirs = ['code/src']
file_extensions = ['*.ini', '*.sql']

for _dir in dirs:
    dir_path = pathlib.Path(pathlib.Path(), _dir)
    for ext in file_extensions:
        all_files = glob.glob(f"{dir_path}/**/{ext}", recursive=True)
        for file in all_files:
            file_path = pathlib.Path(file).absolute()
            dest_path = file_path.relative_to(code_path)
            print('Adding file', file, dest_path.parent)
            datas.append((str(file_path), str(dest_path.parent)))

# =============================================
# 3. Collect Required Libraries
# =============================================

# NiceGUI
ngdata, ngbins, ngimps = collect_all('nicegui')
datas += ngdata

# Celery
celerydata, celerybins, celeryimps = collect_all('celery')
datas += celerydata

# XGBoost and imblearn
datas += collect_data_files('xgboost')
datas += collect_data_files('imblearn')

# Pandas, NumPy, Dask
pandas_data, pandas_bins, pandas_imps = collect_all('pandas')
numpy_data, numpy_bins, numpy_imps = collect_all('numpy')
dask_data, dask_bins, dask_imps = collect_all('dask')

# PyArrow
pyarrow_data, pyarrow_bins, pyarrow_imps = collect_all('pyarrow')

# Combine all
datas += pandas_data + numpy_data + dask_data + pyarrow_data
binaries = ngbins + celerybins + pandas_bins + numpy_bins + dask_bins + pyarrow_bins

hiddenimports = (
    [
        'kombu.transport.pyamqp',
        'pandas',
        'numpy',
        'dask',
        'pyarrow',
        'dask.dataframe',
        'dask.array',
        'dask.delayed',
        'dask.config',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.timestamps',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.timezones',
        'pandas._libs.window.aggregations',
    ]
    + ngimps
    + celeryimps
    + pandas_imps
    + numpy_imps
    + dask_imps
    + pyarrow_imps
)

# =============================================
# 4. Ensure pandas DLL (aggregations.pyd) is Bundled
# =============================================
import os, pandas
from pathlib import Path

pandas_window_dir = Path(pandas.__path__[0]) / "_libs" / "window"
aggregations_pyd = next(pandas_window_dir.glob("aggregations*.pyd"), None)

if aggregations_pyd and aggregations_pyd.exists():
    print(f"Adding missing pandas DLL: {aggregations_pyd}")
    binaries.append((str(aggregations_pyd), "pandas\\_libs\\window"))
else:
    print("Warning: aggregations.pyd not found in", pandas_window_dir)

# =============================================
# 5. Build Analysis
# =============================================
a = Analysis(
    ['code\\src\\setup_service.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# =============================================
# 6. Build Executable
# =============================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KonaAI Intelligence Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set False if you don't want console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    singlefile=True,
    icon=str(pathlib.Path(pathlib.Path(), 'code', 'src', 'assets', 'konaai.ico').absolute())
)