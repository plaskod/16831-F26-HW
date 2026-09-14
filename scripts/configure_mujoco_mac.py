"""Configure the legacy MuJoCo binding inside this workspace's virtualenv."""
from pathlib import Path
import shutil
import subprocess
import sysconfig

root = Path(__file__).resolve().parents[1]
site = Path(sysconfig.get_paths()['purelib'])
runtime = root / '.local/mujoco/mujoco210'
source = Path('/private/tmp/rob831-mujoco211/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/A')
(runtime / 'bin').mkdir(parents=True, exist_ok=True)
if source.exists():
    shutil.copytree(source / 'Headers', runtime / 'include', dirs_exist_ok=True)
    shutil.copy2(source / 'libmujoco.2.1.1.dylib', runtime / 'bin/libmujoco210.dylib')
    shutil.copy2(site / 'glfw/libglfw.3.dylib', runtime / 'bin/libglfw.3.dylib')
for name in ('libmujoco210.dylib', 'libglfw.3.dylib'):
    lib = runtime / 'bin' / name
    subprocess.run(['install_name_tool', '-id', str(lib), str(lib)], check=True)
    subprocess.run(['codesign', '--force', '--sign', '-', str(lib)], check=True)

builder = site / 'mujoco_py/builder.py'
text = builder.read_text()
# Apple clang lacks -fopenmp. This binding imports prange but never uses it.
# Limit the build adjustment to macOS; the simulation implementation is unchanged.
text = text.replace("'-fopenmp',  # needed for OpenMP", "*([] if sys.platform == 'darwin' else ['-fopenmp']),  # Apple clang")
text = text.replace("extra_link_args=['-fopenmp'],", "extra_link_args=[] if sys.platform == 'darwin' else ['-fopenmp'],")
builder.write_text(text)
# Works in the terminal and Jupyter without changing global shell settings.
(site / 'rob831_local_runtime.pth').write_text(
    'import os; os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", ' + repr(str(runtime)) + '); '
    'os.environ.setdefault("PYGLFW_LIBRARY", ' + repr(str(runtime / 'bin/libglfw.3.dylib')) + '); '
    'os.environ.setdefault("CC", "/usr/bin/clang"); '
    'os.environ.setdefault("MPLCONFIGDIR", ' + repr(str(root / '.local/matplotlib')) + ')\n'
)
print('Configured workspace MuJoCo runtime:', runtime)
