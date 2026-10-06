# Build both launchers in one folder so native runtime libraries are shared.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(os.environ["HEARSAFE_BUILD_ROOT"])
binaries = []
datas = []
for package in ("ai_edge_litert", "onnxruntime", "soxr", "_soundfile_data"):
    package_binaries = collect_dynamic_libs(package)
    if package == "ai_edge_litert":
        # The interpreter uses CPU; optional WebGPU/OpenVINO providers add
        # unused downloads and require extra hardware/runtime support.
        package_binaries = [
            item for item in package_binaries
            if "WebGpu" not in item[0] and "vendors" not in Path(item[0]).parts
        ]
    binaries += package_binaries
    datas += collect_data_files(package, includes=["**/*.json", "**/*.txt", "**/*.csv", "**/LICENSE*"])

hiddenimports = [
    "ai_edge_litert.interpreter", "onnxruntime.capi._pybind_state",
    "sounddevice", "_sounddevice_data", "_soundfile_data", "tkinter", "tkinter.ttk",
]
excludes = [
    "hearsafe.training", "torch", "torchvision", "torchaudio", "onnx", "onnxscript",
    "tensorflow", "tensorboard", "pytest", "IPython", "matplotlib",
]

def analyse(entry):
    return Analysis(
        [str(root / "packaging" / entry)], pathex=[str(root / "src")],
        binaries=binaries, datas=datas, hiddenimports=hiddenimports,
        hookspath=[str(root / "packaging" / "hooks")], hooksconfig={}, runtime_hooks=[], excludes=excludes,
        noarchive=False,
    )

gui = analyse("gui_entry.py")
cli = analyse("cli_entry.py")
gui_exe = EXE(
    PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="HearSafe",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
cli_exe = EXE(
    PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="hearsafe-cli",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=True,
)
COLLECT(
    gui_exe, cli_exe, gui.binaries, cli.binaries, gui.datas, cli.datas,
    strip=False, upx=False, name="HearSafe",
)
