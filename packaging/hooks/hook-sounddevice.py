"""Include the x64 standard PortAudio backend; ASIO is outside this release."""

from pathlib import Path

from PyInstaller.utils.hooks import get_module_file_attribute

directory = (
    Path(get_module_file_attribute("sounddevice")).parent
    / "_sounddevice_data"
    / "portaudio-binaries"
)
library = directory / "libportaudio64bit.dll"
if not library.is_file():
    raise RuntimeError(f"Windows x64 PortAudio wheel library is missing: {library}")
binaries = [(str(library), "_sounddevice_data/portaudio-binaries")]
datas = []
readme = directory / "README.md"
if readme.is_file():
    datas.append((str(readme), "_sounddevice_data/portaudio-binaries"))
