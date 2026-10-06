# Third-party components in the portable application

HearSafe's original code is MIT licensed; see `LICENSE`. The included YAMNet
model has separate upstream notices in `models/yamnet/`. Model source and pinned
download provenance are recorded in its manifest. ESC-50 audio and research
weights are not included in this public bundle.

The application includes Python and its standard library (including Tkinter),
Tcl/Tk, NumPy, SciPy, SoundDevice and PortAudio, SoundFile and libsndfile, CFFI,
SoXR, ONNX Runtime, LiteRT, and their runtime dependencies. These components
retain their own licenses. `licenses/inventory.json` records the installed
distribution versions and copied notice files. The build preserves license,
notice, and copying files from the installed wheels under `licenses/`, and
Python/Tcl/Tk license files when supplied by the build interpreter. Some wheels
keep their bundled native library notices within their own license files.

PyInstaller supplies the application's executable bootloader. Its licensing
exception permits distributing applications under the application's own
license. See <https://pyinstaller.org/en/stable/license.html>.

Upstream component references:

- Python: <https://docs.python.org/3/license.html>
- Tcl/Tk: <https://www.tcl-lang.org/software/tcltk/license.html>
- NumPy: <https://numpy.org/doc/stable/license.html>
- SciPy: <https://scipy.org/about/>
- SoundDevice: <https://github.com/spatialaudio/python-sounddevice>
- PortAudio: <https://www.portaudio.com/license.html>
- SoundFile: <https://github.com/bastibe/python-soundfile>
- libsndfile: <https://libsndfile.github.io/libsndfile/>
- CFFI: <https://cffi.readthedocs.io/>
- SoXR: <https://github.com/dofuuz/python-soxr>
- ONNX Runtime: <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>
- LiteRT: <https://github.com/google-ai-edge/LiteRT/blob/main/LICENSE>

Review `licenses/` and the model notices before redistributing a modified
bundle, because dependency and model choices can change those obligations.
