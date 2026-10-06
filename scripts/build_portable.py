"""Build and smoke-test the Windows portable distribution.

Run with the project's Python 3.12 environment. A model package must have been
downloaded first; this script does not download anything during the build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPORTS = (
    "prototype_results.md",
    "yamnet_benchmark.json",
    "yamnet_file_smoke.json",
    "esc50_benchmark.json",
    "esc50_training.json",
    "esc50_evaluation.json",
    "esc50_evaluation.md",
    "esc50_model_manifest.json",
    "portable_smoke.json",
)
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "sounddevice",
    "soundfile",
    "cffi",
    "pycparser",
    "soxr",
    "onnxruntime",
    "ai-edge-litert",
    "flatbuffers",
    "psutil",
    "packaging",
    "protobuf",
    "coloredlogs",
    "humanfriendly",
    "sympy",
    "mpmath",
    "ml-dtypes",
    "pyinstaller",
    "setuptools",
)


def copy_public_reports(bundle: Path) -> None:
    """Copy reviewed reports only; never collect private report files by glob."""
    for filename in PUBLIC_REPORTS:
        source = ROOT / "reports" / filename
        if source.is_file():
            target = bundle / "reports" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def deterministic_zip(source: Path, destination: Path) -> None:
    """Use a stable order and ZIP metadata; executable content can still vary."""
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(source.rglob("*"), key=lambda p: p.as_posix()):
                if not path.is_file():
                    continue
                if path.is_symlink():
                    raise ValueError(f"Symlinks are not allowed in a release: {path}")
                info = zipfile.ZipInfo(
                    (Path(source.name) / path.relative_to(source)).as_posix(),
                    date_time=(2026, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_dependency_notices(destination: Path) -> None:
    """Preserve installed wheels' notices instead of guessing their contents."""
    destination.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        copied: list[str] = []
        for relative in distribution.files or ():
            if not any(word in relative.name.lower() for word in ("license", "copying", "notice")):
                continue
            original = Path(distribution.locate_file(relative))
            if not original.is_file():
                continue
            # Wheel RECORD paths can contain '..'; construct our own safe names.
            components = [part for part in relative.parts if part not in (".", "..")]
            target = destination / name / Path(*components)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            copied.append(target.relative_to(destination).as_posix())
        inventory.append(
            {
                "distribution": name,
                "version": distribution.version,
                "license_expression": distribution.metadata.get("License-Expression"),
                "license": distribution.metadata.get("License"),
                "notice_files": copied,
            }
        )
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copy2(python_license, destination / "PYTHON-LICENSE.txt")
    for tcl_license in sorted((Path(sys.base_prefix) / "tcl").glob("*/license*")):
        if tcl_license.is_file():
            shutil.copy2(tcl_license, destination / f"{tcl_license.parent.name}-LICENSE.txt")
    (destination / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_model(model_dir: Path) -> None:
    if not (model_dir / "model.tflite").is_file():
        raise ValueError("The portable release requires a YAMNet model.tflite package.")
    if any(model_dir.rglob("*.onnx")) or any(model_dir.rglob("*.pt")):
        raise ValueError("Research weights must not be bundled in the public release.")
    if not (model_dir / "manifest.json").is_file():
        raise ValueError(f"Missing model manifest: {model_dir / 'manifest.json'}")
    if not any(p.is_file() and "license" in p.name.lower() for p in model_dir.rglob("*")):
        raise ValueError("The model package must include its license notice.")
    # The loader checks manifest compatibility, labels, and model checksums and
    # also proves that the build interpreter can load the native inference engine.
    from hearsafe.models import load_model

    load_model(model_dir)


def smoke_portable(folder: Path, research_model: Path | None = None) -> dict[str, object]:
    """Exercise a copied bundle with no source-tree or model-dir environment."""
    folder = folder.resolve()
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "HEARSAFE_MODEL_DIR", "VIRTUAL_ENV"):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="HearSafe portable smoke ") as temp:
        moved = Path(temp) / "Moved HearSafe folder"
        shutil.copytree(folder, moved)
        executable = moved / "hearsafe-cli.exe"
        if not executable.is_file():
            raise ValueError(f"Missing portable CLI: {executable}")

        def run(*arguments: str) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                [str(executable), *arguments],
                cwd=temp,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Portable command {arguments!r} failed:\n{result.stderr}\n{result.stdout}"
                )
            return result

        help_result = run("--help")
        if "detect" not in help_result.stdout:
            raise RuntimeError("Portable CLI did not provide the expected help.")
        device_result = run("devices", "--json")
        listed_devices = json.loads(device_result.stdout)
        if not isinstance(listed_devices, list):
            raise RuntimeError("Portable devices command did not return a JSON list.")
        audio_file = Path(temp) / "silence sample.wav"
        with wave.open(str(audio_file), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x00" * 32000)
        detection = run("detect", str(audio_file), "--json")
        frames = [json.loads(line) for line in detection.stdout.splitlines() if line.strip()]
        if not frames or any(frame.get("schema_version") != 1 for frame in frames):
            raise RuntimeError(
                "Portable detection did not produce valid schema-version-1 JSON frames."
            )
        gui = subprocess.run(
            [str(moved / "HearSafe.exe"), "--smoke-test"],
            cwd=temp,
            env=environment,
            check=False,
            capture_output=True,
            timeout=60,
        )
        if gui.returncode:
            raise RuntimeError(f"Portable desktop startup failed: {gui.stderr!r}")
        benchmark = json.loads(run("benchmark", "--iterations", "30", "--paced").stdout)
        result = {
            "relocated_path_contains_spaces": True,
            "default_model_loaded": True,
            "cli_help": True,
            "devices_command": True,
            "desktop_startup": True,
            "detection_frames": len(frames),
            "device_count": len(listed_devices),
            "yamnet_benchmark": benchmark,
            "note": "Runs on the build computer. A separate machine without Python remains a manual check.",
        }
        if research_model:
            research = run(
                "detect", str(audio_file), "--model", str(research_model.resolve()), "--json"
            )
            research_frames = [
                json.loads(line) for line in research.stdout.splitlines() if line.strip()
            ]
            if not research_frames or any(
                frame.get("schema_version") != 1 for frame in research_frames
            ):
                raise RuntimeError("Portable research model did not return valid JSON frames.")
            result["research_model_loaded_without_pytorch"] = True
            result["research_model_id"] = research_frames[0]["model_id"]
            manifest = json.loads((research_model / "manifest.json").read_text(encoding="utf-8"))
            result["research_model_is_interim"] = bool(
                manifest.get("provenance", {}).get("interim_smoke_package", False)
            )
            result["research_benchmark"] = json.loads(
                run(
                    "benchmark",
                    "--model",
                    str(research_model.resolve()),
                    "--iterations",
                    "30",
                    "--paced",
                ).stdout
            )
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "yamnet")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "HearSafe-Windows-x64.zip")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip relocated CLI smoke checks (development only)",
    )
    parser.add_argument(
        "--research-model",
        type=Path,
        help="Smoke-test a local ONNX package without putting its weights in the ZIP",
    )
    arguments = parser.parse_args(argv)
    if platform.system() != "Windows" or struct.calcsize("P") != 8:
        parser.error("Build this Windows x64 artifact on Windows using a 64-bit interpreter.")
    if sys.version_info[:2] != (3, 12):
        parser.error("The portable release requires the project's Python 3.12 environment.")
    try:
        importlib.metadata.version("pyinstaller")
        model_dir = arguments.model_dir.resolve()
        validate_model(model_dir)
        work = ROOT / "build" / "portable"
        dist = ROOT / "dist"
        environment = os.environ.copy()
        environment["HEARSAFE_BUILD_ROOT"] = str(ROOT)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--clean",
                "--workpath",
                str(work),
                "--distpath",
                str(dist),
                str(ROOT / "packaging" / "hearsafe.spec"),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        bundle = dist / "HearSafe"
        bundled_model = bundle / "models" / "yamnet"
        shutil.copytree(model_dir, bundled_model, dirs_exist_ok=True)
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
        shutil.copy2(ROOT / "packaging" / "PORTABLE_README.txt", bundle / "START_HERE.txt")
        shutil.copy2(
            ROOT / "packaging" / "THIRD_PARTY_NOTICES.md", bundle / "THIRD_PARTY_NOTICES.md"
        )
        for filename in ("testing.md", "integration.md", "roadmap.md"):
            target = bundle / "docs" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "docs" / filename, target)
        copy_public_reports(bundle)
        copy_dependency_notices(bundle / "licenses")
        shutil.copytree(ROOT / "packaging" / "licenses", bundle / "licenses", dirs_exist_ok=True)
        # LiteRT's current wheel declares Apache-2.0 but omits a license file.
        # Preserve its source copyright notice and the same upstream license
        # text carried by the official YAMNet package.
        litert_licenses = bundle / "licenses" / "ai-edge-litert"
        litert_licenses.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "packaging" / "LITERT_NOTICE.txt", litert_licenses / "NOTICE.txt")
        for license_file in sorted(model_dir.glob("LICENSE*")):
            if license_file.is_file():
                shutil.copy2(license_file, litert_licenses / "LICENSE.txt")
                break
        if not arguments.skip_smoke:
            result = smoke_portable(bundle, arguments.research_model)
            (bundle / "portable-smoke.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        deterministic_zip(bundle, arguments.output.resolve())
        digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
        arguments.output.with_suffix(".zip.sha256").write_text(
            f"{digest}  {arguments.output.name}\n", encoding="ascii"
        )
        print(f"Portable ZIP: {arguments.output.resolve()}")
        print(f"SHA256: {digest}")
        return 0
    except (
        ValueError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
