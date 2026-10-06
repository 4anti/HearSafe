"""CLI for local audio exploration, integration, evaluation and training."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from .audio import iter_wav_chunks
from .detector import Detector
from .models import download_yamnet, load_model


def analyse_file(detector: Detector, path: Path | str):
    for chunk, rate in iter_wav_chunks(path):
        yield from detector.process(chunk, rate)
    yield from detector.flush()


def _show_frame(frame, as_json: bool, top: int):
    if as_json:
        result = frame.to_dict()
        result["predictions"] = [
            {"label_id": item.label_id, "label": item.label, "score": item.score}
            for item in frame.top(top)
        ]
        print(json.dumps(result, allow_nan=False), flush=True)
    else:
        text = ", ".join(f"{item.label}: {item.score:.3f}" for item in frame.top(top))
        print(f"{frame.start_ms / 1000:.2f}-{frame.end_ms / 1000:.2f}s  {text}", flush=True)
        for event in frame.events:
            print(f"ALERT {event.label} ({event.score:.3f})", flush=True)


def benchmark(model, iterations=30, paced=False) -> dict:
    import platform

    import psutil

    if iterations < 1:
        raise ValueError("Benchmark iterations must be positive")
    process = psutil.Process()
    audio = (
        np.random.default_rng(42).normal(0, 0.01, model.manifest.window_samples).astype(np.float32)
    )
    model.predict(audio)
    cpu_start = sum(process.cpu_times()[:2])
    started = perf_counter()
    timings = []
    peak = process.memory_info().rss
    interval = model.manifest.hop_samples / model.manifest.sample_rate
    for _ in range(iterations):
        before = perf_counter()
        model.predict(audio)
        timings.append((perf_counter() - before) * 1000)
        peak = max(peak, process.memory_info().rss)
        if paced:
            sleep(max(0, interval - (perf_counter() - before)))
    elapsed = perf_counter() - started
    cpu_seconds = sum(process.cpu_times()[:2]) - cpu_start
    return {
        "model_id": model.manifest.model_id,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "iterations": iterations,
        "method": "one window per model hop" if paced else "consecutive inference burst",
        "includes": "model frontend and inference; excludes capture, resampling, and UI",
        "inference_mean_ms": float(np.mean(timings)),
        "inference_max_ms": float(max(timings)),
        "inference_p50_ms": float(np.percentile(timings, 50)),
        "inference_p95_ms": float(np.percentile(timings, 95)),
        "hop_ms": model.manifest.hop_samples / model.manifest.sample_rate * 1000,
        "keeps_up_with_hop": bool(
            np.percentile(timings, 95)
            < model.manifest.hop_samples / model.manifest.sample_rate * 1000
        ),
        "observed_peak_rss_mb": peak / (1024 * 1024),
        "process_peak_working_set_mb": (
            getattr(process.memory_info(), "peak_wset", peak) / (1024 * 1024)
        ),
        "cpu_seconds": cpu_seconds,
        "elapsed_seconds": elapsed,
        "cpu_percent_one_core_basis": cpu_seconds / elapsed * 100 if elapsed else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HearSafe: offline microphone and WAV sound explorer"
    )
    parser.add_argument("--version", action="version", version="HearSafe 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    devices_parser = commands.add_parser("devices", help="List input microphones without recording")
    devices_parser.add_argument("--json", action="store_true")
    download = commands.add_parser(
        "download-model", help="Explicitly download the pinned YAMNet model"
    )
    download.add_argument("--output", type=Path, default=Path("models/yamnet"))
    detect = commands.add_parser("detect", help="Analyse an audio file")
    detect.add_argument("file", type=Path)
    listen = commands.add_parser("listen", help="Analyse a selected microphone live")
    listen.add_argument("--device", type=int)
    listen.add_argument("--duration", type=float, help="Optional duration in seconds")
    for sub in (detect, listen):
        sub.add_argument("--model", type=Path, help="Model package folder (default: YAMNet)")
        sub.add_argument("--json", action="store_true", help="One JSON analysis frame per line")
        sub.add_argument("--top", type=int, default=5)
    bench = commands.add_parser("benchmark", help="Measure CPU inference time and memory")
    bench.add_argument("--model", type=Path)
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument(
        "--paced",
        action="store_true",
        help="Run one window every model hop to measure idle time and CPU use",
    )
    bench.add_argument("--output", type=Path)
    train = commands.add_parser(
        "train", help="Train the local ESC-50 research model (training extra)"
    )
    train.add_argument("--dataset", type=Path, default=Path("data/ESC-50-master"))
    train.add_argument("--output", type=Path, default=Path("data/models/esc50"))
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate an exported model on an untouched ESC-50 fold"
    )
    evaluate.add_argument("--dataset", type=Path, default=Path("data/ESC-50-master"))
    evaluate.add_argument("--model", type=Path, default=Path("data/models/esc50"))
    evaluate.add_argument("--fold", type=int, default=5)
    gui = commands.add_parser("gui", help="Open the desktop testing console")
    gui.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if getattr(sys, "frozen", False) and args.command in {"train", "evaluate"}:
            raise ValueError(
                "Training and evaluation use the source checkout with the train extra. "
                "The portable app can load exported ONNX model folders directly. See docs/testing.md."
            )
        if args.command == "download-model":
            print(f"Model ready: {download_yamnet(args.output)}")
        elif args.command == "devices":
            from .microphone import devices

            result = devices()
            if args.json:
                print(json.dumps(result))
            else:
                for device in result:
                    print(
                        f"{device['id']}: {device['name']} ({device['default_samplerate']:.0f} Hz)"
                    )
                if not result:
                    print("No input microphones found", file=sys.stderr)
        elif args.command == "detect":
            if args.top < 1:
                raise ValueError("Top result count must be positive")
            for frame in analyse_file(Detector(load_model(args.model)), args.file):
                _show_frame(frame, args.json, args.top)
        elif args.command == "listen":
            from .microphone import MicrophoneSession

            if args.top < 1 or (args.duration is not None and args.duration <= 0):
                raise ValueError("Top count and duration must be positive")
            session = MicrophoneSession(
                Detector(load_model(args.model)),
                args.device,
                on_frame=lambda frame: _show_frame(frame, args.json, args.top),
                on_status=lambda message: print(message, file=sys.stderr, flush=True),
            )
            session.start()
            started = perf_counter()
            try:
                while session.running:
                    if args.duration and perf_counter() - started >= args.duration:
                        break
                    sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                session.stop()
            if session.last_error:
                raise RuntimeError(session.last_error)
        elif args.command == "benchmark":
            started = perf_counter()
            model = load_model(args.model)
            startup = (perf_counter() - started) * 1000
            report = benchmark(model, args.iterations, args.paced)
            report["model_load_ms"] = startup
            text = json.dumps(report, indent=2, allow_nan=False)
            print(text)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
        elif args.command == "train":
            from .training import train as train_model

            report = train_model(
                args.dataset,
                args.output,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                threads=args.threads,
                seed=args.seed,
            )
            print(json.dumps(report, indent=2))
        elif args.command == "evaluate":
            from .training import evaluate as evaluate_model

            print(json.dumps(evaluate_model(args.dataset, args.model, fold=args.fold), indent=2))
        elif args.command == "gui":
            from .gui import main as open_gui

            open_gui(["--smoke-test"] if args.smoke_test else [])
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"HearSafe: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
