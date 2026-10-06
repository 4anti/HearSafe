"""Local desktop testing console for the same detector used by the API and CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import numpy as np
import psutil

from hearsafe import __version__
from hearsafe.audio import iter_wav_chunks
from hearsafe.detector import Detector
from hearsafe.microphone import MicrophoneSession, devices
from hearsafe.models import load_model
from hearsafe.types import AlertRule


class _DesktopModel:
    """Serialize model calls when a cancelled file test is winding down."""

    def __init__(self, model: Any) -> None:
        self.manifest = model.manifest
        self._model = model
        self._lock = threading.Lock()

    def predict(self, window: Any) -> Any:
        with self._lock:
            return self._model.predict(window)


class HearSafeApp:
    """Tkinter owns every widget; workers only submit messages to the UI queue."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HearSafe · Sound testing console")
        self.root.geometry("1120x800")
        self.root.minsize(900, 700)
        self.root.configure(background="#eef2f7")
        self._messages: queue.Queue[tuple[str, int, Any]] = queue.Queue(maxsize=256)
        self._generation = 0
        self._model: Any = None
        self._detector: Any = None
        self._session: MicrophoneSession | None = None
        self._busy = False
        self._capture_opening = False
        self._closing = False
        self._file_cancel = threading.Event()
        self._research_path: str | None = None
        self._loaded_selection: tuple[str, str | None] | None = None
        self._device_map: dict[str, int] = {}
        self._rules: dict[str, AlertRule] = {}
        self._labels: dict[str, str] = {}
        self._records: deque[dict[str, Any]] = deque(maxlen=4000)
        self._discarded_records = 0
        self._process = psutil.Process()
        self._analysis_sessions: list[dict[str, Any]] = []
        self._active_metrics: dict[str, Any] | None = None
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._last_memory_sample = 0.0
        self._trials: list[dict[str, Any]] = []
        self._source = ""
        self._pending_start: str | None = None
        self._last_frames: list[Any] = []
        self._status = tk.StringVar(value="Select your microphone, then press Start.")
        self._model_choice = tk.StringVar(value="Sound explorer (YAMNet)")
        self._model_info = tk.StringVar(
            value="Model loads when you press Load model, Start, or Test WAV."
        )
        self._microphone = tk.StringVar()
        self._search = tk.StringVar()
        self._alerts_enabled = tk.BooleanVar(value=False)
        self._threshold = tk.DoubleVar(value=0.5)
        self._transient = tk.BooleanVar(value=False)
        self._expected = tk.StringVar()
        self._outcome = tk.StringVar(value="Hit")
        self._distance = tk.StringVar(value="0.5 m")
        self._notes = tk.StringVar()
        self._alert_text = tk.StringVar(value="Alerts are disabled · Watchlist is optional")
        self._timing = tk.StringVar(value="Waiting for audio")
        self._build()
        self._search.trace_add("write", lambda *args: self._populate_categories())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)
        self._refresh_devices()

    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#eef2f7", foreground="#203047", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#ffffff")
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22, "bold"),
            background="#182840",
            foreground="#ffffff",
        )
        style.configure("Subtitle.TLabel", background="#182840", foreground="#d4e0ef")
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#afbdd1")])
        style.configure("Treeview", rowheight=29, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background="#eef2f7", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(20, 10), font=("Segoe UI", 10))

        header = tk.Frame(self.root, background="#182840", padx=24, pady=18)
        header.pack(fill="x")
        ttk.Label(header, text="HearSafe", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Explore everyday sounds · Test your microphone · Runs on your computer",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        content = ttk.Frame(self.root, padding=18)
        content.pack(fill="both", expand=True)
        modelbar = ttk.Frame(content)
        modelbar.pack(fill="x", pady=(0, 12))
        ttk.Label(modelbar, text="Model", font=("Segoe UI", 10, "bold")).pack(
            side="left", padx=(0, 9)
        )
        self._model_combo = ttk.Combobox(
            modelbar,
            textvariable=self._model_choice,
            width=30,
            state="readonly",
            values=("Sound explorer (YAMNet)", "HearSafe research model"),
        )
        self._model_combo.pack(side="left")
        self._model_combo.bind("<<ComboboxSelected>>", self._switch_model)
        self._load_button = ttk.Button(modelbar, text="Load model", command=self._load_selected)
        self._load_button.pack(side="left", padx=8)
        ttk.Button(modelbar, text="Choose research folder…", command=self._choose_research).pack(
            side="left"
        )
        ttk.Label(content, textvariable=self._model_info, wraplength=1050).pack(
            anchor="w", pady=(0, 12)
        )

        source = ttk.Frame(content, style="Card.TFrame", padding=14)
        source.pack(fill="x", pady=(0, 12))
        row = ttk.Frame(source, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Microphone", style="Card.TLabel").pack(side="left", padx=(0, 10))
        self._mic_combo = ttk.Combobox(
            row, textvariable=self._microphone, state="readonly", width=48
        )
        self._mic_combo.pack(side="left", fill="x", expand=True)
        self._refresh_button = ttk.Button(row, text="Refresh", command=self._refresh_devices)
        self._refresh_button.pack(side="left", padx=8)
        self._start_button = ttk.Button(
            row, text="▶ Start", style="Accent.TButton", command=self._start
        )
        self._start_button.pack(side="left", padx=(0, 8))
        self._stop_button = ttk.Button(row, text="■ Stop", command=self._stop, state="disabled")
        self._stop_button.pack(side="left")
        meterrow = ttk.Frame(source, style="Card.TFrame")
        meterrow.pack(fill="x", pady=(10, 0))
        ttk.Label(meterrow, text="Input level", style="Card.TLabel", width=12).pack(side="left")
        self._meter = ttk.Progressbar(meterrow, maximum=100, mode="determinate")
        self._meter.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self._level_text = ttk.Label(meterrow, text="— dBFS", style="Card.TLabel", width=12)
        self._level_text.pack(side="left")
        self._wav_button = ttk.Button(meterrow, text="Test WAV file…", command=self._choose_wav)
        self._wav_button.pack(side="left")

        notebook = ttk.Notebook(content)
        notebook.pack(fill="both", expand=True)
        predictions = ttk.Frame(notebook, padding=15)
        watchlist = ttk.Frame(notebook, padding=15)
        testlog = ttk.Frame(notebook, padding=15)
        notebook.add(predictions, text="Live candidates")
        notebook.add(watchlist, text="Categories & watchlist")
        notebook.add(testlog, text="Test log")

        ttk.Label(predictions, text="Top five sounds", font=("Segoe UI", 14, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            predictions, text="Scores describe the model output. They are not accuracy percentages."
        ).pack(anchor="w", pady=(3, 10))
        self._candidate_tree = ttk.Treeview(
            predictions, columns=("sound", "score"), show="headings", height=5
        )
        self._candidate_tree.heading("sound", text="Sound candidate")
        self._candidate_tree.heading("score", text="Model score")
        self._candidate_tree.column("sound", width=700)
        self._candidate_tree.column("score", width=140, anchor="e")
        self._candidate_tree.pack(fill="x")
        ttk.Label(predictions, textvariable=self._timing).pack(anchor="w", pady=(8, 15))
        self._alert_banner = tk.Label(
            predictions,
            textvariable=self._alert_text,
            anchor="w",
            background="#dbeafe",
            foreground="#1e3a8a",
            padx=14,
            pady=12,
            font=("Segoe UI", 11, "bold"),
        )
        self._alert_banner.pack(fill="x")
        ttk.Label(
            predictions,
            text="Try speech, clapping, whistling or knocks. Play recordings through a phone or speaker so the microphone can hear them.",
            wraplength=1000,
        ).pack(anchor="w", pady=(12, 0))

        controls = ttk.Frame(watchlist)
        controls.pack(fill="x")
        ttk.Checkbutton(
            controls,
            text="Enable visual alerts",
            variable=self._alerts_enabled,
            command=self._apply_alerts,
        ).pack(side="left")
        ttk.Label(controls, text="Threshold").pack(side="left", padx=(24, 6))
        ttk.Spinbox(
            controls, textvariable=self._threshold, from_=0.01, to=1.0, increment=0.05, width=5
        ).pack(side="left")
        ttk.Checkbutton(
            controls, text="Brief sound (one strong result)", variable=self._transient
        ).pack(side="left", padx=15)
        ttk.Button(controls, text="Apply to selected", command=self._update_rule).pack(side="right")
        ttk.Label(
            watchlist,
            text="Sustained sounds require two results. Alerts rearm after the sound fades and use a five-second cooldown.",
        ).pack(anchor="w", pady=(7, 12))
        sides = ttk.Frame(watchlist)
        sides.pack(fill="both", expand=True)
        sides.columnconfigure(0, weight=1)
        sides.columnconfigure(2, weight=1)
        sides.rowconfigure(1, weight=1)
        ttk.Entry(sides, textvariable=self._search).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(sides, text="Your watchlist", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=2, sticky="w"
        )
        self._category_tree = ttk.Treeview(
            sides, columns=("sound",), show="headings", selectmode="extended"
        )
        self._category_tree.heading("sound", text="Search model categories above")
        self._category_tree.column("sound", width=380)
        self._category_tree.grid(row=1, column=0, sticky="nsew")
        buttons = ttk.Frame(sides, padding=10)
        buttons.grid(row=1, column=1)
        ttk.Button(buttons, text="Add →", command=self._add_rules).pack(pady=5)
        ttk.Button(buttons, text="← Remove", command=self._remove_rules).pack(pady=5)
        self._watch_tree = ttk.Treeview(
            sides, columns=("sound", "threshold", "mode"), show="headings", selectmode="extended"
        )
        for name, title, width in (
            ("sound", "Sound", 260),
            ("threshold", "Score", 70),
            ("mode", "Evidence", 85),
        ):
            self._watch_tree.heading(name, text=title)
            self._watch_tree.column(name, width=width)
        self._watch_tree.grid(row=1, column=2, sticky="nsew")
        self._watch_tree.bind("<<TreeviewSelect>>", self._rule_selected)

        trialbar = ttk.Frame(testlog)
        trialbar.pack(fill="x")
        ttk.Label(trialbar, text="Expected sound").pack(side="left")
        ttk.Entry(trialbar, textvariable=self._expected, width=22).pack(side="left", padx=7)
        ttk.Combobox(
            trialbar,
            textvariable=self._outcome,
            state="readonly",
            width=10,
            values=("Hit", "Miss", "Wrong", "Duplicate"),
        ).pack(side="left")
        ttk.Combobox(
            trialbar,
            textvariable=self._distance,
            width=9,
            values=("0.5 m", "2 m", "WAV", "Background"),
        ).pack(side="left", padx=7)
        ttk.Button(trialbar, text="Log trial", command=self._log_trial).pack(side="left")
        ttk.Button(trialbar, text="Export JSON…", command=self._export).pack(side="right")
        notesbar = ttk.Frame(testlog)
        notesbar.pack(fill="x", pady=(8, 10))
        ttk.Label(notesbar, text="Trial notes").pack(side="left", padx=(0, 9))
        ttk.Entry(notesbar, textvariable=self._notes).pack(side="left", fill="x", expand=True)
        self._history_tree = ttk.Treeview(
            testlog, columns=("kind", "time", "sound", "result"), show="headings", height=8
        )
        for name, title, width in (
            ("kind", "Type", 90),
            ("time", "Audio time / logged time", 190),
            ("sound", "Sound", 370),
            ("result", "Result / score", 140),
        ):
            self._history_tree.heading(name, text=title)
            self._history_tree.column(name, width=width)
        self._history_tree.pack(fill="both", expand=True)
        ttk.Label(
            testlog,
            text="Log ten trials per sound and distance. Export includes model results, visual alerts and your trial notes.",
            wraplength=1000,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(content, textvariable=self._status, wraplength=1060).pack(
            anchor="w", pady=(12, 0)
        )

    def _post(self, kind: str, payload: Any, generation: int | None = None) -> None:
        self._messages.put((kind, self._generation if generation is None else generation, payload))

    def _refresh_devices(self) -> None:
        def work() -> None:
            try:
                self._post("devices", devices(), -1)
            except (OSError, RuntimeError, ValueError) as exc:
                self._post("status", str(exc), -1)

        threading.Thread(target=work, name="HearSafe device inventory", daemon=True).start()

    def _switch_model(self, event: Any = None) -> None:
        self._stop(invalidate=True)
        self._model = self._detector = None
        self._loaded_selection = None
        self._rules.clear()
        self._labels.clear()
        self._last_frames = []
        self._timing.set("Waiting for audio")
        self._alerts_enabled.set(False)
        self._populate_categories()
        self._populate_watchlist()
        self._candidate_tree.delete(*self._candidate_tree.get_children())
        self._model_info.set(
            "Choose a model and press Load model or Start. Audio and watchlist state were cleared."
        )
        self._alert_text.set("Alerts are disabled · Watchlist is optional")

    def _choose_research(self) -> None:
        path = filedialog.askdirectory(
            title="Select the trained model folder containing manifest.json"
        )
        if path:
            self._research_path = path
            self._model_choice.set("HearSafe research model")
            self._switch_model()
            self._load_selected()

    def _load_selected(self, after: str | None = None) -> None:
        if self._busy:
            return
        selection = self._model_choice.get()
        if selection == "HearSafe research model" and not self._research_path:
            self._choose_research()
            if after and self._busy:
                self._pending_start = after
            return
        path = self._research_path if selection == "HearSafe research model" else None
        if self._model is not None and self._loaded_selection == (selection, path):
            if after == "microphone":
                self._start_capture()
            elif after:
                self._run_wav(after)
            return
        self._stop(invalidate=True)
        self._busy = True
        self._pending_start = after
        generation = self._generation
        self._status.set("Loading the local model…")
        self._set_active(False)
        self._load_button.configure(state="disabled")

        def work() -> None:
            try:
                self._post("model", (_DesktopModel(load_model(path)), selection, path), generation)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                self._post("error", f"Cannot load model: {exc}", generation)

        threading.Thread(target=work, name="HearSafe model loader", daemon=True).start()

    def _start(self) -> None:
        if self._microphone.get() not in self._device_map:
            self._status.set(
                "Choose an available input microphone. Refresh if you just connected it."
            )
            return
        self._load_selected("microphone")

    def _start_capture(self) -> None:
        if self._model is None or self._session and self._session.running:
            return
        self._generation += 1
        generation = self._generation
        self._source = self._microphone.get()
        self._begin_metrics("microphone")
        self._detector = Detector(self._model)
        self._detector.set_alerts(self._rules if self._alerts_enabled.get() else {})
        self._session = MicrophoneSession(
            self._detector,
            device=self._device_map[self._microphone.get()],
            on_frame=lambda frame: self._post("frame", frame, generation),
            on_status=lambda message: self._post("status", message, generation),
            on_level=lambda level: self._post("level", level, generation),
        )
        session = self._session
        self._capture_opening = True
        self._set_active(True)
        self._status.set("Opening selected microphone…")

        def work() -> None:
            try:
                if self._session is not session or self._generation != generation:
                    return
                session.start()
                self._post("started", session, generation)
            except (OSError, RuntimeError, ValueError) as exc:
                self._post("error", str(exc), generation)

        threading.Thread(target=work, name="HearSafe microphone opener", daemon=True).start()

    def _set_active(self, active: bool) -> None:
        self._start_button.configure(state="disabled" if active or self._busy else "normal")
        self._stop_button.configure(state="normal" if active or self._busy else "disabled")
        self._mic_combo.configure(state="disabled" if active else "readonly")
        self._refresh_button.configure(state="disabled" if active else "normal")
        self._wav_button.configure(state="disabled" if self._busy else "normal")

    def _stop(self, invalidate: bool = False) -> None:
        self._file_cancel.set()
        session, self._session = self._session, None
        self._finish_metrics(session)
        if session:
            threading.Thread(
                target=session.stop, name="HearSafe microphone closer", daemon=True
            ).start()
        if invalidate:
            self._generation += 1
        self._busy = False
        self._capture_opening = False
        self._pending_start = None
        self._set_active(False)
        self._load_button.configure(state="normal")
        self._meter["value"] = 0
        self._level_text.configure(text="— dBFS")
        self._status.set("Stopped. Raw microphone audio has not been saved.")

    def _choose_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Test a WAV recording", filetypes=(("WAV audio", "*.wav"), ("All files", "*.*"))
        )
        if path:
            self._load_selected(path)

    def _run_wav(self, path: str) -> None:
        self._stop(invalidate=True)
        generation = self._generation
        self._source = str(Path(path).resolve())
        self._begin_metrics("wav")
        self._file_cancel = threading.Event()
        cancel = self._file_cancel
        model = self._model
        rules = dict(self._rules) if self._alerts_enabled.get() else {}
        self._busy = True
        self._set_active(True)
        self._status.set(f"Testing {Path(path).name}…")

        def work() -> None:
            try:
                detector = Detector(model)
                detector.set_alerts(rules)
                count = 0
                for audio, rate in iter_wav_chunks(path):
                    if cancel.is_set():
                        return
                    for frame in detector.process(audio, rate):
                        self._post("frame", frame, generation)
                        count += 1
                if not cancel.is_set():
                    for frame in detector.flush():
                        self._post("frame", frame, generation)
                        count += 1
                    self._post(
                        "complete",
                        f"WAV test complete: {count} analysis windows. Export or log a trial in Test log.",
                        generation,
                    )
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                self._post("error", f"WAV test failed: {exc}", generation)

        threading.Thread(target=work, name="HearSafe WAV analysis", daemon=True).start()

    def _populate_categories(self) -> None:
        self._category_tree.delete(*self._category_tree.get_children())
        needle = self._search.get().casefold()
        for label_id, name in self._labels.items():
            if needle in name.casefold() or needle in str(label_id).casefold():
                self._category_tree.insert("", "end", iid=label_id, values=(name,))

    def _populate_watchlist(self) -> None:
        self._watch_tree.delete(*self._watch_tree.get_children())
        for label_id, rule in self._rules.items():
            self._watch_tree.insert(
                "",
                "end",
                iid=label_id,
                values=(
                    self._labels.get(label_id, label_id),
                    f"{rule.threshold:.2f}",
                    "Brief" if rule.transient else "Sustained",
                ),
            )

    def _selected_rule(self) -> AlertRule | None:
        try:
            threshold = self._threshold.get()
            if not 0 < threshold <= 1:
                raise ValueError("Use a threshold greater than zero and at most one.")
            return AlertRule(
                threshold=threshold,
                transient=self._transient.get(),
                cooldown_seconds=5,
                consecutive=2,
            )
        except (ValueError, tk.TclError) as exc:
            self._status.set(f"Invalid threshold: {exc}")
            return None

    def _add_rules(self) -> None:
        rule = self._selected_rule()
        if rule is not None:
            for label_id in self._category_tree.selection():
                self._rules[label_id] = rule
            self._populate_watchlist()
            self._apply_alerts()

    def _remove_rules(self) -> None:
        for label_id in self._watch_tree.selection():
            self._rules.pop(label_id, None)
        self._populate_watchlist()
        self._apply_alerts()

    def _update_rule(self) -> None:
        rule = self._selected_rule()
        if rule is not None:
            for label_id in self._watch_tree.selection():
                self._rules[label_id] = rule
            self._populate_watchlist()
            self._apply_alerts()

    def _rule_selected(self, event: Any) -> None:
        selected = self._watch_tree.selection()
        if selected:
            rule = self._rules[selected[0]]
            self._threshold.set(rule.threshold)
            self._transient.set(rule.transient)

    def _apply_alerts(self) -> None:
        rules = self._rules if self._alerts_enabled.get() else {}
        if self._session:
            self._session.set_alerts(rules)
        self._alert_text.set(
            f"Watching {len(rules)} categories · Awaiting matching audio"
            if rules
            else "Alerts are disabled · Watchlist is optional"
        )
        self._alert_banner.configure(background="#dbeafe", foreground="#1e3a8a")
        if self._busy:
            self._status.set(
                "Watchlist edits apply to the next WAV test. Live microphone edits apply between audio chunks."
            )

    def _show_frame(self, frame: Any) -> None:
        self._last_frames = frame.top(5)
        self._candidate_tree.delete(*self._candidate_tree.get_children())
        for prediction in self._last_frames:
            self._candidate_tree.insert(
                "", "end", values=(prediction.label, f"{prediction.score:.4f}")
            )
        self._timing.set(
            f"Audio {frame.start_ms / 1000:.2f}–{frame.end_ms / 1000:.2f} s · Inference {frame.inference_ms:.1f} ms"
        )
        # Keep long tests inexpensive; full predictions remain in the API/CLI.
        payload = {
            "schema_version": frame.schema_version,
            "model_id": frame.model_id,
            "start_ms": frame.start_ms,
            "end_ms": frame.end_ms,
            "predictions": [asdict(item) for item in self._last_frames],
            "rms": frame.rms,
            "inference_ms": frame.inference_ms,
            "events": [asdict(event) for event in frame.events],
        }
        if self._active_metrics is not None:
            self._active_metrics["_inference_times"].append(frame.inference_ms)
            self._active_metrics["windows"] += 1
        if len(self._records) == self._records.maxlen:
            self._discarded_records += 1
        self._records.append(
            {
                "source": self._source,
                "session_id": self._active_metrics["session_id"] if self._active_metrics else None,
                "recorded_at": datetime.now(UTC).isoformat(),
                "analysis": payload,
            }
        )
        if frame.events:
            names = ", ".join(event.label for event in frame.events)
            self._alert_text.set(f"Detected: {names}")
            self._alert_banner.configure(background="#fef3c7", foreground="#78350f")
            for event in frame.events:
                self._history_tree.insert(
                    "",
                    0,
                    values=(
                        "Alert",
                        f"{frame.end_ms / 1000:.2f} s",
                        event.label,
                        f"{event.score:.4f}",
                    ),
                )
                self._trim_history()

    def _trim_history(self) -> None:
        children = self._history_tree.get_children()
        if len(children) > 1000:
            self._history_tree.delete(*children[1000:])

    def _begin_metrics(self, source_kind: str) -> None:
        cpu = self._process.cpu_times()
        manifest = self._model.manifest
        metrics = {
            "session_id": len(self._analysis_sessions) + 1,
            "source": self._source,
            "source_kind": source_kind,
            "model_id": manifest.model_id,
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "windows": 0,
            "dropped_chunks": 0,
            "last_error": None,
            "sampled_peak_rss_bytes": self._process.memory_info().rss,
            "analysis_interval_ms": manifest.hop_samples / manifest.sample_rate * 1000,
            "_started_monotonic": time.perf_counter(),
            "_cpu_at_start": cpu.user + cpu.system,
            "_inference_times": [],
        }
        self._analysis_sessions.append(metrics)
        self._active_metrics = metrics

    def _finish_metrics(self, session: MicrophoneSession | None = None) -> None:
        metrics = self._active_metrics
        if metrics is None:
            return
        metrics["ended_at"] = datetime.now(UTC).isoformat()
        cpu = self._process.cpu_times()
        metrics["_ended_monotonic"] = time.perf_counter()
        metrics["_cpu_at_end"] = cpu.user + cpu.system
        metrics["sampled_peak_rss_bytes"] = max(
            metrics["sampled_peak_rss_bytes"], self._process.memory_info().rss
        )
        if session is not None:
            metrics["dropped_chunks"] = session.dropped_chunks
            metrics["capture_sample_rate"] = session.sample_rate
            metrics["last_error"] = session.last_error
        self._active_metrics = None

    def _metric_reports(self) -> list[dict[str, Any]]:
        cpu = self._process.cpu_times()
        current_cpu = cpu.user + cpu.system
        current_wall = time.perf_counter()
        reports = []
        for metrics in self._analysis_sessions:
            result = {key: value for key, value in metrics.items() if not key.startswith("_")}
            elapsed = max(
                1e-9, metrics.get("_ended_monotonic", current_wall) - metrics["_started_monotonic"]
            )
            cpu_seconds = max(0, metrics.get("_cpu_at_end", current_cpu) - metrics["_cpu_at_start"])
            timings = metrics["_inference_times"]
            result.update(
                {
                    "elapsed_seconds": elapsed,
                    "process_cpu_seconds": cpu_seconds,
                    "average_process_cpu_percent": 100 * cpu_seconds / elapsed,
                    "inference_p50_ms": float(np.percentile(timings, 50)) if timings else None,
                    "inference_p95_ms": float(np.percentile(timings, 95)) if timings else None,
                    "inference_keeps_up": bool(
                        np.percentile(timings, 95) < metrics["analysis_interval_ms"]
                    )
                    if timings
                    else None,
                }
            )
            if metrics is self._active_metrics and self._session is not None:
                result["dropped_chunks"] = self._session.dropped_chunks
                result["capture_sample_rate"] = self._session.sample_rate
                result["last_error"] = self._session.last_error
            reports.append(result)
        return reports

    def _log_trial(self) -> None:
        expected = self._expected.get().strip()
        if not expected:
            self._status.set("Enter the sound you intended to test, then log the trial.")
            return
        trial = {
            "logged_at": datetime.now(UTC).isoformat(),
            "model_id": self._model.manifest.model_id if self._model else None,
            "source": self._source,
            "expected_label": expected,
            "outcome": self._outcome.get(),
            "distance": self._distance.get(),
            "notes": self._notes.get().strip(),
            "latest_candidates": [
                {
                    "label_id": prediction.label_id,
                    "label": prediction.label,
                    "score": prediction.score,
                }
                for prediction in self._last_frames
            ],
            "alerts_enabled": self._alerts_enabled.get(),
        }
        self._trials.append(trial)
        self._history_tree.insert(
            "",
            0,
            values=(
                "Trial",
                datetime.now(UTC).astimezone().strftime("%H:%M:%S"),
                expected,
                trial["outcome"],
            ),
        )
        self._status.set(f"Logged trial {len(self._trials)}: {expected} / {trial['outcome']}.")
        self._trim_history()

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export test results",
            defaultextension=".json",
            initialfile="hearsafe-test-results.json",
            filetypes=(("JSON report", "*.json"),),
        )
        if not path:
            return
        report = {
            "schema_version": 1,
            "exported_at": datetime.now(UTC).isoformat(),
            "score_note": "Model scores are not calibrated accuracy percentages.",
            "audio_saved": False,
            "app_version": __version__,
            "hardware": {
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
            },
            "models": self._model_metadata,
            "performance_sessions": self._metric_reports(),
            "performance_note": "CPU includes the whole application; 100% is one logical CPU. Peak RSS is sampled, not a lifetime operating-system high-water mark. Inference timing excludes capture, preprocessing and UI.",
            "saved_candidates_per_window": 5,
            "analysis_record_capacity": self._records.maxlen,
            "older_analysis_records_discarded": self._discarded_records,
            "analysis_records": list(self._records),
            "trials": self._trials,
            "watchlist": {
                label_id: {
                    "threshold": rule.threshold,
                    "transient": rule.transient,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "consecutive": rule.consecutive,
                }
                for label_id, rule in self._rules.items()
            },
        }
        try:
            Path(path).write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._status.set(
                f"Exported {len(self._records)} windows and {len(self._trials)} trials to {path}"
            )
        except OSError as exc:
            self._status.set(f"Cannot export test results: {exc}")

    def _poll(self) -> None:
        if self._closing:
            return
        latest_level = None
        for _ in range(120):
            try:
                kind, generation, payload = self._messages.get_nowait()
            except queue.Empty:
                break
            if generation not in (-1, self._generation):
                continue
            if kind == "devices":
                previous = self._microphone.get()
                self._device_map = {f"{item['id']}: {item['name']}": item["id"] for item in payload}
                self._mic_combo["values"] = list(self._device_map)
                if previous not in self._device_map:
                    self._microphone.set(next(iter(self._device_map), ""))
                if not self._device_map:
                    self._status.set(
                        "No microphone found. Connect your headset and press Refresh. WAV testing is available."
                    )
            elif kind == "model":
                self._model, selection, path = payload
                self._loaded_selection = (selection, path)
                manifest = self._model.manifest
                self._model_metadata[manifest.model_id] = asdict(manifest)
                self._labels = {str(label.id): label.name for label in manifest.labels}
                self._populate_categories()
                self._model_info.set(
                    f"{manifest.name} · {len(manifest.labels)} categories · Window / warmup {manifest.window_samples / manifest.sample_rate:.3f} s · Updates every {manifest.hop_samples / manifest.sample_rate:.2f} s"
                )
                self._busy = False
                self._load_button.configure(state="normal")
                self._set_active(False)
                self._status.set("Model ready. Press Start or Test WAV to analyse audio.")
                pending, self._pending_start = self._pending_start, None
                if pending == "microphone":
                    self._start_capture()
                elif pending:
                    self._run_wav(pending)
            elif kind == "frame":
                # Preserve every result for export, even when files run faster
                # than the interface can repaint.
                self._show_frame(payload)
            elif kind == "level":
                latest_level = payload
            elif kind == "status":
                self._status.set(payload)
            elif kind == "started":
                self._capture_opening = False
                if self._session is not payload:
                    threading.Thread(target=payload.stop, daemon=True).start()
            elif kind in ("error", "complete"):
                self._finish_metrics(self._session)
                self._busy = False
                self._capture_opening = False
                self._load_button.configure(state="normal")
                self._set_active(False)
                self._status.set(payload)
                self._pending_start = None
                if kind == "error" and self._session:
                    session, self._session = self._session, None
                    threading.Thread(target=session.stop, daemon=True).start()
        if latest_level is not None:
            db = 20 * math.log10(max(latest_level, 1e-6))
            self._meter["value"] = max(0, min(100, (db + 60) / 60 * 100))
            self._level_text.configure(text=f"{db:.0f} dBFS")
        if self._session and not self._session.running and not self._capture_opening:
            self._finish_metrics(self._session)
            self._set_active(False)
        if self._active_metrics is not None and time.perf_counter() - self._last_memory_sample >= 1:
            self._active_metrics["sampled_peak_rss_bytes"] = max(
                self._active_metrics["sampled_peak_rss_bytes"], self._process.memory_info().rss
            )
            self._last_memory_sample = time.perf_counter()
        self.root.after(80, self._poll)

    def close(self) -> None:
        self._closing = True
        self._file_cancel.set()
        if self._session:
            self._session.stop()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HearSafe desktop sound testing console")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Open and close the interface without loading a model or capturing audio",
    )
    args = parser.parse_args(argv)
    root = tk.Tk()
    app = HearSafeApp(root)
    if args.smoke_test:
        root.after(600, app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
