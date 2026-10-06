# HearSafe implementation roadmap

## First working prototype

The source implements the four requested stages:

1. **Shared detector:** verified YAMNet, native microphone rates, continuous
   resampling, overlapping windows, file/stream parity, versioned JSON, and
   gaps/errors handled before further detection.
2. **Local training:** all 50 ESC-50 classes, fixed folds 1–3/4/5, shared mel
   preprocessing, training-only normalization, seeded CPU CNN training, early
   stopping, ONNX export, parity checks, and held-out reports.
3. **Desktop tester:** microphone selector, level meter, Start/Stop, selectable
   models, live top five, searchable categories, optional watchlist, file tests,
   event history, manual trial notes, and JSON export.
4. **Windows packaging:** GUI and CLI executables, YAMNet, runtime dependencies,
   notices, checksums, relocated smoke checks, and a Windows CI build.

## Start testing now

1. Use the bundled Sound explorer and confirm the microphone meter reacts to
   speech. Try claps, whistles, knocks, and sounds from a phone speaker.
2. Test known WAV files before speaker-to-microphone trials. Export results.
3. Load the locally trained research model and compare its 50 categories and
   five-second analysis window with YAMNet.
4. Run ten trials for each target sound at 0.5 m and 2 m. Record device settings,
   wrong labels, misses, duplicates, and delay. Repeat with available noise
   suppression settings. Follow the [testing guide](testing.md).
5. Run the 30-minute background check and try the extracted ZIP on a separate
   Windows machine without Python, internet, or administrator rights.

## Before claiming a tested first release

- Publish the measured held-out accuracy, macro F1, per-class recall, confusion
  matrix, model size, inference time, CPU use, and observed memory.
- Publish microphone trial counts and weak categories under the actual room and
  headset settings. Automated file tests cannot supply those measurements.
- Complete the second-computer portability check. The build-computer smoke
  check is useful but does not establish clean-machine compatibility.
- Keep the release marked as a prototype while these checks are pending.

## Later stages

1. Measure the reusable detector on Linux and Raspberry Pi; choose compatible
   runtimes and build those packages on their own operating systems.
2. Improve weak categories using new, appropriately licensed evaluation data
   and repeatable room/device tests.
3. Investigate system-audio capture and custom user-recorded categories.
4. Add recording only as an explicit opt-in flow with visible recording state.

Runtime use has no cloud charge. Costs are the computer/board, microphone,
speaker used for testing, and electricity. Begin with the existing headset;
consider another microphone only if its speech processing hides target sounds.
