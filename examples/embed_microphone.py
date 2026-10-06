"""Small caller-owned input worker; a GUI should run this loop off its UI thread."""

import argparse
import json
import sys

import sounddevice as sd

from hearsafe import Detector, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    detector = Detector(load_model(args.model))
    info = sd.query_devices(args.device, "input")
    rate = int(info["default_samplerate"])
    # Blocking capture belongs in an input/inference worker. The production
    # adapter uses a bounded queue to keep its capture callback lightweight.
    with sd.InputStream(
        device=args.device, samplerate=rate, channels=1, dtype="float32"
    ) as microphone:
        try:
            while True:
                audio, overflowed = microphone.read(max(1, rate // 10))
                if overflowed:
                    print("Microphone gap: resetting detector", file=sys.stderr)
                    detector.reset()
                for frame in detector.process(audio, rate):
                    print(json.dumps(frame.to_dict()), flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
