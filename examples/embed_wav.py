"""Feed a WAV file to the shared detector without a desktop window."""

import argparse
import json

import soundfile as sf

from hearsafe import Detector, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    detector = Detector(load_model(args.model))
    with sf.SoundFile(args.audio) as source:
        for audio in source.blocks(blocksize=4096, dtype="float32", always_2d=True):
            for frame in detector.process(audio, source.samplerate):
                print(json.dumps(frame.to_dict()))
    for frame in detector.flush():
        print(json.dumps(frame.to_dict()))


if __name__ == "__main__":
    main()
