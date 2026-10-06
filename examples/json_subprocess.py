"""Consume versioned CLI output without importing the inference engine."""

import argparse
import json
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", help="Path to hearsafe-cli.exe")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--file", default=None, help="Analyse a WAV instead of listening")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    command = [args.executable, "detect", args.file] if args.file else [args.executable, "listen"]
    command.append("--json")
    if args.device is not None and not args.file:
        command += ["--device", str(args.device)]
    if args.model:
        command += ["--model", args.model]
    # Stderr stays separate from the JSON stream and remains visible to the host.
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8"
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            frame = json.loads(line)
            if frame.get("schema_version") != 1:
                raise RuntimeError("Unsupported HearSafe output schema")
            # Replace this with the host program's display or event handler.
            print(json.dumps(frame))
        code = process.wait()
        if code:
            raise RuntimeError(f"HearSafe exited with status {code}")
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
