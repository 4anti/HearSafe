# Contributing to HearSafe

Thanks for helping make environmental sound awareness more useful and accessible.

## Good first contributions

- Review the initial sound classes and alert wording with deaf or hard-of-hearing users.
- Improve accessible controls, headset testing instructions, and hardware integration examples.
- Extend meaningful tests for audio streaming, preprocessing, or evaluation.
- Report false alerts and missed sounds with enough detail to reproduce them, without uploading private recordings.

## Before opening a pull request

1. Open an issue for a substantial change so the scope can be discussed.
2. Keep changes focused and explain how you checked them.
3. Use the dataset's predefined folds for benchmark claims and state which folds were used.
4. Do not commit datasets, private recordings, secrets, or model weights without checking their rights and the project policy.
5. Describe relevant accessibility and privacy effects.

Use Python 3.12 and `uv sync --frozen --extra dev` for runtime development. Run
`uv run --frozen --extra dev pytest` and `uv run --frozen --extra dev ruff check src tests`.
Add the `train` extra for export tests; the `build` extra produces the Windows ZIP.

All contributions to original HearSafe code and documentation are submitted under the repository's MIT license. Third-party datasets and assets retain their own licenses.
