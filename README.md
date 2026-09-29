# HearSafe

HearSafe is an early-stage, open-source project for local environmental sound awareness. The goal is to turn selected sounds into clear visual alerts for people who benefit from them, including deaf and hard-of-hearing users.

**Status:** dataset exploration. There is no working detector or downloadable model yet. HearSafe is an assistive awareness project, not an emergency or life-safety system.

## First milestone

1. Inspect the [ESC-50](https://github.com/karolpiczak/ESC-50) metadata and the ten starting sound classes listed below.
2. Build a reproducible audio-file classifier and report results using ESC-50's predefined folds.
3. Add microphone input and visual alerts after file inference has been evaluated.

The initial classes are baby crying, footsteps, coughing, door knock, clock alarm, glass breaking, siren, car horn, dog, and crackling fire. These are dataset labels for the first experiment; they are not a promise that the eventual application can reliably detect every real-world instance.

## Dataset setup

Download ESC-50 from its [official repository](https://github.com/karolpiczak/ESC-50) and extract it to `data/ESC-50-master/`. On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -L --fail -o data/esc50-master.zip https://github.com/karoldvl/ESC-50/archive/master.zip
tar.exe -xf data/esc50-master.zip -C data
```

The dataset is kept out of Git. Then run:

```powershell
python scripts/report_esc50.py
```

The command writes [the class-count report](reports/esc50_class_counts.md) from `data/ESC-50-master/meta/esc50.csv`. The script uses only Python's standard library.

**Data rights:** ESC-50 is CC BY-NC; its ESC-10 subset is CC BY. The MIT license in this repository covers HearSafe's own code and documentation, not third-party audio. See the [ESC-50 license and attributions](https://github.com/karolpiczak/ESC-50#license) before using or sharing the data. No dataset audio or trained weights are included in this repository.

## Resources

- [Project plan](HearSafe_PROJECT_PLAN.md)
- [ESC-50 dataset and paper](https://github.com/karolpiczak/ESC-50)
- [PyTorch installation guide](https://docs.pytorch.org/get-started/locally/)
- Later datasets: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), [FSD50K](https://zenodo.org/records/4060432)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues about accessibility, sound priorities, and evaluation in realistic environments are especially useful.

## License

HearSafe's original code and documentation are available under the [MIT License](LICENSE).
