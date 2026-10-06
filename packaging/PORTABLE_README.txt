HearSafe portable prototype - Windows x64

1. Extract the entire ZIP to any ordinary folder.
2. Double-click HearSafe.exe.
3. Refresh microphones, choose your headset microphone, and click Start.
4. Speak or clap and check that the level meter moves. Sound candidates appear
   after the model has collected its first audio window.

Keep HearSafe.exe, hearsafe-cli.exe, _internal, and models together. Moving the
whole folder is supported. Python, an account, internet, and a GPU are not
required to detect sounds. The included model has 521 sound categories.

Try a phone or external speaker to play test sounds into the microphone.
Playing a sound only into headphones may not reach your headset microphone.
Model scores express the model's output, not a measured accuracy percentage.

Alerts begin disabled. Choose a watchlist, adjust thresholds, then enable them.
For controlled headset trials and a 30-minute background test, read
docs/testing.md. Read docs/integration.md to feed audio from another program.

Command-line examples (PowerShell in this folder):
  .\hearsafe-cli.exe devices
  .\hearsafe-cli.exe detect "C:\audio\sound.wav" --json
  .\hearsafe-cli.exe listen --device 0 --json

This prototype does not record microphone audio. Exports contain scores,
events, and trial notes.
This prototype needs testing in your environment before relying on its alerts.
It is an assistive awareness tool, not an emergency alarm.

See LICENSE, THIRD_PARTY_NOTICES.md, licenses/, and models/yamnet/ for notices.
Research weights trained from ESC-50 are loaded separately and are not included
in this portable release.
