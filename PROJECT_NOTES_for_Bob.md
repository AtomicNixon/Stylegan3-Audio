# Stylegan-3 / Culture Shock — Project Handoff Notes

Quick orientation for picking this up in Fable. Directory: `D:\Stylegan-3`

## What this project does

`culture_shock.py` generates a music-reactive video from a StyleGAN3 face model.
It takes an MP3, splits it into 4 instrument stems (drums/bass/other/vocals) via
Demucs, then drives a SLERP-interpolated latent walk through the model with each
stem controlling a different visual property:

| Stem | Drives |
|---|---|
| **drums** | truncation psi (0.5–1.0) + coarse style layers (identity/pose) |
| **bass** | mid style layers (facial structure) |
| **other** | fine style layers (texture/detail) |
| **vocals** | mouth-open vector magnitude (lip sync feel) |

The latent sequence itself is a slow, continuous SLERP walk between random seeds
(not synced to the beat) — the stems only modulate *how strongly* that walk
expresses itself.

## Running it

```
D:\Stylegan-3\venv\Scripts\python.exe culture_shock.py --network ffhq.pkl --audio "data/Culture Shock.mp3"
```

Flags (all optional, sensible defaults):
- `--network` — path to a `.pkl` StyleGAN3 model (prompts if omitted)
- `--audio` — path to source MP3 (prompts if omitted)
- `--fps` — output frame rate, default **60**
- `--size` — output resolution, default **512**
- `--batch` — GPU batch size for synthesis, default **16** (RTX 3090 handles this fine)

First run on a new MP3 triggers Demucs stem separation automatically (~1–2 min);
results are cached in `data/<audio_name>_stems/` so subsequent runs skip straight
to rendering. Output lands at `data/<audio_name>.mp4`.

## Environment

Key facts about the env:
- Python venv at `D:\Stylegan-3\venv\` (NOT conda — conda had persistent
  `pkg_resources` issues from `setuptools>=72`; venv sidesteps it)
- `torch 2.7.1+cu118` / `torchaudio 2.7.1+cu118` / `torchvision 0.22.1+cu118` —
  **versions must match exactly** or you'll get DLL load errors on Windows
- `demucs 4.0.1` handles stem separation
- `spleeter`/`tensorflow` are still installed in the venv from an earlier
  approach we abandoned in favor of Demucs — harmless dead weight, safe to
  `pip uninstall` if you want a leaner env, not required for anything
- Always invoke as `venv\Scripts\python.exe`, never bare `python`

## Files

- `culture_shock.py` — main render script (current, working)
- `vector_explorer.py` — standalone Tkinter GUI: sliders for each vector in
  `/vectors`, PSI slider, network/seed picker, live preview. Independent tool,
  not required by `culture_shock.py`.
- `legacy.py` — converts old TF-era StyleGAN checkpoints to the current pickle
  format. Only needed if importing a new/foreign `.pkl`.
- `pretrained_networks.py`, `dnnlib/`, `torch_utils/`, `training/` — core
  StyleGAN3 library, required by everything else (do not remove)
- `vectors/*.npy` — ~45 pre-extracted latent direction vectors (age, gender,
  smile, hair color, eye/mouth attributes, pose, etc.) usable with either
  `culture_shock.py` or `vector_explorer.py`
- `data/` — working directory for stems, cached demucs output, and rendered
  MP4s. Gets large fast; safe to archive/delete `*_stems/` and `demucs_out/`
  subfolders between sessions since they regenerate automatically.

## Known quirks

- **Frame count sanity check**: frame count is `ceil(audio_duration_seconds * fps)`.
  If you ever see a wildly wrong frame count, check that only ONE audio file's
  stems are being loaded — an earlier bug pulled in *every* mp3 in `data/`.
  This is now fixed but worth remembering if it recurs.
- **fp16/AMP does not work** with StyleGAN3's custom CUDA ops (`upfirdn2d`,
  `bias_act`) — they hard-assert on float32 filter kernels. Don't waste time
  re-attempting this without patching the ops themselves.
- Occasional `mp3float ... overread` warnings from ffmpeg during encode are
  harmless (malformed trailing MP3 frames); suppressed via `-loglevel warning`.

## Models available

`ffhq.pkl` (base), `myface.pkl`, `myfacenew.pkl`, `newmyface.pkl`,
`myalienface-140new.pkl`, `newguitar.pkl`, `stylegan3-r-ffhq-1024x1024.pkl`

## Audio tracks with cached stems ready to go

Culture Shock, elasticadadada, elasticalineup, elasticavaseline, hereandnow,
Hiccups, Ja Ja Ja, negquiet, ratchicken3, YBR
