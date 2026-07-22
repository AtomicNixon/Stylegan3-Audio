# Stylegan3-Audio

Music-reactive video generation from StyleGAN3 models. Feed it an MP3, get a
video of a continuous latent-space walk where each instrument stem drives a
different visual property of the output.

Two faces, one core:

- **`culture_shock.py`** — command line
- **`mixer_gui.py`** — the Mixer: a patchbay GUI (row per stem: envelope
  thumbnail, target dropdown, vector picker, strength, invert)
- **`stylegan3_audio.py`** — the shared core library both wrap

## How it works

Audio is split into stems — your own DAW exports (`--stems-dir`), or
[Demucs](https://github.com/facebookresearch/demucs) separation
(drums / bass / other / vocals) as the automatic fallback. Each stem becomes a
per-frame amplitude envelope, routed to a visual target:

| Target | Effect |
|---|---|
| `psi` | truncation psi (psi_base … psi_base+psi_range) — overall intensity |
| `coarse` | style-mix toward a second walk on coarse layers (identity/pose) |
| `mid` | style-mix on mid layers (facial structure) |
| `fine` | style-mix on fine layers (texture/detail) |
| `vector` | add a named latent direction (`vectors/*.npy`), scaled by the stem |

The latent walk itself is a slow SLERP between seeds, deliberately not
beat-synced — the music shapes *intensity*, not *position*.

Default routing (no `--map` flags) reproduces the original Culture Shock look:
drums→psi+coarse, bass→mid, other→fine, vocals→vector(mouth_ratio, 1.5, inverted).

## CLI

```
python culture_shock.py --network ffhq.pkl --audio "data/YourTrack.mp3"
```

| Flag | Default | Meaning |
|---|---|---|
| `--network` | prompts | path to a StyleGAN3 `.pkl` |
| `--audio` | prompts | source audio (mp3/wav/flac) |
| `--fps` / `--size` / `--batch` | 60 / 512 / 16 | output & GPU settings |
| `--seed` | derived from audio | latent walk seed |
| `--walk-speed` | 1.0 | seeds per second of audio |
| `--stems-dir` | — | folder of pre-made stems (skips Demucs) |
| `--map` | legacy routing | `STEM=TARGET[:VECTOR][:STRENGTH][:invert]`, repeatable |
| `--max-seconds` | — | render only the first N seconds (fast previews) |
| `--out` | `./<track>.mp4` | output path |
| `--check` | — | preflight only |

Example custom patch:

```
python culture_shock.py --network myface.pkl --audio "data/YBR.mp3" ^
    --map drums=psi --map drums=coarse --map bass=vector:age:1.2 ^
    --map vocals=vector:mouth_ratio:1.5:invert --walk-speed 0.5 --seed 42
```

## Mixer GUI

```
python mixer_gui.py ["data/YourTrack.mp3"] [--smoke]
```

Pick network + audio, **Load stems**, route each stem, **Preview** (first 15 s)
or **Render Full**. `--smoke` runs a headless self-test (auto-load, preview,
exit). Output mp4 lands in the project root, named after the track.

## Also included

- `vector_explorer.py` — quick-and-dirty slider GUI for eyeballing latent
  direction vectors on still images.
- `vectors/` — ~45 pre-extracted latent direction `.npy` vectors (age, gender,
  smile, hair, eyes, mouth, pose, …) for FFHQ-space models.
- `legacy.py` — converts TF-era StyleGAN checkpoints to current pickle format.

## Environment

- Windows, RTX-class GPU. Python venv (not conda — `setuptools>=72` /
  `pkg_resources` issues). `pip install -r requirements.txt`.
- `torch 2.7.1+cu118`, `torchaudio`, `torchvision` — the trio must match
  exactly or you get DLL load errors on Windows.
- fp16/AMP does **not** work with StyleGAN3's custom CUDA ops (`upfirdn2d`,
  `bias_act` hard-assert float32). Don't bother.

### Windows: compiling the custom CUDA ops

StyleGAN3 JIT-compiles its CUDA ops on first synthesis. On Windows this needs
a real MSVC + CUDA toolkit environment, and there are two traps:

1. **`The input line is too long`** from `vcvars64.bat` — your PATH is too fat
   for cmd's env limit. Launch with a minimal PATH first.
2. **CUDA header mismatch** (`cublasLt.h not found` etc.) — the toolkit picked
   up must match your torch build. For `+cu118`, set `CUDA_HOME` to the v11.8
   toolkit; its supported compiler pairing is **VS2019 Build Tools**.

Launcher template (adjust paths to your installs):

```bat
@echo off
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;<your-ffmpeg-dir>;%CUDA_HOME%\bin"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d <project-dir>
venv\Scripts\pythonw.exe mixer_gui.py %*
```

Model pickles are not in the repo (GitHub size limits). Grab official ones from
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) or train your own.

## Credits & license

`dnnlib/`, `torch_utils/`, `training/` and `legacy.py` are from
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3), © NVIDIA Corporation,
under the [NVIDIA Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
(non-commercial research use). The audio-reactive layer (`stylegan3_audio.py`,
`culture_shock.py`, `mixer_gui.py`) by Art Nixon (AtomicNixon), with Bob and
Verdent.
