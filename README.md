# Stylegan3-Audio

Music-reactive video generation from StyleGAN3 models. Feed it an MP3, get a
video of a continuous latent-space walk where each instrument stem drives a
different visual property of the output.

`culture_shock.py` splits the audio into four stems (drums / bass / other /
vocals) using [Demucs](https://github.com/facebookresearch/demucs), then renders
a SLERP-interpolated walk between random seeds with the stems modulating how
strongly the walk expresses itself:

| Stem | Drives |
|---|---|
| **drums** | truncation psi (0.5–1.0) + coarse style layers (identity/pose) |
| **bass** | mid style layers (facial structure) |
| **other** | fine style layers (texture/detail) |
| **vocals** | mouth-open vector magnitude (lip-sync feel) |

The latent walk itself is slow and continuous, deliberately not beat-synced —
the music shapes *intensity*, not *position*.

## Usage

```
python culture_shock.py --network ffhq.pkl --audio "data/YourTrack.mp3"
```

| Flag | Default | Meaning |
|---|---|---|
| `--network` | prompts | path to a StyleGAN3 `.pkl` |
| `--audio` | prompts | source MP3 |
| `--fps` | 60 | output frame rate |
| `--size` | 512 | output resolution |
| `--batch` | 16 | GPU synthesis batch size |

First run on a new track triggers Demucs separation (~1–2 min), cached in
`data/<name>_stems/`. Output: `data/<name>.mp4`.

## Also included

- `vector_explorer.py` — Tkinter GUI with sliders for every latent direction
  vector in `vectors/`, plus PSI, network and seed pickers, live preview.
- `vectors/` — ~45 pre-extracted latent direction `.npy` vectors (age, gender,
  smile, hair, eyes, mouth, pose, …) for FFHQ-space models.
- `legacy.py` — converts TF-era StyleGAN checkpoints to current pickle format.

## Environment

- Windows, RTX-class GPU. Python venv (not conda — `setuptools>=72` /
  `pkg_resources` issues).
- `torch 2.7.1+cu118`, `torchaudio 2.7.1+cu118`, `torchvision 0.22.1+cu118` —
  versions must match exactly or you get DLL load errors on Windows.
- `demucs 4.0.1` for stem separation.
- fp16/AMP does **not** work with StyleGAN3's custom CUDA ops (`upfirdn2d`,
  `bias_act` hard-assert float32). Don't bother.

Model pickles are not in the repo (GitHub size limits). Grab official ones from
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) or train your own.

## Credits & license

`dnnlib/`, `torch_utils/`, `training/` and `legacy.py` are from
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3), © NVIDIA Corporation,
under the [NVIDIA Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
(non-commercial research use). The audio-reactive layer (`culture_shock.py`,
`vector_explorer.py`) by Art Nixon (AtomicNixon), with Bob and Verdent.
