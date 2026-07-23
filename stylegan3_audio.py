"""stylegan3_audio — core library for music-reactive StyleGAN3 video.

The patchbay model: audio stems are discovered (DAW exports via stems_dir, or
Demucs separation as fallback), converted to per-frame amplitude envelopes,
and routed to visual targets via StemMapping entries:

    target 'psi'     — envelope drives truncation psi (psi_base..psi_base+psi_range)
    target 'coarse'  — style-mix toward the second walk on coarse layers
    target 'mid'     — style-mix on mid layers
    target 'fine'    — style-mix on fine layers
    target 'vector'  — add a named latent direction vector, scaled by envelope

Default mappings reproduce the original culture_shock behavior exactly:
drums->psi + coarse, bass->mid, other->fine, vocals->vector(mouth_ratio, 1.5, inverted).

Frontends (CLI, GUI) should only ever call: preflight(), discover_stems(),
build_envelopes(), render().
"""
from __future__ import annotations

import glob
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# torch imported lazily inside functions that need it, so that cheap operations
# (stem discovery, envelope building for GUI thumbnails) don't pay CUDA startup.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGETS = ('psi', 'coarse', 'mid', 'fine', 'vector', 'none')


@dataclass
class StemMapping:
    stem: str                    # stem name, e.g. 'drums' or 'Culture Shock (Synth)'
    target: str                  # one of TARGETS
    vector: str | None = None    # vector name (basename, no .npy) when target == 'vector'
    strength: float = 1.0
    invert: bool = False         # vector: negate direction; layers: use (1 - envelope)

    def validate(self, errors: list[str]) -> None:
        if self.target not in TARGETS:
            errors.append(f"mapping '{self.stem}': unknown target '{self.target}'")
        if self.target == 'vector' and not self.vector:
            errors.append(f"mapping '{self.stem}': target 'vector' needs a vector name")


def legacy_mappings() -> list[StemMapping]:
    return [
        StemMapping('drums', 'psi'),
        StemMapping('drums', 'coarse'),
        StemMapping('bass', 'mid'),
        StemMapping('other', 'fine'),
        StemMapping('vocals', 'vector', vector='mouth_ratio', strength=1.5, invert=True),
    ]


@dataclass
class RenderConfig:
    network: str = ''
    audio: str = ''
    fps: int = 60
    size: int = 512
    batch: int = 16
    seed: int | None = None          # None -> legacy: derived from longest stem's sample count
    walk_speed: float = 1.0          # seeds per second of audio (legacy: 1.0)
    psi_base: float = 0.5
    psi_range: float = 0.5
    base_psi_build: float = 0.7      # truncation applied when building the base walk
    mix_psi: float = 0.75            # truncation for the second (style-mix source) walk
    layer_coarse: tuple[int, int] = (0, 4)
    layer_mid: tuple[int, int] = (4, 8)   # fine = (8, num_ws)
    mappings: list[StemMapping] = field(default_factory=legacy_mappings)
    stems_dir: str | None = None     # explicit stem folder (DAW exports); overrides demucs
    out: str | None = None           # default: ./<audio_base>.mp4 (project root)
    data_dir: str = 'data'
    vectors_dir: str = 'vectors'
    save_envelope_plots: bool = True
    max_seconds: float | None = None  # truncate render for previews/smoke tests

    @property
    def audio_base(self) -> str:
        return os.path.splitext(os.path.basename(self.audio))[0]

    @property
    def out_path(self) -> str:
        return self.out or (self.audio_base + '.mp4')


# ---------------------------------------------------------------------------
# Preflight — fail fast, fail with sentences
# ---------------------------------------------------------------------------

def preflight(cfg: RenderConfig) -> list[str]:
    """Return a list of human-readable problems. Empty list == good to go."""
    errors = []
    if not cfg.network:
        errors.append("no network specified")
    elif not os.path.exists(cfg.network):
        errors.append(f"network file not found: {cfg.network}")
    if not cfg.audio:
        errors.append("no audio file specified")
    elif not os.path.exists(cfg.audio):
        errors.append(f"audio file not found: {cfg.audio}")
    if shutil.which('ffmpeg') is None:
        errors.append("ffmpeg not found on PATH — install it or add it to PATH")
    if cfg.stems_dir and not os.path.isdir(cfg.stems_dir):
        errors.append(f"stems dir not found: {cfg.stems_dir}")
    for m in cfg.mappings:
        m.validate(errors)
        if m.target == 'vector' and m.vector:
            vpath = os.path.join(cfg.vectors_dir, m.vector + '.npy')
            if not os.path.exists(vpath):
                errors.append(f"vector file not found: {vpath}")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available to torch — this pipeline requires a GPU")
    except Exception as ex:  # noqa: BLE001
        errors.append(f"torch failed to import: {ex}")
    return errors


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

DEMUCS_STEMS = ('drums', 'bass', 'other', 'vocals')
AUDIO_EXTS = ('.wav', '.mp3', '.flac')


def _demucs_dir(cfg: RenderConfig) -> str:
    return os.path.join(cfg.data_dir, cfg.audio_base + '_stems')


def discover_stems(cfg: RenderConfig) -> dict[str, str]:
    """Return {stem_name: path}. Explicit stems_dir wins; else demucs cache."""
    if cfg.stems_dir:
        found = {}
        for ext in AUDIO_EXTS:
            for p in sorted(glob.glob(os.path.join(cfg.stems_dir, '*' + ext))):
                name = os.path.splitext(os.path.basename(p))[0]
                found.setdefault(name, p)  # prefer first ext hit (wav sorts before mp3 alphabetically? no—use ext order)
        return found
    d = _demucs_dir(cfg)
    return {s: os.path.join(d, s + '.wav')
            for s in DEMUCS_STEMS
            if os.path.exists(os.path.join(d, s + '.wav'))}


def separate_stems(cfg: RenderConfig, log=print) -> dict[str, str]:
    """Run demucs if the cache is missing; return discovered stems."""
    stems = discover_stems(cfg)
    if stems:
        return stems
    if cfg.stems_dir:
        raise RuntimeError(f"no audio files found in stems dir: {cfg.stems_dir}")
    d = _demucs_dir(cfg)
    log(f"Stems not found in '{d}' — running demucs (a minute or two)…")
    os.makedirs(d, exist_ok=True)
    out_root = os.path.join(cfg.data_dir, 'demucs_out')
    result = subprocess.run(
        [sys.executable, '-m', 'demucs', '--out', out_root, cfg.audio],
        capture_output=False, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed (exit {result.returncode}) on {cfg.audio}")
    track_dir = os.path.join(out_root, 'htdemucs', cfg.audio_base)
    if not os.path.isdir(track_dir):
        candidates = glob.glob(os.path.join(out_root, '*', cfg.audio_base))
        if candidates:
            track_dir = candidates[0]
        else:
            raise RuntimeError(f"demucs output not found under {out_root}")
    for s in DEMUCS_STEMS:
        src = os.path.join(track_dir, s + '.wav')
        if os.path.exists(src):
            shutil.move(src, os.path.join(d, s + '.wav'))
        else:
            log(f"  warning: expected stem missing from demucs output: {src}")
    stems = discover_stems(cfg)
    if not stems:
        raise RuntimeError("demucs ran but no stems were produced")
    return stems


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

def build_envelopes(stem_paths: dict[str, str], fps: int,
                    log=print) -> tuple[dict[str, np.ndarray], int, float]:
    """Per-frame mean-|amplitude| envelope for each stem, normalized to [0,1].

    Returns (envelopes, frames, duration_seconds). Frame count follows the
    longest stem (legacy behavior).
    """
    import librosa
    raw = {}
    max_samples, ref_rate = 0, None
    for name, path in stem_paths.items():
        if not os.path.exists(path):
            log(f"  warning: stem file missing: {path}")
            continue
        y, rate = librosa.load(path, sr=None, mono=True)
        raw[name] = (y, rate)
        if y.shape[0] > max_samples:
            max_samples, ref_rate = y.shape[0], rate
    if max_samples == 0:
        raise RuntimeError("no stem files could be loaded")

    duration = max_samples / ref_rate
    frames = int(np.ceil(duration * fps))

    envelopes = {}
    for name, (y, rate) in raw.items():
        signal = np.abs(y)
        stem_frames = int(np.ceil(signal.shape[0] / rate * fps))
        spf = signal.shape[0] / stem_frames
        env = np.array([
            np.mean(signal[int(round(f * spf)):int(round((f + 1) * spf))])
            for f in range(stem_frames)
        ], dtype=np.float32)
        env /= max(env.max(), 1e-8)
        if stem_frames < frames:
            env = np.pad(env, (0, frames - stem_frames))
        envelopes[name] = env[:frames]
    # legacy seed derivation rides along
    envelopes['__max_samples__'] = np.array([max_samples], dtype=np.int64)
    return envelopes, frames, duration


def save_envelope_plots(envelopes: dict[str, np.ndarray], audio_base: str,
                        data_dir: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(data_dir, exist_ok=True)
    for name, env in envelopes.items():
        if name.startswith('__'):
            continue
        safe = ''.join(c if c not in '\\/:*?"<>|' else '_' for c in name)
        out = os.path.join(data_dir, f'{audio_base}_{safe}.png')
        if os.path.exists(out):
            continue
        plt.figure(figsize=(8, 3)); plt.title(name); plt.plot(env)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def load_or_build_envelopes(cfg: 'RenderConfig', stem_paths: dict[str, str],
                            log=print) -> tuple[dict[str, np.ndarray], int, float]:
    """build_envelopes with an npz cache in data_dir, invalidated when any
    stem file is newer than the cache."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    cache = os.path.join(cfg.data_dir, f'{cfg.audio_base}_envelopes_{cfg.fps}fps.npz')
    newest = max(os.path.getmtime(p) for p in stem_paths.values() if os.path.exists(p))
    if os.path.exists(cache) and os.path.getmtime(cache) >= newest:
        z = np.load(cache)
        envs = {k: z[k] for k in z.files if not k.startswith('__')}
        envs['__max_samples__'] = z['__max_samples__']
        frames = int(z['__frames__'][0])
        duration = float(z['__duration__'][0])
        log(f"Envelopes: cache hit ({os.path.basename(cache)})")
        return envs, frames, duration
    envs, frames, duration = build_envelopes(stem_paths, cfg.fps, log=log)
    save = dict(envs)
    save['__frames__'] = np.array([frames])
    save['__duration__'] = np.array([duration], dtype=np.float64)
    np.savez(cache, **save)
    log(f"Envelopes: built and cached ({os.path.basename(cache)})")
    return envs, frames, duration


def ensure_network(path: str, log=print) -> str:
    """Return a new-format network pkl path, converting TF-era pickles once.

    New-format pkls (dict with 'G_ema') pass straight through. Old-style
    pickles are converted via legacy.load_network_pkl and written alongside
    the original as '<name>-new.pkl' — original untouched. If the converted
    file already exists, it is used without reconverting."""
    try:
        with open(path, 'rb') as f:
            head = pickle.load(f)
        if isinstance(head, dict) and 'G_ema' in head:
            return path
        reason = f'old container type: {type(head).__name__}'
    except Exception as ex:  # noqa: BLE001 — TF-era pkls raise on plain unpickle
        reason = f'plain unpickle failed: {type(ex).__name__}'
    new_path = os.path.splitext(path)[0] + '-new.pkl'
    if os.path.exists(new_path):
        log(f"Old-format pkl ({reason}) — using existing conversion: {new_path}")
        return new_path
    log(f"Old-format pkl ({reason}) — converting once via legacy.py …")
    import legacy as _legacy
    with open(path, 'rb') as f:
        data = _legacy.load_network_pkl(f)
    with open(new_path, 'wb') as f:
        pickle.dump(data, f)
    log(f"Converted → {new_path} (original untouched)")
    return new_path


# ---------------------------------------------------------------------------
# Latent math (identical to legacy culture_shock)
# ---------------------------------------------------------------------------

def _slerp(t, w0, w1):
    import torch
    w0f = w0.reshape(-1).float()
    w1f = w1.reshape(-1).float()
    n0 = w0f / (torch.norm(w0f) + 1e-8)
    n1 = w1f / (torch.norm(w1f) + 1e-8)
    dot = torch.clamp(torch.dot(n0, n1), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return ((1 - t) * w0f + t * w1f).reshape(w0.shape)
    s = torch.sin(omega)
    return ((torch.sin((1 - t) * omega) / s) * w0f +
            (torch.sin(t * omega) / s) * w1f).reshape(w0.shape)


def _build_walk(G, w_avg, n_seeds, total_frames, seed, psi, device):
    import torch
    num_ws = G.num_ws
    rng = np.random.RandomState(seed)
    ws = []
    for _ in range(n_seeds):
        z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            w = G.mapping(z.float(), None).float()
        w = w_avg + psi * (w - w_avg)
        ws.append(w.squeeze(0))
    frames_per_step = total_frames / n_seeds
    seq = []
    for i in range(total_frames):
        step = i / frames_per_step
        idx0 = int(step) % n_seeds
        idx1 = (idx0 + 1) % n_seeds
        t = step - int(step)
        import torch as _t
        interp = _t.stack([_slerp(t, ws[idx0][l], ws[idx1][l]) for l in range(num_ws)])
        seq.append(interp)
    return seq


def _normalize_vector(v: np.ndarray, w_avg_np: np.ndarray) -> np.ndarray:
    return v * np.std(w_avg_np) / (np.std(v) + 1e-8) + np.mean(w_avg_np) - np.mean(v)


def _mix_styles(wa, wb, layer_range, v):
    w = np.copy(wa)
    for i in layer_range:
        w[i] = wa[i] * (1 - v) + wb[i] * v
    return w


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(cfg: RenderConfig, progress=None, log=print) -> str:
    """Full pipeline. Returns the output mp4 path. progress(frac, msg) optional."""
    import torch
    import PIL.Image

    def report(frac, msg):
        if progress:
            progress(frac, msg)

    problems = preflight(cfg)
    if problems:
        raise RuntimeError("preflight failed:\n  " + "\n  ".join(problems))

    os.makedirs(cfg.data_dir, exist_ok=True)
    stems = separate_stems(cfg, log=log)
    log(f"Stems: {', '.join(sorted(stems))}")
    report(0.02, 'stems ready')

    envelopes, frames, duration = load_or_build_envelopes(cfg, stems, log=log)
    max_samples = int(envelopes.pop('__max_samples__')[0])
    if cfg.max_seconds is not None and cfg.max_seconds < duration:
        frames = int(np.ceil(cfg.max_seconds * cfg.fps))
        envelopes = {k: v[:frames] for k, v in envelopes.items()}
        log(f"Truncating render to {cfg.max_seconds:.1f}s ({frames} frames)")
    if cfg.save_envelope_plots:
        save_envelope_plots(envelopes, cfg.audio_base, cfg.data_dir)
    report(0.05, 'envelopes built')

    # warn about mappings pointing at stems we don't have
    active = []
    for m in cfg.mappings:
        if m.target == 'none':
            continue
        if m.stem not in envelopes:
            log(f"  warning: mapping '{m.stem}' -> {m.target}: no such stem, skipping")
            continue
        active.append(m)
    if not active:
        log("  warning: no active mappings — output will be the bare latent walk")

    net_path = ensure_network(cfg.network, log=log)
    log(f"Loading network: {net_path}")
    with open(net_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    device = torch.device('cuda')
    w_avg = G.mapping.w_avg.float()
    w_avg_np = w_avg.cpu().numpy()
    num_ws, w_dim = G.num_ws, G.w_dim
    report(0.08, 'network loaded')

    seed = cfg.seed if cfg.seed is not None else max_samples
    n_seeds = max(2, int(np.ceil(duration * cfg.walk_speed)))
    log(f"Audio: {cfg.audio_base} | {duration:.1f}s | {frames} frames @ {cfg.fps}fps | "
        f"{n_seeds} seeds | seed={seed}")

    log("Building base walk…")
    base_seq = _build_walk(G, w_avg, n_seeds, frames, seed, cfg.base_psi_build, device)
    log("Building mix walk…")
    mix_seq = _build_walk(G, w_avg, n_seeds, frames, seed + 1, cfg.mix_psi, device)
    report(0.15, 'walks built')

    # preload vectors
    vectors = {}
    for m in active:
        if m.target == 'vector' and m.vector not in vectors:
            v = np.load(os.path.join(cfg.vectors_dir, m.vector + '.npy'))
            if m.invert:
                v = -v
            vectors[m.vector] = _normalize_vector(v, w_avg_np)

    fine_range = range(cfg.layer_mid[1], num_ws)
    layer_ranges = {
        'coarse': range(*cfg.layer_coarse),
        'mid': range(*cfg.layer_mid),
        'fine': fine_range,
    }

    log("Pre-computing per-frame W vectors…")
    all_w = np.empty((frames, num_ws, w_dim), dtype=np.float32)
    for i in range(frames):
        base_w = base_seq[i].cpu().numpy()
        # psi from mappings (legacy: drums)
        psi_drive = 0.0
        has_psi = False
        for m in active:
            if m.target == 'psi':
                has_psi = True
                e = envelopes[m.stem][i]
                psi_drive += (1.0 - e if m.invert else e) * m.strength
        if has_psi:
            psi = cfg.psi_base + min(max(psi_drive, 0.0), 1.0) * cfg.psi_range
            base_w = w_avg_np + (base_w - w_avg_np) * psi

        mix_w = mix_seq[i].cpu().numpy()
        mix_w = w_avg_np + (mix_w - w_avg_np) * cfg.mix_psi

        w = base_w
        for m in active:
            if m.target in layer_ranges:
                e = envelopes[m.stem][i]
                v = (1.0 - e if m.invert else e) * m.strength
                w = _mix_styles(w, mix_w, layer_ranges[m.target], min(max(v, 0.0), 1.0))
            elif m.target == 'vector':
                e = envelopes[m.stem][i]
                w = w + vectors[m.vector] * e * m.strength
        all_w[i] = w
        if i % 600 == 0:
            report(0.15 + 0.15 * (i / frames), f'precompute {i}/{frames}')
    del base_seq, mix_seq
    report(0.30, 'W tensors ready')

    # ---- synthesis → ffmpeg pipe ----
    out_path = cfg.out_path
    log(f"Rendering → {out_path} (batch={cfg.batch})")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'warning',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{cfg.size}x{cfg.size}', '-pix_fmt', 'rgb24',
        '-r', str(cfg.fps),
        '-i', 'pipe:0',
        '-i', cfg.audio,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest', '-map', '0:v', '-map', '1:a',
        out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def convert_frame(img_tensor):
        arr = (img_tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).byte().cpu().numpy()
        if arr.shape[0] != cfg.size or arr.shape[1] != cfg.size:
            arr = np.array(PIL.Image.fromarray(arr).resize((cfg.size, cfg.size),
                                                           PIL.Image.LANCZOS))
        return arr

    executor = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
    total_batches = (frames + cfg.batch - 1) // cfg.batch
    try:
        for b in range(total_batches):
            start = b * cfg.batch
            end = min(start + cfg.batch, frames)
            w_batch = torch.from_numpy(all_w[start:end]).float().to(device)
            with torch.no_grad():
                imgs = G.synthesis(w_batch, noise_mode='const')
            futures = [executor.submit(convert_frame, imgs[k]) for k in range(end - start)]
            for fut in futures:
                try:
                    proc.stdin.write(fut.result().tobytes())
                except (BrokenPipeError, OSError) as ex:
                    raise RuntimeError(
                        "ffmpeg pipe closed mid-render — encoder died "
                        "(disk full? codec problem?). Check ffmpeg output above.") from ex
            report(0.30 + 0.70 * (end / frames), f'render {end}/{frames}')
            log(f"\r  {end}/{frames}  ({end / frames * 100:.0f}%)", )
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
        executor.shutdown()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc} — {out_path} is probably unusable")
    log(f"Done → {out_path}")
    report(1.0, 'done')
    return out_path
