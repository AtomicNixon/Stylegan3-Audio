import argparse
import os
import sys
import subprocess
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import librosa
import torch
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Arguments / interactive prompts
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Culture Shock – StyleGAN3 music video')
parser.add_argument('--network',    default=None,  help='Path to .pkl network file')
parser.add_argument('--audio',      default=None,  help='Path to source MP3 file')
parser.add_argument('--fps',        default=60,    type=int,   help='Output FPS (default 60)')
parser.add_argument('--size',       default=512,   type=int,   help='Output resolution (default 512)')
parser.add_argument('--batch',      default=16,    type=int,   help='GPU batch size (default 16)')
args = parser.parse_args()

if args.network is None:
    default_net = 'ffhq.pkl'
    ans = input(f"Network PKL [{default_net}]: ").strip().strip('"')
    args.network = ans if ans else default_net

if args.audio is None:
    ans = input("Audio MP3 file: ").strip().strip('"')
    args.audio = ans

network_pkl = args.network
audio_path  = args.audio
fps         = args.fps
size        = args.size
batch_size  = args.batch
audio_base  = os.path.splitext(os.path.basename(audio_path))[0]

# ---------------------------------------------------------------------------
# Stem separation (demucs) if stems not already present
# ---------------------------------------------------------------------------
STEM_NAMES = ['drums', 'bass', 'other', 'vocals']
stem_dir   = os.path.join('data', audio_base + '_stems')

def stems_exist(d):
    return os.path.isdir(d) and all(
        os.path.exists(os.path.join(d, s + '.wav')) for s in STEM_NAMES
    )

if not stems_exist(stem_dir):
    print(f"Stems not found in '{stem_dir}'. Running demucs (this may take a minute)...")
    os.makedirs(stem_dir, exist_ok=True)
    demucs_out_root = os.path.join('data', 'demucs_out')
    subprocess.run(
        [sys.executable, '-m', 'demucs', '--out', demucs_out_root, audio_path],
        check=True
    )
    demucs_track = os.path.join(demucs_out_root, 'htdemucs', audio_base)
    for stem in STEM_NAMES:
        src = os.path.join(demucs_track, stem + '.wav')
        dst = os.path.join(stem_dir, stem + '.wav')
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"  Warning: expected stem not found: {src}")
    print("Stem separation complete.")

# ---------------------------------------------------------------------------
# Load stems & build per-frame amplitude envelopes
# ---------------------------------------------------------------------------
audio_env = {}
raw_stems  = {}
max_samples, ref_rate = 0, None

for stem_name in STEM_NAMES:
    stem_path = os.path.join(stem_dir, stem_name + '.wav')
    if not os.path.exists(stem_path):
        print(f"  Warning: stem file missing: {stem_path}")
        continue
    y, rate = librosa.load(stem_path, sr=None, mono=True)
    raw_stems[stem_name] = (y, rate)
    if y.shape[0] > max_samples:
        max_samples, ref_rate = y.shape[0], rate

if max_samples == 0:
    sys.exit("No stem files loaded – aborting.")

duration = max_samples / ref_rate
frames   = int(np.ceil(duration * fps))

for stem_name, (y, rate) in raw_stems.items():
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
    audio_env[stem_name] = env

os.makedirs('data', exist_ok=True)
for stem_name, env in audio_env.items():
    plt.figure(figsize=(8, 3)); plt.title(stem_name); plt.plot(env)
    plt.tight_layout()
    plt.savefig(os.path.join('data', f'{audio_base}_{stem_name}.png'))
    plt.close()

# ---------------------------------------------------------------------------
# StyleGAN3 model loading
# ---------------------------------------------------------------------------
print(f"Loading network: {network_pkl}")
with open(network_pkl, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

device = torch.device('cuda')
w_avg  = G.mapping.w_avg.float()
num_ws = G.num_ws

# ---------------------------------------------------------------------------
# Latent utilities
# ---------------------------------------------------------------------------
def slerp(t, w0, w1):
    w0f = w0.reshape(-1).float()
    w1f = w1.reshape(-1).float()
    n0  = w0f / (torch.norm(w0f) + 1e-8)
    n1  = w1f / (torch.norm(w1f) + 1e-8)
    dot = torch.clamp(torch.dot(n0, n1), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return ((1 - t) * w0f + t * w1f).reshape(w0.shape)
    s = torch.sin(omega)
    return ((torch.sin((1 - t) * omega) / s) * w0f +
            (torch.sin(      t  * omega) / s) * w1f).reshape(w0.shape)


def build_slerp_sequence(n_seeds, total_frames, seed, psi):
    rng = np.random.RandomState(seed)
    ws  = []
    for _ in range(n_seeds):
        z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            w = G.mapping(z.float(), None).float()
        w = w_avg + psi * (w - w_avg)
        ws.append(w.squeeze(0))
    fps_per_step = total_frames / n_seeds
    sequence = []
    for i in range(total_frames):
        step = i / fps_per_step
        idx0 = int(step) % n_seeds
        idx1 = (idx0 + 1) % n_seeds
        t    = step - int(step)
        interp = torch.stack([slerp(t, ws[idx0][l], ws[idx1][l])
                              for l in range(num_ws)])
        sequence.append(interp)
    return sequence


def normalize_vector(v):
    w_avg_np = w_avg.cpu().numpy()
    return v * np.std(w_avg_np) / (np.std(v) + 1e-8) + np.mean(w_avg_np) - np.mean(v)


def mix_styles(wa, wb, layer_range, v):
    w = np.copy(wa)
    for i in layer_range:
        w[i] = wa[i] * (1 - v) + wb[i] * v
    return w


# ---------------------------------------------------------------------------
# Pre-compute ALL per-frame W tensors (CPU, fast)
# ---------------------------------------------------------------------------
seconds = int(np.ceil(duration))
print(f"Audio: {audio_base}  |  {duration:.1f}s  |  {frames} frames @ {fps}fps  |  {seconds} seeds")

print(f"Building base SLERP sequence ({seconds} seeds, {frames} frames)...")
base_seq = build_slerp_sequence(seconds, frames, seed=max_samples,     psi=0.7)
print(f"Building mix  SLERP sequence ({seconds} seeds, {frames} frames)...")
mix_seq  = build_slerp_sequence(seconds, frames, seed=max_samples + 1, psi=0.75)

mouth_vec = normalize_vector(-np.load('vectors/mouth_ratio.npy'))
age_vec = normalize_vector(-np.load('vectors/age.npy'))
w_avg_np  = w_avg.cpu().numpy()

drums_env  = audio_env.get('drums',  np.zeros(frames))
bass_env   = audio_env.get('bass',   np.zeros(frames))
other_env  = audio_env.get('other',  np.zeros(frames))
vocals_env = audio_env.get('vocals', np.zeros(frames))

print("Pre-computing per-frame W vectors...")
all_w = np.empty((frames, num_ws, 512), dtype=np.float32)
for i in range(frames):
    base_w = base_seq[i].cpu().numpy()
    # PSI driven by drums amplitude (maps [0,1] → [0.5, 1.0])
    psi    = 0.5 + drums_env[i] * 0.5
    base_w = w_avg_np + (base_w - w_avg_np) * psi

    mix_w  = mix_seq[i].cpu().numpy()
    mix_w  = w_avg_np + (mix_w - w_avg_np) * 0.75

    # Style mixing: drums→coarse, bass→mid, other→fine
    w = mix_styles(base_w, mix_w, range(0, 4),    drums_env[i])
    w = mix_styles(w,      mix_w, range(4, 8),    bass_env[i])
    w = mix_styles(w,      mix_w, range(8, num_ws), other_env[i])

    # Mouth vector driven by vocals
    w += mouth_vec * vocals_env[i] * 1.5
#     w += age_vec * bass_env[i] * 1.5
    all_w[i] = w

# Free SLERP sequences from memory
del base_seq, mix_seq

# ---------------------------------------------------------------------------
# Batched GPU synthesis → direct ffmpeg pipe
# ---------------------------------------------------------------------------
mp4_filename = os.path.join('data', audio_base + '.mp4')
print(f"Rendering → {mp4_filename}  (batch={batch_size})")

ffmpeg_cmd = [
    'ffmpeg', '-y', '-loglevel', 'warning',
    '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{size}x{size}', '-pix_fmt', 'rgb24',
    '-r', str(fps),
    '-i', 'pipe:0',
    '-i', audio_path,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-c:a', 'aac', '-b:a', '192k',
    '-shortest', '-map', '0:v', '-map', '1:a',
    mp4_filename
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def convert_frame(img_tensor):
    """Convert a CHW float GPU tensor to a resized HWC uint8 numpy array (CPU)."""
    arr = (img_tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).byte().cpu().numpy()
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = np.array(PIL.Image.fromarray(arr).resize((size, size), PIL.Image.LANCZOS))
    return arr

n_workers = max(2, os.cpu_count() // 2)
executor  = ThreadPoolExecutor(max_workers=n_workers)

total_batches = (frames + batch_size - 1) // batch_size

for b in range(total_batches):
    start = b * batch_size
    end   = min(start + batch_size, frames)
    w_batch = torch.from_numpy(all_w[start:end]).float().to(device)

    with torch.no_grad():
        imgs = G.synthesis(w_batch, noise_mode='const')

    futures = [executor.submit(convert_frame, imgs[k]) for k in range(end - start)]
    for fut in futures:
        ffmpeg_proc.stdin.write(fut.result().tobytes())

    pct = (end / frames) * 100
    print(f"\r  {end}/{frames}  ({pct:.0f}%)", end='', flush=True)

print()
ffmpeg_proc.stdin.close()
ffmpeg_proc.wait()
executor.shutdown()
print(f"Done → {mp4_filename}")
