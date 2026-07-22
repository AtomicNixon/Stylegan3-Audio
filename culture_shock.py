"""Culture Shock — music-reactive StyleGAN3 video (CLI).

Thin wrapper over stylegan3_audio. With no new flags this behaves exactly like
the original script (same defaults, same seed derivation, same look), except
the output mp4 lands in the project root, named after the track.

Mapping syntax (repeatable):
    --map STEM=TARGET[:VECTOR][:STRENGTH][:invert]
    TARGET: psi | coarse | mid | fine | vector | none
Examples:
    --map drums=psi --map drums=coarse --map bass=mid
    --map vocals=vector:mouth_ratio:1.5:invert
    --map "Culture Shock (Synth)=fine:0.8"        (DAW stems via --stems-dir)
"""
import argparse
import sys

from stylegan3_audio import RenderConfig, StemMapping, legacy_mappings, preflight, render


def parse_map(spec: str) -> StemMapping:
    if '=' not in spec:
        raise argparse.ArgumentTypeError(f"bad --map '{spec}': expected STEM=TARGET[...]")
    stem, rest = spec.split('=', 1)
    parts = rest.split(':')
    target = parts[0].strip()
    vector = None
    strength = 1.0
    invert = False
    for p in parts[1:]:
        p = p.strip()
        if p.lower() == 'invert':
            invert = True
        else:
            try:
                strength = float(p)
            except ValueError:
                vector = p
    return StemMapping(stem.strip(), target, vector=vector, strength=strength, invert=invert)


def main() -> int:
    ap = argparse.ArgumentParser(description='Culture Shock — StyleGAN3 music video')
    ap.add_argument('--network', default=None, help='Path to .pkl network file')
    ap.add_argument('--audio', default=None, help='Path to source audio file')
    ap.add_argument('--fps', default=60, type=int)
    ap.add_argument('--size', default=512, type=int)
    ap.add_argument('--batch', default=16, type=int, help='GPU batch size')
    ap.add_argument('--seed', default=None, type=int,
                    help='Walk seed (default: derived from audio, legacy behavior)')
    ap.add_argument('--walk-speed', default=1.0, type=float,
                    help='Seeds per second of audio (default 1.0)')
    ap.add_argument('--out', default=None, help='Output mp4 path (default ./<track>.mp4)')
    ap.add_argument('--stems-dir', default=None,
                    help='Folder of pre-made stems (e.g. DAW exports); skips demucs')
    ap.add_argument('--map', action='append', type=parse_map, default=None,
                    metavar='STEM=TARGET[:VECTOR][:STRENGTH][:invert]',
                    help='Stem routing; repeatable. Omit for legacy default mapping.')
    ap.add_argument('--max-seconds', default=None, type=float,
                    help='Render only the first N seconds (previews/smoke tests)')
    ap.add_argument('--no-plots', action='store_true', help='Skip envelope PNGs')
    ap.add_argument('--check', action='store_true', help='Preflight only, no render')
    args = ap.parse_args()

    if args.network is None:
        ans = input("Network PKL [ffhq.pkl]: ").strip().strip('"')
        args.network = ans if ans else 'ffhq.pkl'
    if args.audio is None:
        args.audio = input("Audio file: ").strip().strip('"')

    cfg = RenderConfig(
        network=args.network,
        audio=args.audio,
        fps=args.fps,
        size=args.size,
        batch=args.batch,
        seed=args.seed,
        walk_speed=args.walk_speed,
        out=args.out,
        stems_dir=args.stems_dir,
        mappings=args.map if args.map else legacy_mappings(),
        save_envelope_plots=not args.no_plots,
        max_seconds=args.max_seconds,
    )

    if args.check:
        problems = preflight(cfg)
        if problems:
            print("Preflight problems:")
            for p in problems:
                print("  -", p)
            return 1
        print("Preflight OK.")
        return 0

    try:
        render(cfg)
    except RuntimeError as ex:
        print(f"\nERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
