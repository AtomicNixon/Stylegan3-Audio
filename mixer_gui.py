"""Stylegan3-Audio Mixer — the patchbay, with a face.

A row per discovered stem: envelope thumbnail, target dropdown, vector picker,
strength, invert. Global knobs up top. Preview renders the first 15 seconds.
Everything runs through stylegan3_audio.render() on a worker thread.
"""
import os
import glob
import queue
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from stylegan3_audio import (
    DEMUCS_STEMS, RenderConfig, StemMapping, build_envelopes,
    legacy_mappings, preflight, render, separate_stems,
)

BG = '#1e1e1e'
BG2 = '#2a2a2a'
FG = '#ddd'
ACCENT = '#7cf'
TARGETS = ['none', 'psi', 'coarse', 'mid', 'fine', 'vector']
THUMB_W, THUMB_H = 240, 44

LEGACY_DEFAULTS = {
    'drums': ('psi', None, 1.0, False),      # + coarse, handled specially below
    'bass': ('mid', None, 1.0, False),
    'other': ('fine', None, 1.0, False),
    'vocals': ('vector', 'mouth_ratio', 1.5, True),
}


def envelope_thumbnail(env: np.ndarray, w=THUMB_W, h=THUMB_H) -> Image.Image:
    img = Image.new('RGB', (w, h), BG2)
    draw = ImageDraw.Draw(img)
    if env.size:
        idx = np.linspace(0, env.size - 1, w).astype(int)
        ys = env[idx]
        for x in range(w):
            bar = int(ys[x] * (h - 2))
            draw.line([(x, h - 1), (x, h - 1 - bar)], fill='#5ab')
    return img


class StemRow:
    def __init__(self, parent, name, env, vector_names, on_change=None):
        self.name = name
        self.frame = tk.Frame(parent, bg=BG)
        self.frame.pack(fill='x', padx=8, pady=3)

        self._thumb = ImageTk.PhotoImage(envelope_thumbnail(env))
        tk.Label(self.frame, image=self._thumb, bg=BG).pack(side='left', padx=(0, 8))

        tk.Label(self.frame, text=name, bg=BG, fg=FG, width=22, anchor='w',
                 font=('Consolas', 9)).pack(side='left')

        self.target_var = tk.StringVar(value='none')
        self.target_cb = ttk.Combobox(self.frame, textvariable=self.target_var,
                                      values=TARGETS, state='readonly', width=7)
        self.target_cb.pack(side='left', padx=4)
        self.target_cb.bind('<<ComboboxSelected>>', lambda e: self._sync_state())

        self.vector_var = tk.StringVar(value='')
        self.vector_cb = ttk.Combobox(self.frame, textvariable=self.vector_var,
                                      values=vector_names, state='disabled', width=16)
        self.vector_cb.pack(side='left', padx=4)

        self.strength_var = tk.DoubleVar(value=1.0)
        tk.Scale(self.frame, variable=self.strength_var, from_=0.0, to=3.0,
                 resolution=0.05, orient='horizontal', bg=BG, fg=FG, length=120,
                 troughcolor='#444', highlightthickness=0,
                 label=None, showvalue=False).pack(side='left', padx=4)
        tk.Label(self.frame, textvariable=self.strength_var, bg=BG, fg=ACCENT,
                 width=5, font=('Consolas', 8)).pack(side='left')

        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.frame, text='invert', variable=self.invert_var,
                       bg=BG, fg=FG, selectcolor=BG2,
                       activebackground=BG).pack(side='left', padx=6)

        # secondary target (for the drums→psi+coarse legacy pattern)
        self.target2_var = tk.StringVar(value='none')
        tk.Label(self.frame, text='+', bg=BG, fg='#777').pack(side='left')
        ttk.Combobox(self.frame, textvariable=self.target2_var, values=TARGETS,
                     state='readonly', width=7).pack(side='left', padx=4)

    def _sync_state(self):
        self.vector_cb.configure(
            state='readonly' if self.target_var.get() == 'vector' else 'disabled')

    def set_defaults(self, target, vector, strength, invert, target2='none'):
        self.target_var.set(target)
        self.vector_var.set(vector or '')
        self.strength_var.set(strength)
        self.invert_var.set(invert)
        self.target2_var.set(target2)
        self._sync_state()

    def mappings(self):
        out = []
        for tvar in (self.target_var, self.target2_var):
            t = tvar.get()
            if t == 'none':
                continue
            out.append(StemMapping(
                self.name, t,
                vector=self.vector_var.get() or None if t == 'vector' else None,
                strength=self.strength_var.get(),
                invert=self.invert_var.get()))
        return out


class MixerApp(tk.Tk):
    def __init__(self, audio=None, smoke=False):
        super().__init__()
        self.smoke = smoke
        self.title('Stylegan3-Audio Mixer')
        self.configure(bg=BG)
        self.geometry('1080x640')

        self.q = queue.Queue()
        self.rows = []
        self._busy = False
        self.vector_names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join('vectors', '*.npy')))

        self._build_top()
        self.stems_frame = tk.Frame(self, bg=BG)
        self.stems_frame.pack(fill='both', expand=True, pady=(6, 0))
        tk.Label(self.stems_frame, text='Load a track to see its stems.',
                 bg=BG, fg='#888').pack(pady=30)
        self._build_bottom()
        if audio:
            self.audio_var.set(audio)
            self.after(400, self._load_stems)
        # tk Spinbox with values= stomps its textvariable at creation; restore defaults
        self.fps_var.set(60)
        self.size_var.set(512)
        self.after(100, self._poll)

    # ---------------- UI construction ----------------
    def _build_top(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill='x', padx=8, pady=8)

        def lab(parent, text):
            tk.Label(parent, text=text, bg=BG, fg=FG).pack(side='left', padx=(10, 2))

        row1 = tk.Frame(top, bg=BG); row1.pack(fill='x')
        lab(row1, 'Network:')
        self.network_var = tk.StringVar()
        pkls = sorted(glob.glob('*.pkl'))
        cb = ttk.Combobox(row1, textvariable=self.network_var, values=pkls,
                          state='readonly', width=30)
        cb.pack(side='left')
        if pkls:
            cb.current(0)

        lab(row1, 'Audio:')
        self.audio_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.audio_var, width=38, bg=BG2, fg=FG,
                 insertbackground=FG).pack(side='left')
        tk.Button(row1, text='…', command=self._pick_audio, bg=BG2, fg=FG,
                  width=3).pack(side='left', padx=2)
        tk.Button(row1, text='Load stems', command=self._load_stems,
                  bg='#345', fg='white').pack(side='left', padx=8)
        lab(row1, 'Stems dir (opt):')
        self.stems_dir_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.stems_dir_var, width=22, bg=BG2, fg=FG,
                 insertbackground=FG).pack(side='left')
        tk.Button(row1, text='…', command=self._pick_stems_dir, bg=BG2, fg=FG,
                  width=3).pack(side='left', padx=2)

        row2 = tk.Frame(top, bg=BG); row2.pack(fill='x', pady=(6, 0))
        lab(row2, 'Seed:')
        self.seed_var = tk.StringVar(value='')
        tk.Entry(row2, textvariable=self.seed_var, width=10, bg=BG2, fg=FG,
                 insertbackground=FG).pack(side='left')
        tk.Button(row2, text='🎲', command=lambda: self.seed_var.set(
            str(random.randint(0, 2**31 - 1))), bg=BG2, fg=FG).pack(side='left', padx=2)
        lab(row2, 'Walk speed:')
        self.walk_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(row2, textvariable=self.walk_var, from_=0.1, to=8.0,
                   increment=0.1, width=5, bg=BG2, fg=FG).pack(side='left')
        lab(row2, 'FPS:')
        self.fps_var = tk.IntVar(value=60)
        tk.Spinbox(row2, textvariable=self.fps_var, values=(24, 30, 60), width=4,
                   bg=BG2, fg=FG).pack(side='left')
        lab(row2, 'Size:')
        self.size_var = tk.IntVar(value=512)
        tk.Spinbox(row2, textvariable=self.size_var, values=(256, 512, 1024),
                   width=5, bg=BG2, fg=FG).pack(side='left')
        lab(row2, 'Batch:')
        self.batch_var = tk.IntVar(value=16)
        tk.Spinbox(row2, textvariable=self.batch_var, from_=1, to=64, width=4,
                   bg=BG2, fg=FG).pack(side='left')
        lab(row2, 'Preview s:')
        self.prev_var = tk.IntVar(value=15)
        tk.Spinbox(row2, textvariable=self.prev_var, from_=5, to=60, width=4,
                   bg=BG2, fg=FG).pack(side='left')
        self.open_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text='open when done', variable=self.open_var, bg=BG,
                       fg=FG, selectcolor=BG2, activebackground=BG).pack(side='left', padx=10)

    def _build_bottom(self):
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill='x', side='bottom', padx=8, pady=8)
        self.preview_btn = tk.Button(bottom, text='Preview', command=self._preview,
                                     bg='#354', fg='white', width=12, state='disabled')
        self.preview_btn.pack(side='left')
        self.render_btn = tk.Button(bottom, text='Render Full', command=self._render_full,
                                    bg='#435', fg='white', width=12, state='disabled')
        self.render_btn.pack(side='left', padx=8)
        self.progress = ttk.Progressbar(bottom, length=380, mode='determinate')
        self.progress.pack(side='left', padx=10)
        self.status_var = tk.StringVar(value='Ready.')
        tk.Label(bottom, textvariable=self.status_var, bg=BG, fg='#aaa',
                 font=('Consolas', 9)).pack(side='left', padx=8)

    # ---------------- actions ----------------
    def _pick_audio(self):
        p = filedialog.askopenfilename(initialdir='data',
                                       filetypes=[('Audio', '*.mp3 *.wav *.flac')])
        if p:
            self.audio_var.set(os.path.relpath(p))

    def _pick_stems_dir(self):
        p = filedialog.askdirectory(initialdir='data')
        if p:
            self.stems_dir_var.set(os.path.relpath(p))

    def _cfg(self, max_seconds=None, out=None):
        seed = self.seed_var.get().strip()
        return RenderConfig(
            network=self.network_var.get(),
            audio=self.audio_var.get(),
            fps=self.fps_var.get(),
            size=self.size_var.get(),
            batch=self.batch_var.get(),
            seed=int(seed) if seed else None,
            walk_speed=self.walk_var.get(),
            stems_dir=self.stems_dir_var.get() or None,
            mappings=[m for r in self.rows for m in r.mappings()],
            max_seconds=max_seconds,
            out=out,
        )

    def _load_stems(self):
        if self._busy:
            return
        cfg = self._cfg()
        if not cfg.audio:
            self.status_var.set('Pick an audio file first.')
            return
        self._set_busy(True, 'Loading stems…')

        def work():
            try:
                stems = separate_stems(cfg, log=lambda m: self.q.put(('status', str(m))))
                envs, frames, duration = build_envelopes(stems, cfg.fps,
                                                         log=lambda m: self.q.put(('status', str(m))))
                envs.pop('__max_samples__', None)
                self.q.put(('stems', envs, duration))
            except Exception as ex:  # noqa: BLE001
                self.q.put(('error', str(ex)))
        threading.Thread(target=work, daemon=True).start()

    def _populate_rows(self, envelopes, duration):
        for w in self.stems_frame.winfo_children():
            w.destroy()
        self.rows = []
        hdr = tk.Frame(self.stems_frame, bg=BG)
        hdr.pack(fill='x', padx=8)
        for text, width in (('envelope', 34), ('stem', 22), ('target', 9),
                            ('vector', 18), ('strength', 18), ('', 8), ('+2nd', 9)):
            tk.Label(hdr, text=text, bg=BG, fg='#777', width=width,
                     anchor='w', font=('Consolas', 8)).pack(side='left')
        is_demucs = set(envelopes) == set(DEMUCS_STEMS)
        for name in sorted(envelopes):
            row = StemRow(self.stems_frame, name, envelopes[name], self.vector_names)
            if is_demucs and name in LEGACY_DEFAULTS:
                t, v, s, inv = LEGACY_DEFAULTS[name]
                row.set_defaults(t, v, s, inv,
                                 target2='coarse' if name == 'drums' else 'none')
            self.rows.append(row)
        self.status_var.set(
            f'{len(self.rows)} stems | {duration:.1f}s'
            + (' | legacy routing applied' if is_demucs else ' | route your stems'))
        self.preview_btn.configure(state='normal')
        self.render_btn.configure(state='normal')

    def _run_render(self, cfg, label):
        self._set_busy(True, f'{label}…')

        def work():
            try:
                out = render(cfg,
                             progress=lambda f, m: self.q.put(('progress', f, m)),
                             log=lambda m: self.q.put(('status', str(m).strip())))
                self.q.put(('done', out))
            except Exception as ex:  # noqa: BLE001
                self.q.put(('error', str(ex)))
        threading.Thread(target=work, daemon=True).start()

    def _preview(self):
        if not self._busy:
            self._run_render(self._cfg(max_seconds=float(self.prev_var.get()),
                                       out='preview.mp4'), 'Preview')

    def _render_full(self):
        if not self._busy:
            self._run_render(self._cfg(), 'Rendering')

    def _set_busy(self, busy, msg=None):
        self._busy = busy
        state = 'disabled' if busy else 'normal'
        for b in (self.preview_btn, self.render_btn):
            b.configure(state=state if self.rows else 'disabled')
        if msg:
            self.status_var.set(msg)

    # ---------------- queue pump ----------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == 'status':
                    if msg[1]:
                        self.status_var.set(msg[1][:110])
                elif kind == 'progress':
                    self.progress['value'] = msg[1] * 100
                    self.status_var.set(msg[2][:110])
                elif kind == 'stems':
                    self._set_busy(False)
                    self._populate_rows(msg[1], msg[2])
                    if self.smoke:
                        self.after(500, self._preview)
                elif kind == 'done':
                    self._set_busy(False)
                    self.progress['value'] = 100
                    self.status_var.set(f'Done → {msg[1]}')
                    if self.smoke:
                        print(f'SMOKE OK -> {msg[1]}', flush=True)
                        self.after(400, self.destroy)
                        continue
                    if self.open_var.get() and not self.smoke:
                        try:
                            os.startfile(os.path.abspath(msg[1]))  # noqa: S606
                        except OSError:
                            pass
                elif kind == 'error':
                    self._set_busy(False)
                    self.status_var.set(f'ERROR: {msg[1][:200]}')
                    if self.smoke:
                        print(f'SMOKE FAIL: {msg[1]}', flush=True)
                        self.after(400, self.destroy)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == '__main__':
    import sys
    smoke = '--smoke' in sys.argv
    audio = next((a for a in sys.argv[1:] if not a.startswith('--')), None)
    MixerApp(audio=audio, smoke=smoke).mainloop()
