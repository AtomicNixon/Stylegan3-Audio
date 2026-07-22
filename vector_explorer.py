import os
import glob
import pickle
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch

VECTORS_DIR = 'vectors'
PKL_DIR = '.'
IMAGE_SIZE = 1024
VECTOR_SCALE = 20.0


def find_pkls():
    return sorted(glob.glob(os.path.join(PKL_DIR, '*.pkl')))


def find_vectors():
    paths = sorted(glob.glob(os.path.join(VECTORS_DIR, '*.npy')))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]


def load_and_normalize_vector(path, device):
    v = np.load(path).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-8:
        v = v / norm
    return torch.from_numpy(v).to(device)


def build_slider_panel(parent, vec_list, slider_vars, on_release):
    outer = tk.Frame(parent, bg='#1e1e1e')
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg='#1e1e1e', highlightthickness=0, width=260)
    scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky='nsew')
    scrollbar.grid(row=0, column=1, sticky='ns')

    inner = tk.Frame(canvas, bg='#1e1e1e')
    win_id = canvas.create_window((0, 0), window=inner, anchor='nw')

    def _on_frame_configure(e):
        canvas.configure(scrollregion=canvas.bbox('all'))
    inner.bind('<Configure>', _on_frame_configure)

    def _on_canvas_configure(e):
        canvas.itemconfig(win_id, width=e.width)
    canvas.bind('<Configure>', _on_canvas_configure)

    canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

    for name, path in vec_list:
        var = tk.DoubleVar(value=0.0)
        slider_vars[name] = (var, path)

        row = tk.Frame(inner, bg='#1e1e1e')
        row.pack(fill='x', padx=6, pady=2)

        tk.Label(row, text=name, bg='#1e1e1e', fg='#ccc',
                 width=20, anchor='w', font=('Consolas', 8)).pack(side='left')

        s = tk.Scale(row, variable=var, from_=0.0, to=1.0, resolution=0.01,
                     orient='horizontal', bg='#1e1e1e', fg='white',
                     highlightthickness=0, troughcolor='#444', activebackground='#888',
                     length=110, showvalue=False)
        s.pack(side='left', padx=(2, 0))
        s.bind('<ButtonRelease-1>', on_release)

        tk.Label(row, textvariable=var, bg='#1e1e1e', fg='#7cf',
                 width=5, anchor='e', font=('Consolas', 8)).pack(side='left', padx=(8, 0))

    return outer


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('StyleGAN3 Vector Explorer')
        self.resizable(True, True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = None
        self.w_avg = None
        self.vectors = {}
        self.current_pkl = tk.StringVar()
        self.seed_var = tk.IntVar(value=0)
        self.psi_var = tk.DoubleVar(value=0.7)
        self.slider_vars = {}
        self.status_var = tk.StringVar(value='Select a PKL to begin.')

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=1)

        vec_list = find_vectors()
        mid = len(vec_list) // 2
        left_vecs = vec_list[:mid]
        right_vecs = vec_list[mid:]

        on_release = lambda e: self._generate()

        # --- Left vector panel ---
        left_panel = build_slider_panel(self, left_vecs, self.slider_vars, on_release)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(6, 0), pady=6)

        # --- Center panel ---
        center = tk.Frame(self, bg='#111')
        center.grid(row=0, column=1, sticky='nsew')
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        controls = tk.Frame(center, bg='#1e1e1e', pady=6)
        controls.grid(row=0, column=0, sticky='ew', padx=8)

        # PKL selector
        tk.Label(controls, text='Network:', bg='#1e1e1e', fg='white').pack(side='left', padx=(0, 4))
        pkls = find_pkls()
        self._pkls = pkls
        self.pkl_combo = ttk.Combobox(controls, textvariable=self.current_pkl,
                                       values=[os.path.basename(p) for p in pkls],
                                       state='readonly', width=28)
        self.pkl_combo.pack(side='left', padx=(0, 16))
        if pkls:
            self.pkl_combo.current(0)
        self.pkl_combo.bind('<<ComboboxSelected>>', lambda e: self._on_pkl_change())

        # Seed
        tk.Label(controls, text='Seed:', bg='#1e1e1e', fg='white').pack(side='left', padx=(0, 4))
        seed_entry = tk.Spinbox(controls, from_=0, to=99999, textvariable=self.seed_var, width=6)
        seed_entry.pack(side='left', padx=(0, 16))
        seed_entry.bind('<Return>', lambda e: self._generate())
        seed_entry.bind('<FocusOut>', lambda e: self._generate())

        # PSI
        tk.Label(controls, text='PSI:', bg='#1e1e1e', fg='white').pack(side='left', padx=(0, 4))
        psi_slider = tk.Scale(controls, variable=self.psi_var, from_=0.0, to=1.0,
                               resolution=0.01, orient='horizontal', bg='#1e1e1e', fg='white',
                               highlightthickness=0, troughcolor='#444', activebackground='#888',
                               length=140, showvalue=False)
        psi_slider.pack(side='left')
        psi_slider.bind('<ButtonRelease-1>', lambda e: self._generate())
        self.psi_label = tk.Label(controls, textvariable=self.psi_var, bg='#1e1e1e', fg='#7cf',
                                   width=4, font=('Consolas', 9))
        self.psi_label.pack(side='left', padx=(8, 16))

        # Status
        tk.Label(controls, textvariable=self.status_var, bg='#1e1e1e', fg='#aaa',
                 font=('Consolas', 8)).pack(side='left')

        # Image
        self.image_label = tk.Label(center, bg='#111')
        self.image_label.grid(row=1, column=0, sticky='nsew')
        placeholder = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (30, 30, 30))
        self._show_image(placeholder)

        # --- Right vector panel ---
        right_panel = build_slider_panel(self, right_vecs, self.slider_vars, on_release)
        right_panel.grid(row=0, column=2, sticky='nsew', padx=(0, 6), pady=6)

    def _on_pkl_change(self):
        name = self.current_pkl.get()
        path = next((p for p in self._pkls if os.path.basename(p) == name), None)
        if not path:
            return
        self.status_var.set(f'Loading {name}...')
        self.update_idletasks()
        try:
            with open(path, 'rb') as f:
                self.G = pickle.load(f)['G_ema'].to(self.device)
            self.w_avg = self.G.mapping.w_avg
            self.vectors = {}
            self.status_var.set(f'Loaded {name}')
            self._generate()
        except Exception as ex:
            self.status_var.set(f'Error: {ex}')

    def _generate(self):
        if self.G is None:
            self.status_var.set('No network loaded.')
            return
        self.status_var.set('Generating...')
        self.update_idletasks()
        try:
            seed = self.seed_var.get()
            psi = self.psi_var.get()

            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            with torch.no_grad():
                w = self.G.mapping(z, None)
                w = self.w_avg + psi * (w - self.w_avg)

                for name, (var, path) in self.slider_vars.items():
                    val = var.get()
                    if val == 0.0:
                        continue
                    if name not in self.vectors:
                        self.vectors[name] = load_and_normalize_vector(path, self.device)
                    w = w + self.vectors[name] * val * VECTOR_SCALE

                img_tensor = self.G.synthesis(w, noise_mode='const')

            img = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            if img.width != IMAGE_SIZE or img.height != IMAGE_SIZE:
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            self._show_image(img)
            self.status_var.set(f'seed={seed}  psi={psi:.2f}')
        except Exception as ex:
            self.status_var.set(f'Error: {ex}')

    def _show_image(self, img: Image.Image):
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo


if __name__ == '__main__':
    app = App()
    app.geometry('1800x1100')
    app.mainloop()
