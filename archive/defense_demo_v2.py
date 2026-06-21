"""
Defense Demo — Enhanced Federated Learning Cycle  (v2)
=======================================================
Two-tab Tkinter application for thesis defense presentation.

Tab 1 — FL Simulation Replay
    Step through 10 rounds of the actual training run (May 5, 2026).
    Animated step cards light up sequentially each round.
    Round 6 triggers a gold "Best Checkpoint" banner with all three metrics.
    Live chart shown as a compact side panel.

Tab 2 — Deepfake Detector
    Runs the trained TFLite model on a video file.
    This is the main/dominant panel — full-width video canvas + stats.

Model resolution order:
  1. Same directory as this script  (effnet_global_fl_final_quantised.tflite)
  2. Browse dialog if not found

Usage:
    python defense_demo_v2.py
    python defense_demo_v2.py --model path/to/model.tflite
"""

import argparse
import collections
import csv
import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from PIL import Image, ImageTk

# Fix for Pylance / TF dynamic import
try:
    from keras.applications.efficientnet import preprocess_input
except ImportError:
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# ─────────────────────────────────────────────────────────────
# TFLite runtime (same fallback chain as original prototype)
# ─────────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

# ══════════════════════════════════════════════════════════════
# REAL TRAINING DATA  (May 5 2026 run — used verbatim in paper)
# ══════════════════════════════════════════════════════════════
ROUND_DATA = [
    # round, acc,    f1,     auc,    precision, recall, selected_clients
    (0,  0.9435, 0.9435, 0.9901, None,   None,   []),
    (1,  0.9478, 0.9478, 0.9924, 0.9504, 0.9496, ["006","004","005"]),
    (2,  0.9543, 0.9543, 0.9963, 0.9558, 0.9558, ["000","003","008"]),
    (3,  0.9543, 0.9543, 0.9963, 0.9558, 0.9558, ["002","007","009"]),
    (4,  0.9630, 0.9630, 0.9964, 0.9639, 0.9643, ["006","004","005"]),
    (5,  0.9630, 0.9630, 0.9964, 0.9639, 0.9643, ["000","003","008"]),
    (6,  0.9652, 0.9652, 0.9964, 0.9656, 0.9662, ["002","007","009"]),  # PEAK
    (7,  0.9620, 0.9620, 0.9970, 0.9635, 0.9635, ["006","004","005"]),
    (8,  0.9587, 0.9587, 0.9981, 0.9602, 0.9602, ["000","003","008"]),
    (9,  0.9530, 0.9530, 0.9975, 0.9545, 0.9545, ["002","007","009"]),
    (10, 0.9478, 0.9478, 0.9962, 0.9507, 0.9497, ["006","004","005"]),
]

CLIENT_REP = {
    "000": 0.465, "001": 0.452, "002": 0.469,
    "003": 0.465, "004": 0.467, "005": 0.470,
    "006": 0.467, "007": 0.465, "008": 0.465,
    "009": 0.465,
}
ALL_CLIENTS = ["000","001","002","003","004","005","006","007","008","009"]

# ── Colours ──────────────────────────────────────────────────
HEX_BG       = "#1A1A2E"   # deep navy
HEX_PANEL    = "#16213E"   # slightly lighter navy
HEX_CARD     = "#0F3460"   # card background
HEX_TEXT     = "#E0E0E0"
HEX_ACCENT   = "#3A6EA5"
HEX_GREEN    = "#44CC66"
HEX_SELECTED = "#00B4D8"   # cyan-blue for selected clients
HEX_IDLE     = "#2A2A4A"
HEX_BEST     = "#F0C040"   # gold
HEX_REAL     = "#44FF44"
HEX_FAKE     = "#FF4444"
HEX_UNC      = "#FFD700"
HEX_NEUTRAL  = "#AAAAAA"

# Step card colours
STEP_COLORS = {
    "select":    ("#F0C040", "#3A3000"),   # gold
    "train":     ("#4EA8F0", "#001E3A"),   # blue
    "validate":  ("#A78BFA", "#1E0A3A"),   # purple
    "aggregate": ("#2ECC71", "#003A1E"),   # green
    "update":    ("#F07850", "#3A1800"),   # orange
}
STEP_IDLE_FG  = "#555577"
STEP_IDLE_BG  = "#1E1E38"

COLOR_REAL      = (136, 255,  68)
COLOR_FAKE      = ( 68,  68, 255)
COLOR_UNCERTAIN = (  0, 255, 255)
COLOR_NO_FACE   = (  0, 200, 255)

INPUT_SIZE     = 260
FACE_PADDING   = 0.20
HISTORY_SIZE   = 8
REAL_THRESHOLD = 0.67
FAKE_THRESHOLD = 0.45
FRAME_SKIP     = 4
DISPLAY_MAX_W  = 880
DISPLAY_MAX_H  = 500
DETECT_W       = 640
DETECT_H       = 360
SPEED_OPTIONS  = [0.25, 0.5, 1.0, 2.0, 4.0]
VIDEO_EXTS     = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v"),
    ("All files",   "*.*"),
)


# ══════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════
def load_tflite_model(model_path: str):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def resolve_model_path(given):
    default_name = "effnet_global_fl_final_quantised.tflite"
    script_dir   = Path(__file__).parent

    if given:
        p = Path(given)
        if p.exists():
            return str(p)
        print(f"[WARN] Specified model not found: {given}")

    auto = script_dir / default_name
    if auto.exists():
        return str(auto)

    print("[INFO] Model not found automatically — opening browse dialog.")
    root_tmp = tk.Tk()
    root_tmp.withdraw()
    path = filedialog.askopenfilename(
        title="Locate TFLite model",
        filetypes=[("TFLite model", "*.tflite"), ("All files", "*.*")],
        initialdir=str(script_dir),
    )
    root_tmp.destroy()
    if not path:
        print("[ERROR] No model selected. Exiting.")
        sys.exit(1)
    return path


# ══════════════════════════════════════════════════════════════
# Preprocessing + classification (identical to prototype)
# ══════════════════════════════════════════════════════════════
def preprocess_face_tflite(face_bgr, input_dtype, input_scale, input_zero_point):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = tf.cast(face_rgb, tf.float32)
    img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    if input_dtype == np.uint8:
        img = preprocess_input(img)
        if input_scale:
            img = img / input_scale + input_zero_point
        img = tf.clip_by_value(img, 0, 255)
        return np.expand_dims(img.numpy().astype(np.uint8), axis=0)
    else:
        img = preprocess_input(img)
        return np.expand_dims(img.numpy().astype(np.float32), axis=0)


def classify(smoothed: float):
    if smoothed >= REAL_THRESHOLD:
        return "REAL",      COLOR_REAL,      HEX_REAL,  smoothed * 100.0
    elif smoothed < FAKE_THRESHOLD:
        return "FAKE",      COLOR_FAKE,      HEX_FAKE,  (1.0 - smoothed) * 100.0
    else:
        conf = (abs(smoothed - 0.575) / 0.375) * 100.0
        return "UNCERTAIN", COLOR_UNCERTAIN, HEX_UNC,   min(conf, 100.0)


# ══════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════
class DefenseDemo:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root       = root
        self.model_path = model_path

        self.root.title("Enhanced FL Cycle — Defense Demo")
        self.root.configure(bg=HEX_BG)
        self.root.resizable(True, True)
        self.root.minsize(1280, 720)

        # Load model once, shared across both tabs
        self.interpreter, self.inp, self.out = load_tflite_model(model_path)
        self.inp_dtype = self.inp["dtype"]
        self.inp_scale, self.inp_zp = self.inp.get("quantization", (1.0, 0))
        self.out_scale, self.out_zp = self.out.get("quantization", (1.0, 0))

        # MediaPipe face detector
        mp_fd = mp.solutions.face_detection
        self.face_detector = mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    # Top-level UI
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header bar
        header = tk.Frame(self.root, bg="#0D0D1A", pady=10)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="An Enhanced Federated Cycle for DeepFake Detection  ·  Defense Demo",
            bg="#0D0D1A", fg=HEX_TEXT, font=("Helvetica", 13, "bold")
        ).pack()

        # Notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",       background=HEX_BG,    borderwidth=0)
        style.configure("TNotebook.Tab",   background=HEX_PANEL, foreground=HEX_TEXT,
                        padding=[22, 7],   font=("Helvetica", 11, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", HEX_ACCENT)],
                  foreground=[("selected", "white")])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_detection_tab()   # Tab 1 — dominant component
        self._build_fl_tab()          # Tab 2 — FL simulation

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Deepfake Detector (main component)
    # ══════════════════════════════════════════════════════════
    def _build_detection_tab(self):
        frame = tk.Frame(self.nb, bg=HEX_BG)
        self.nb.add(frame, text="  Deepfake Detector  ")

        # ── Toolbar ──────────────────────────────────────────
        toolbar = tk.Frame(frame, bg=HEX_PANEL, pady=7)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Button(
            toolbar, text="📂  Open Video",
            command=self._det_open_file,
            bg=HEX_ACCENT, fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, padx=14, pady=5
        ).pack(side=tk.LEFT, padx=(12, 6))

        tk.Button(
            toolbar, text="💾  Export CSV",
            command=self._det_export_csv,
            bg="#2A2A4A", fg=HEX_TEXT,
            font=("Helvetica", 10),
            relief=tk.FLAT, padx=10, pady=5
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            toolbar, text="🖼  Save Frame",
            command=self._det_save_frame,
            bg="#2A2A4A", fg=HEX_TEXT,
            font=("Helvetica", 10),
            relief=tk.FLAT, padx=10, pady=5
        ).pack(side=tk.LEFT, padx=4)

        # Model badge on right
        model_name = Path(self.model_path).name
        model_badge = tk.Frame(toolbar, bg="#0F3460", padx=10, pady=4)
        model_badge.pack(side=tk.RIGHT, padx=12)
        tk.Label(model_badge, text="Model:", bg="#0F3460", fg="#888888",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        tk.Label(model_badge, text=model_name, bg="#0F3460", fg=HEX_GREEN,
                 font=("Helvetica", 8, "bold")).pack(side=tk.LEFT, padx=(4,0))

        # ── Main area: video canvas + stats panel ─────────────
        main_area = tk.Frame(frame, bg=HEX_BG)
        main_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 0))

        # Video canvas — dominant
        video_frame = tk.Frame(main_area, bg="black")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.det_canvas = tk.Label(video_frame, bg="#0A0A0A",
                                    width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.det_canvas.pack(fill=tk.BOTH, expand=True)

        # Stats panel — right side, fixed width
        stats_outer = tk.Frame(main_area, bg=HEX_PANEL, width=220)
        stats_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        stats_outer.pack_propagate(False)

        # Per-frame verdict header
        tk.Label(stats_outer, text="CURRENT FRAME",
                 bg=HEX_PANEL, fg="#666688",
                 font=("Helvetica", 8, "bold")).pack(pady=(14, 2))

        self.det_lbl_verdict = tk.Label(
            stats_outer, text="—",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 32, "bold")
        )
        self.det_lbl_verdict.pack(pady=(2, 0))

        self.det_lbl_confidence = tk.Label(
            stats_outer, text="Confidence: —",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 10)
        )
        self.det_lbl_confidence.pack()

        self.det_conf_bar = ttk.Progressbar(
            stats_outer, orient="horizontal", length=180, mode="determinate"
        )
        self.det_conf_bar.pack(pady=(6, 4))

        self._sep(stats_outer)

        # Per-frame stats
        tk.Label(stats_outer, text="INFERENCE STATS",
                 bg=HEX_PANEL, fg="#666688",
                 font=("Helvetica", 8, "bold")).pack(pady=(4, 2))

        self.det_stat_vars = {}
        for key, label in [("frame","Frame"), ("fps","Display FPS"),
                            ("inf_ms","Inference ms"), ("raw","Raw prob"),
                            ("smooth","Smoothed prob")]:
            row = tk.Frame(stats_outer, bg=HEX_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=label+":", bg=HEX_PANEL, fg="#888888",
                     font=("Helvetica", 9), anchor="w", width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self.det_stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=HEX_PANEL, fg=HEX_TEXT,
                     font=("Helvetica", 9, "bold"), anchor="e").pack(side=tk.RIGHT)

        self._sep(stats_outer)

        # Frame tally
        tk.Label(stats_outer, text="FRAME TALLY",
                 bg=HEX_PANEL, fg="#666688",
                 font=("Helvetica", 8, "bold")).pack(pady=(4, 2))

        self.det_tally_vars = {}
        for verdict, hex_c in [("REAL", HEX_REAL), ("FAKE", HEX_FAKE),
                                 ("UNCERTAIN", HEX_UNC)]:
            row = tk.Frame(stats_outer, bg=HEX_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=verdict+":", bg=HEX_PANEL, fg=hex_c,
                     font=("Helvetica", 9, "bold"), anchor="w",
                     width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value="0  (0.0%)")
            self.det_tally_vars[verdict] = var
            tk.Label(row, textvariable=var, bg=HEX_PANEL, fg=HEX_TEXT,
                     font=("Helvetica", 9), anchor="e").pack(side=tk.RIGHT)

        self._sep(stats_outer)

        # Overall verdict — prominent
        tk.Label(stats_outer, text="OVERALL VERDICT",
                 bg=HEX_PANEL, fg="#666688",
                 font=("Helvetica", 8, "bold")).pack(pady=(4, 2))
        self.det_lbl_overall = tk.Label(
            stats_outer, text="—",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 22, "bold")
        )
        self.det_lbl_overall.pack(pady=(2, 0))
        self.det_lbl_overall_sub = tk.Label(
            stats_outer, text="",
            bg=HEX_PANEL, fg="#888888",
            font=("Helvetica", 9), wraplength=200, justify="center"
        )
        self.det_lbl_overall_sub.pack(padx=8, pady=(0, 8))

        # ── Progress bar ──────────────────────────────────────
        prog_frame = tk.Frame(frame, bg=HEX_BG, pady=4)
        prog_frame.pack(fill=tk.X, padx=10)

        self.det_progress_var = tk.DoubleVar(value=0.0)
        self.det_progress_bar = ttk.Scale(
            prog_frame, from_=0, to=100,
            orient="horizontal", variable=self.det_progress_var,
            command=self._det_on_seek
        )
        self.det_progress_bar.pack(fill=tk.X)

        self.det_lbl_time = tk.Label(
            prog_frame, text="0:00 / 0:00",
            bg=HEX_BG, fg="#888888", font=("Helvetica", 9)
        )
        self.det_lbl_time.pack(anchor="e")

        # ── Controls ──────────────────────────────────────────
        ctrl_frame = tk.Frame(frame, bg=HEX_PANEL, pady=8)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_cfg = dict(bg="#2A2A4A", fg=HEX_TEXT,
                       font=("Helvetica", 12), relief=tk.FLAT,
                       width=4, pady=4)

        self.det_btn_play = tk.Button(ctrl_frame, text="▶",
                                       command=self._det_toggle_play, **btn_cfg)
        self.det_btn_play.pack(side=tk.LEFT, padx=(14, 2))

        tk.Button(ctrl_frame, text="⏹", command=self._det_stop,
                  **btn_cfg).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="⟳", command=self._det_restart,
                  **btn_cfg).pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl_frame, text="Speed:", bg=HEX_PANEL,
                 fg=HEX_TEXT, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(16, 4))

        self.det_speed_var = tk.StringVar(value="1.0×")
        speed_menu = ttk.Combobox(
            ctrl_frame, textvariable=self.det_speed_var,
            values=[f"{s}×" for s in SPEED_OPTIONS],
            width=6, state="readonly"
        )
        speed_menu.pack(side=tk.LEFT, padx=4)
        speed_menu.bind("<<ComboboxSelected>>", self._det_on_speed_change)

        self.det_lbl_status = tk.Label(
            ctrl_frame, text="Open a video file to begin.  [Space = play/pause  ·  Esc = stop  ·  R = restart]",
            bg=HEX_PANEL, fg="#888888", font=("Helvetica", 9)
        )
        self.det_lbl_status.pack(side=tk.LEFT, padx=16)

        # Keyboard shortcuts
        self.root.bind("<space>",  lambda e: self._det_toggle_play()
                       if self.nb.index(self.nb.select()) == 0 else None)
        self.root.bind("<Escape>", lambda e: self._det_stop())
        self.root.bind("<r>",      lambda e: self._det_restart())
        self.root.bind("<s>",      lambda e: self._det_save_frame())

        # ── Playback state ────────────────────────────────────
        self.det_cap           = None
        self.det_video_path    = None
        self.det_total_frames  = 0
        self.det_video_fps     = 30.0
        self.det_frame_idx     = 0
        self.det_playing       = False
        self.det_speed         = 1.0
        self.det_seek_pending  = None
        self.det_after_id      = None
        self.det_loop_active   = True
        self.det_history       = collections.deque(maxlen=HISTORY_SIZE)
        self.det_raw_prob      = 0.5
        self.det_smoothed      = 0.5
        self.det_label         = "NO FACE"
        self.det_cv_color      = COLOR_NO_FACE
        self.det_hex_color     = HEX_NEUTRAL
        self.det_confidence    = 0.0
        self.det_inf_ms        = 0.0
        self.det_bbox          = None
        self.det_verdict_counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
        self.det_frame_log     = []
        self.det_last_det      = None
        self.det_fps_display   = 0.0
        self.det_t_prev        = time.time()
        self.det_vid_w         = DISPLAY_MAX_W
        self.det_vid_h         = DISPLAY_MAX_H
        self.det_display_w     = DISPLAY_MAX_W
        self.det_display_h     = DISPLAY_MAX_H

        self.det_infer_lock       = threading.Lock()
        self.det_infer_queue      = []
        self.det_infer_queue_lock = threading.Lock()
        self.det_stop_event       = threading.Event()

        self._det_show_placeholder()
        self._det_start_inference_thread()
        self.det_after_id = self.root.after(33, self._det_update_loop)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — FL Simulation Replay
    # ══════════════════════════════════════════════════════════
    def _build_fl_tab(self):
        frame = tk.Frame(self.nb, bg=HEX_BG)
        self.nb.add(frame, text="  FL Simulation  ")

        self.fl_round_index = 0
        self.fl_animating   = False

        # ── Layout: left panel (controls + steps + clients) | right (chart) ──
        left = tk.Frame(frame, bg=HEX_BG, width=500)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 6), pady=10)
        left.pack_propagate(False)

        right = tk.Frame(frame, bg=HEX_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=10)

        # ── Round header ─────────────────────────────────────
        header_frame = tk.Frame(left, bg=HEX_CARD, pady=10, padx=14)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_round = tk.Label(
            header_frame, text="Pre-Training Baseline",
            bg=HEX_CARD, fg=HEX_TEXT, font=("Helvetica", 15, "bold"),
            anchor="w"
        )
        self.lbl_round.pack(fill=tk.X)

        self.lbl_phase = tk.Label(
            header_frame,
            text="Global model initialised — no FL rounds applied yet.",
            bg=HEX_CARD, fg="#9090AA", font=("Helvetica", 9),
            anchor="w", wraplength=460, justify="left"
        )
        self.lbl_phase.pack(fill=tk.X, pady=(4, 0))

        # ── Metrics row ───────────────────────────────────────
        metrics_frame = tk.Frame(left, bg=HEX_CARD)
        metrics_frame.pack(fill=tk.X, pady=(0, 8))

        self.metric_vars = {}
        self.metric_frames = {}
        for col, (key, label) in enumerate([
            ("acc",  "Accuracy"),
            ("f1",   "F1 Score"),
            ("auc",  "ROC-AUC"),
        ]):
            cell = tk.Frame(metrics_frame, bg=HEX_CARD, padx=10, pady=10)
            cell.grid(row=0, column=col, sticky="nsew")
            metrics_frame.columnconfigure(col, weight=1)
            tk.Label(cell, text=label, bg=HEX_CARD, fg="#9090AA",
                     font=("Helvetica", 8, "bold")).pack()
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            lbl = tk.Label(cell, textvariable=var, bg=HEX_CARD, fg=HEX_GREEN,
                     font=("Helvetica", 20, "bold"))
            lbl.pack()
            self.metric_frames[key] = (cell, lbl)

        # ── Round 6 Best Checkpoint Banner (hidden until R6) ──
        self.best_banner = tk.Frame(left, bg=HEX_BEST, pady=0)
        # not packed yet — appears at Round 6

        self.lbl_best_title = tk.Label(
            self.best_banner,
            text="★  BEST CHECKPOINT  —  ROUND 6  ★",
            bg=HEX_BEST, fg="#1A1000",
            font=("Helvetica", 12, "bold")
        )
        self.lbl_best_title.pack(pady=(8, 2))

        self.best_metric_frame = tk.Frame(self.best_banner, bg=HEX_BEST)
        self.best_metric_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        for col, (key, label, val) in enumerate([
            ("bacc",  "Accuracy",  "96.52%"),
            ("bf1",   "F1 Score",  "96.52%"),
            ("bauc",  "ROC-AUC",  "0.9964"),
        ]):
            cell = tk.Frame(self.best_metric_frame, bg="#D4A800", padx=12, pady=6)
            cell.grid(row=0, column=col, padx=4, sticky="nsew")
            self.best_metric_frame.columnconfigure(col, weight=1)
            tk.Label(cell, text=label, bg="#D4A800", fg="#3A2A00",
                     font=("Helvetica", 8, "bold")).pack()
            tk.Label(cell, text=val, bg="#D4A800", fg="#1A0A00",
                     font=("Helvetica", 16, "bold")).pack()

        # ── Step Cards ────────────────────────────────────────
        tk.Label(left, text="ROUND STEPS", bg=HEX_BG, fg="#555577",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(4, 2))

        steps_def = [
            ("select",    "1. Client Selection",
             "Top-K clients selected by score = V_i × H_i × R_i\n"
             "with 2-round cooldown preventing repeat selection."),
            ("train",     "2. Local Training",
             "Each selected client trains for 5 local epochs\n"
             "on its private partition (920 images, batch 32)."),
            ("validate",  "3. Update Validation",
             "Server checks each update: L2 norm bound + gain test.\n"
             "Updates failing either check are rejected."),
            ("aggregate", "4. Weighted Aggregation",
             "Accepted updates are merged with adaptive weights\n"
             "proportional to each client's contribution score."),
            ("update",    "5. Reputation Update",
             "Ledger updated for all 10 clients: +reward for valid,\n"
             "+penalty for invalid, ×0.99 decay applied each round."),
        ]

        self.step_cards   = {}
        self.step_title_lbl = {}
        self.step_desc_lbl  = {}

        steps_container = tk.Frame(left, bg=HEX_BG)
        steps_container.pack(fill=tk.X, pady=(0, 6))

        for key, title, desc in steps_def:
            card = tk.Frame(steps_container, bg=STEP_IDLE_BG,
                            relief=tk.FLAT, pady=6, padx=10)
            card.pack(fill=tk.X, pady=2)

            title_lbl = tk.Label(card, text=title, bg=STEP_IDLE_BG,
                                  fg=STEP_IDLE_FG,
                                  font=("Helvetica", 10, "bold"), anchor="w")
            title_lbl.pack(fill=tk.X)

            desc_lbl = tk.Label(card, text=desc, bg=STEP_IDLE_BG,
                                 fg=STEP_IDLE_FG,
                                 font=("Helvetica", 8), anchor="w",
                                 justify="left", wraplength=440)
            desc_lbl.pack(fill=tk.X)

            self.step_cards[key]    = card
            self.step_title_lbl[key] = title_lbl
            self.step_desc_lbl[key]  = desc_lbl

        # ── Client grid ───────────────────────────────────────
        tk.Label(left, text="CLIENT POOL", bg=HEX_BG, fg="#555577",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(4, 2))

        client_grid = tk.Frame(left, bg=HEX_BG)
        client_grid.pack(fill=tk.X, pady=(0, 6))

        self.client_frames   = {}
        self.client_name_lbl = {}
        self.client_rep_vars = {}
        for i, cid in enumerate(ALL_CLIENTS):
            row_i, col_i = divmod(i, 5)
            cell = tk.Frame(client_grid, bg=HEX_IDLE, relief=tk.FLAT,
                            padx=6, pady=6)
            cell.grid(row=row_i, column=col_i, padx=3, pady=3, sticky="nsew")
            client_grid.columnconfigure(col_i, weight=1)

            name_lbl = tk.Label(cell, text=f"C{cid}", bg=HEX_IDLE, fg=HEX_TEXT,
                                 font=("Helvetica", 9, "bold"))
            name_lbl.pack()
            rep_var = tk.StringVar(value=f"R:{CLIENT_REP[cid]:.3f}")
            self.client_rep_vars[cid] = rep_var
            rep_lbl = tk.Label(cell, textvariable=rep_var, bg=HEX_IDLE,
                                fg="#888888", font=("Helvetica", 7))
            rep_lbl.pack()
            self.client_frames[cid]   = (cell, name_lbl, rep_lbl)

        # ── Controls ──────────────────────────────────────────
        ctrl = tk.Frame(left, bg=HEX_BG)
        ctrl.pack(fill=tk.X, pady=(6, 0))

        self.btn_next = tk.Button(
            ctrl, text="▶  Next Round",
            command=self._fl_next_round,
            bg=HEX_ACCENT, fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, padx=18, pady=7
        )
        self.btn_next.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            ctrl, text="⟳  Reset",
            command=self._fl_reset,
            bg="#2A2A4A", fg=HEX_TEXT,
            font=("Helvetica", 10),
            relief=tk.FLAT, padx=12, pady=7
        ).pack(side=tk.LEFT)

        self.lbl_fl_status = tk.Label(
            left, text="Click 'Next Round' to begin the simulation.",
            bg=HEX_BG, fg="#666688", font=("Helvetica", 9),
            wraplength=460, justify="left"
        )
        self.lbl_fl_status.pack(fill=tk.X, pady=(6, 0))

        # ── Chart (right side) ────────────────────────────────
        self._build_fl_chart(right)

        # Draw initial state
        self._fl_update_display(0)

    def _build_fl_chart(self, parent):
        fig_bg = "#1A1A2E"
        ax_bg  = "#0F1428"

        self.fl_fig, self.fl_axes = plt.subplots(
            2, 1, figsize=(5.2, 7.0),
            facecolor=fig_bg, gridspec_kw={"hspace": 0.52}
        )
        self.fl_fig.patch.set_facecolor(fig_bg)

        for ax in self.fl_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors="#888888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        ax_acc, ax_auc = self.fl_axes

        ax_acc.set_title("Accuracy & F1 Score per Round",
                         color=HEX_TEXT, fontsize=9, pad=6,
                         fontweight="bold")
        ax_acc.set_xlabel("Round", color="#888888", fontsize=8)
        ax_acc.set_ylabel("Score", color="#888888", fontsize=8)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(0.92, 0.975)
        ax_acc.set_xticks(range(0, 11))
        ax_acc.set_xticklabels(
            ["Pre"] + [str(r) for r in range(1, 11)],
            color="#888888", fontsize=7
        )
        ax_acc.axhline(0.942, color="#666666", linewidth=1.0,
                       linestyle="--", label="FL-TENB4 (94.2%)")
        ax_acc.legend(fontsize=7, facecolor="#1E1E38",
                      labelcolor="#888888", loc="lower right",
                      framealpha=0.8)

        ax_auc.set_title("ROC-AUC per Round",
                         color=HEX_TEXT, fontsize=9, pad=6,
                         fontweight="bold")
        ax_auc.set_xlabel("Round", color="#888888", fontsize=8)
        ax_auc.set_ylabel("AUC", color="#888888", fontsize=8)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(0.985, 1.001)
        ax_auc.set_xticks(range(0, 11))
        ax_auc.set_xticklabels(
            ["Pre"] + [str(r) for r in range(1, 11)],
            color="#888888", fontsize=7
        )
        ax_auc.axhline(0.96, color="#666666", linewidth=1.0,
                       linestyle="--", label="FL-TENB4 (0.96)")
        ax_auc.legend(fontsize=7, facecolor="#1E1E38",
                      labelcolor="#888888", loc="lower right",
                      framealpha=0.8)

        self.line_acc, = ax_acc.plot([], [], color="#4EA8F0",
                                      linewidth=2, marker="o",
                                      markersize=5, label="Accuracy",
                                      zorder=3)
        self.line_f1,  = ax_acc.plot([], [], color="#F07850",
                                      linewidth=1.5, linestyle="--",
                                      marker="s", markersize=4, label="F1",
                                      zorder=3)
        ax_acc.legend(fontsize=7, facecolor="#1E1E38",
                      labelcolor="#888888", loc="lower right",
                      framealpha=0.8)

        self.line_auc, = ax_auc.plot([], [], color=HEX_GREEN,
                                      linewidth=2, marker="o",
                                      markersize=5, zorder=3)

        # Round 6 vertical markers (hidden until triggered)
        self.vline_acc = ax_acc.axvline(x=6, color=HEX_BEST,
                                         linewidth=2.0, linestyle=":",
                                         alpha=0.0, zorder=2)
        self.vline_auc = ax_auc.axvline(x=6, color=HEX_BEST,
                                         linewidth=2.0, linestyle=":",
                                         alpha=0.0, zorder=2)

        # Round 6 star annotation (hidden until triggered)
        self.ann_r6_acc = ax_acc.annotate(
            "★ Best\nR6: 96.52%",
            xy=(6, 0.9652), xytext=(7.2, 0.960),
            color=HEX_BEST, fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=HEX_BEST, lw=1.2),
            alpha=0.0
        )
        self.ann_r6_auc = ax_auc.annotate(
            "★ R6: 0.9964",
            xy=(6, 0.9964), xytext=(7.2, 0.9950),
            color=HEX_BEST, fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=HEX_BEST, lw=1.2),
            alpha=0.0
        )

        self.fl_canvas = FigureCanvasTkAgg(self.fl_fig, master=parent)
        self.fl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fl_canvas.draw()

    # ── FL simulation logic ───────────────────────────────────
    def _fl_next_round(self):
        if self.fl_animating:
            return
        next_idx = self.fl_round_index + 1
        if next_idx >= len(ROUND_DATA):
            self.lbl_fl_status.config(
                text="All 10 rounds complete. Best-weight tracking deployed Round 6 model. Click Reset to replay."
            )
            self.btn_next.config(state=tk.DISABLED)
            return
        self.fl_animating = True
        self.btn_next.config(state=tk.DISABLED)
        threading.Thread(
            target=self._fl_animate_round,
            args=(next_idx,),
            daemon=True
        ).start()

    def _fl_animate_round(self, idx):
        rd       = ROUND_DATA[idx]
        selected = rd[6]

        steps_seq = [
            ("select",
             f"Selecting clients: {', '.join('C'+c for c in selected)}\n"
             f"Score = V_i × H_i × R_i  ·  cooldown enforced",
             0.95),
            ("train",
             f"C{selected[0]}, C{selected[1]}, C{selected[2]} training locally…\n"
             "5 epochs each · batch size 32 · private data only",
             0.90),
            ("validate",
             "Server validates each update  →  L2 norm check  +  gain test\n"
             "Result this round: 3 / 3 updates accepted ✓",
             0.85),
            ("aggregate",
             "Weighted FedAvg  ·  weights ∝ contribution scores\n"
             "Global model updated with aggregated weights",
             0.85),
            ("update",
             "Reputation ledger updated  ·  decay × 0.99 applied\n"
             "Valid clients +reward  ·  invalid clients +penalty",
             0.65),
        ]

        def reset_steps():
            for key in self.step_cards:
                self._fl_step_idle(key)

        def activate(step_key, msg, delay):
            self.root.after(0, lambda sk=step_key, m=msg: self._fl_set_step(sk, m))
            time.sleep(delay)
            # Mark step as done (dimmed active)
            self.root.after(0, lambda sk=step_key: self._fl_step_done(sk))
            time.sleep(0.1)

        self.root.after(0, reset_steps)
        time.sleep(0.1)
        self.root.after(0, lambda: self._fl_highlight_clients(selected))
        time.sleep(0.2)

        for key, msg, delay in steps_seq:
            activate(key, msg, delay)

        # Update metrics & chart
        self.root.after(0, lambda: self._fl_update_display(idx))
        self.root.after(0, lambda: self._fl_update_chart(idx))
        self.root.after(0, lambda: self._fl_finish_round(idx))

    def _fl_step_idle(self, key):
        card      = self.step_cards[key]
        title_lbl = self.step_title_lbl[key]
        desc_lbl  = self.step_desc_lbl[key]
        card.config(bg=STEP_IDLE_BG)
        title_lbl.config(bg=STEP_IDLE_BG, fg=STEP_IDLE_FG)
        desc_lbl.config(bg=STEP_IDLE_BG,  fg=STEP_IDLE_FG)

    def _fl_set_step(self, active_key, msg):
        """Light up the active step card."""
        for key in self.step_cards:
            self._fl_step_idle(key)

        fg_bright, bg_dark = STEP_COLORS[active_key]
        card      = self.step_cards[active_key]
        title_lbl = self.step_title_lbl[active_key]
        desc_lbl  = self.step_desc_lbl[active_key]

        card.config(bg=bg_dark)
        title_lbl.config(bg=bg_dark, fg=fg_bright)
        desc_lbl.config(bg=bg_dark, fg=fg_bright, text=desc_lbl.cget("text"))

        self.lbl_fl_status.config(text=msg)

    def _fl_step_done(self, key):
        """Dim slightly to show completion, keep it different from idle."""
        fg_bright, bg_dark = STEP_COLORS[key]
        card      = self.step_cards[key]
        title_lbl = self.step_title_lbl[key]
        desc_lbl  = self.step_desc_lbl[key]
        # use a muted version of the active colour
        card.config(bg="#1C1C30")
        title_lbl.config(bg="#1C1C30", fg="#505060")
        desc_lbl.config(bg="#1C1C30",  fg="#404050")

    def _fl_highlight_clients(self, selected):
        for cid, (cell, name_lbl, rep_lbl) in self.client_frames.items():
            if cid in selected:
                cell.config(bg=HEX_SELECTED)
                name_lbl.config(bg=HEX_SELECTED, fg="#001A22")
                rep_lbl.config(bg=HEX_SELECTED,  fg="#002A33")
            else:
                cell.config(bg=HEX_IDLE)
                name_lbl.config(bg=HEX_IDLE, fg=HEX_TEXT)
                rep_lbl.config(bg=HEX_IDLE,  fg="#888888")

    def _fl_update_display(self, idx):
        rd  = ROUND_DATA[idx]
        rnd, acc, f1, auc = rd[0], rd[1], rd[2], rd[3]

        if rnd == 0:
            self.lbl_round.config(text="Pre-Training Baseline")
            self.lbl_phase.config(
                text="Global model initialised with pre-trained EfficientNetB4 weights. "
                     "No federated rounds applied yet."
            )
        else:
            self.lbl_round.config(text=f"Round {rnd}  /  10")
            clients_str = "   ·   ".join(f"C{c}" for c in rd[6])
            self.lbl_phase.config(
                text=f"Selected: {clients_str}     |     Updates accepted: 3 / 3"
            )

        # Update metric cards
        for key, val_str in [
            ("acc", f"{acc*100:.2f}%"),
            ("f1",  f"{f1*100:.2f}%"),
            ("auc", f"{auc:.4f}"),
        ]:
            self.metric_vars[key].set(val_str)
            cell, lbl = self.metric_frames[key]
            # gold colour at peak round 6, green otherwise
            color = HEX_BEST if rnd == 6 else HEX_GREEN
            lbl.config(fg=color)
            cell.config(bg=("#1A1500" if rnd == 6 else HEX_CARD))
            cell_parent = cell.master
            cell_parent.config(bg=("#1A1500" if rnd == 6 else HEX_CARD))

        # Show/hide best banner
        if rnd == 6:
            self.best_banner.pack(fill=tk.X, pady=(0, 8), before=self.step_cards["select"].master)
        elif rnd == 0:
            try:
                self.best_banner.pack_forget()
            except Exception:
                pass

    def _fl_update_chart(self, idx):
        xs   = [ROUND_DATA[i][0] for i in range(idx + 1)]
        accs = [ROUND_DATA[i][1] for i in range(idx + 1)]
        f1s  = [ROUND_DATA[i][2] for i in range(idx + 1)]
        aucs = [ROUND_DATA[i][3] for i in range(idx + 1)]

        self.line_acc.set_data(xs, accs)
        self.line_f1.set_data(xs, f1s)
        self.line_auc.set_data(xs, aucs)

        if ROUND_DATA[idx][0] >= 6:
            self.vline_acc.set_alpha(0.9)
            self.vline_auc.set_alpha(0.9)
            self.ann_r6_acc.set_alpha(1.0)
            self.ann_r6_auc.set_alpha(1.0)

        self.fl_canvas.draw_idle()

    def _fl_finish_round(self, idx):
        self.fl_round_index = idx
        self.fl_animating   = False
        rnd = ROUND_DATA[idx][0]

        # Reset all step cards to idle after round finishes
        for key in self.step_cards:
            self._fl_step_idle(key)

        if rnd == 6:
            self.lbl_fl_status.config(
                text="★  Round 6 — Peak performance reached! Best checkpoint saved. "
                     "Subsequent rounds will show convergence decline."
            )
        elif idx == len(ROUND_DATA) - 1:
            self.lbl_fl_status.config(
                text="All 10 rounds complete. Best-weight tracking preserved the Round 6 "
                     "checkpoint (96.52%) as the deployed model — not Round 10 (94.78%)."
            )
            self.btn_next.config(state=tk.DISABLED)
            return
        else:
            self.lbl_fl_status.config(
                text=f"Round {rnd} complete — accuracy {ROUND_DATA[idx][1]*100:.2f}%. "
                     f"Click 'Next Round' to continue."
            )
        self.btn_next.config(state=tk.NORMAL)

    def _fl_reset(self):
        self.fl_round_index = 0
        self.fl_animating   = False
        self.btn_next.config(state=tk.NORMAL)

        # Reset client colours
        for cid, (cell, name_lbl, rep_lbl) in self.client_frames.items():
            cell.config(bg=HEX_IDLE)
            name_lbl.config(bg=HEX_IDLE, fg=HEX_TEXT)
            rep_lbl.config(bg=HEX_IDLE,  fg="#888888")

        # Reset step cards
        for key in self.step_cards:
            self._fl_step_idle(key)

        # Reset chart
        self.line_acc.set_data([], [])
        self.line_f1.set_data([], [])
        self.line_auc.set_data([], [])
        self.vline_acc.set_alpha(0.0)
        self.vline_auc.set_alpha(0.0)
        self.ann_r6_acc.set_alpha(0.0)
        self.ann_r6_auc.set_alpha(0.0)
        self.fl_canvas.draw_idle()

        # Reset metrics
        for key, lbl_str in [("acc","—"),("f1","—"),("auc","—")]:
            self.metric_vars[key].set(lbl_str)
            cell, lbl = self.metric_frames[key]
            lbl.config(fg=HEX_GREEN)
            cell.config(bg=HEX_CARD)

        try:
            self.best_banner.pack_forget()
        except Exception:
            pass

        self.lbl_fl_status.config(
            text="Click 'Next Round' to begin the simulation."
        )
        self._fl_update_display(0)

    # ══════════════════════════════════════════════════════════
    # Detection helpers
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _sep(parent):
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

    def _det_show_placeholder(self):
        img = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), dtype=np.uint8)
        img[:] = (20, 20, 35)
        cv2.putText(img, "No video loaded",
                    (DISPLAY_MAX_W // 2 - 140, DISPLAY_MAX_H // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 130), 2)
        cv2.putText(img, "Click  Open Video  to begin",
                    (DISPLAY_MAX_W // 2 - 200, DISPLAY_MAX_H // 2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55, 55, 90), 1)
        self._det_render(img)

    def _det_open_file(self):
        path = filedialog.askopenfilename(
            title="Select a video file", filetypes=VIDEO_EXTS
        )
        if path:
            self._det_load_video(path)

    def _det_load_video(self, path: str):
        self._det_cancel_loop()
        self._det_stop()
        if self.det_cap:
            self.det_cap.release()

        self.det_cap = cv2.VideoCapture(path)
        if not self.det_cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            return

        self.det_video_path    = path
        self.det_total_frames  = int(self.det_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.det_video_fps     = self.det_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.det_frame_idx     = 0
        self.det_history.clear()
        self.det_raw_prob      = 0.5
        self.det_smoothed      = 0.5
        self.det_label         = "NO FACE"
        self.det_cv_color      = COLOR_NO_FACE
        self.det_hex_color     = HEX_NEUTRAL
        self.det_confidence    = 0.0
        self.det_inf_ms        = 0.0
        self.det_bbox          = None
        self.det_verdict_counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
        self.det_frame_log     = []
        self.det_last_det      = None
        self.det_fps_display   = 0.0
        self.det_t_prev        = time.time()

        vid_w = int(self.det_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.det_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(DISPLAY_MAX_W / vid_w, DISPLAY_MAX_H / vid_h, 1.0)
        self.det_display_w = int(vid_w * scale)
        self.det_display_h = int(vid_h * scale)
        self.det_vid_w     = vid_w
        self.det_vid_h     = vid_h

        self.det_progress_bar.config(to=self.det_total_frames)
        fname = Path(path).name
        self.det_lbl_status.config(
            text=f"{fname}  |  {vid_w}×{vid_h}  |  "
                 f"{self.det_video_fps:.1f} fps  |  {self.det_total_frames} frames"
        )
        self.root.title(f"Enhanced FL Demo — {fname}")
        self._det_update_overall_verdict()

        ret, first = self.det_cap.read()
        if ret:
            self._det_render_and_show(first)
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.det_after_id = self.root.after(33, self._det_update_loop)

    def _det_start_inference_thread(self):
        self.det_infer_result = {
            "raw_prob": 0.5, "smoothed": 0.5,
            "label": "NO FACE", "cv_color": COLOR_NO_FACE,
            "hex_color": HEX_NEUTRAL, "confidence": 0.0,
            "inf_ms": 0.0, "bbox": None,
        }

        def worker():
            while not self.det_stop_event.is_set():
                crop = bb = None
                with self.det_infer_queue_lock:
                    if self.det_infer_queue:
                        crop, bb = self.det_infer_queue.pop()
                        self.det_infer_queue.clear()
                if crop is None:
                    time.sleep(0.005)
                    continue
                t0 = time.perf_counter()
                inp_arr = preprocess_face_tflite(
                    crop, self.inp_dtype, self.inp_scale, self.inp_zp
                )
                self.interpreter.set_tensor(self.inp["index"], inp_arr)
                self.interpreter.invoke()
                raw_out  = self.interpreter.get_tensor(self.out["index"])
                raw_prob = float((raw_out[0][0] - self.out_zp) * self.out_scale) \
                           if self.inp_dtype == np.uint8 else float(raw_out[0][0])
                inf_ms   = (time.perf_counter() - t0) * 1000

                self.det_history.append(raw_prob)
                smoothed = float(np.mean(self.det_history))
                label, cv_color, hex_color, confidence = classify(smoothed)

                with self.det_infer_lock:
                    self.det_infer_result.update({
                        "raw_prob": raw_prob, "smoothed": smoothed,
                        "label": label, "cv_color": cv_color,
                        "hex_color": hex_color, "confidence": confidence,
                        "inf_ms": inf_ms, "bbox": bb,
                    })

        threading.Thread(target=worker, daemon=True).start()

    def _det_cancel_loop(self):
        if self.det_after_id is not None:
            try:
                self.root.after_cancel(self.det_after_id)
            except Exception:
                pass
            self.det_after_id = None

    def _det_update_loop(self):
        if self.det_playing and self.det_cap and self.det_cap.isOpened():
            if self.det_seek_pending is not None:
                self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, self.det_seek_pending)
                self.det_history.clear()
                self.det_seek_pending = None

            ret, frame_bgr = self.det_cap.read()
            if not ret:
                self.det_playing = False
                self.det_btn_play.config(text="▶")
                self.det_lbl_status.config(text="Playback complete.")
                self._det_update_overall_verdict()
            else:
                self.det_frame_idx = int(self.det_cap.get(cv2.CAP_PROP_POS_FRAMES))
                t_now = time.time()
                self.det_fps_display = (
                    0.9 * self.det_fps_display +
                    0.1 * (1.0 / max(t_now - self.det_t_prev, 1e-6))
                )
                self.det_t_prev = t_now

                if self.det_frame_idx % FRAME_SKIP == 0:
                    small = cv2.resize(frame_bgr, (DETECT_W, DETECT_H),
                                       interpolation=cv2.INTER_LINEAR)
                    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    results = self.face_detector.process(small_rgb)
                    self.det_last_det = results
                else:
                    results = self.det_last_det

                det_scale_x = self.det_vid_w / DETECT_W
                det_scale_y = self.det_vid_h / DETECT_H

                if results and results.detections:
                    largest = max(
                        results.detections,
                        key=lambda d: (
                            d.location_data.relative_bounding_box.width *
                            d.location_data.relative_bounding_box.height
                        )
                    )
                    rb = largest.location_data.relative_bounding_box
                    x1 = int(rb.xmin  * DETECT_W * det_scale_x)
                    y1 = int(rb.ymin  * DETECT_H * det_scale_y)
                    bw = int(rb.width * DETECT_W * det_scale_x)
                    bh = int(rb.height* DETECT_H * det_scale_y)
                    px = int(bw * FACE_PADDING)
                    py = int(bh * FACE_PADDING)
                    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
                    x2 = min(self.det_vid_w, x1 + bw + 2*px)
                    y2 = min(self.det_vid_h, y1 + bh + 2*py)
                    if x2 > x1 and y2 > y1:
                        face_crop = frame_bgr[y1:y2, x1:x2]
                        with self.det_infer_queue_lock:
                            self.det_infer_queue.clear()
                            self.det_infer_queue.append(
                                (face_crop.copy(), (x1, y1, x2, y2))
                            )

                with self.det_infer_lock:
                    res = dict(self.det_infer_result)

                no_face    = not (results and results.detections)
                label      = "NO FACE" if no_face else res["label"]
                cv_color   = COLOR_NO_FACE if no_face else res["cv_color"]
                hex_color  = HEX_NEUTRAL   if no_face else res["hex_color"]
                confidence = 0.0           if no_face else res["confidence"]
                raw_prob   = res["raw_prob"]
                smoothed   = res["smoothed"]
                inf_ms     = res["inf_ms"]
                bbox       = None          if no_face else res["bbox"]

                if not no_face and label in self.det_verdict_counts:
                    self.det_verdict_counts[label] += 1

                if len(self.det_frame_log) < 50_000:
                    self.det_frame_log.append({
                        "frame":      self.det_frame_idx,
                        "face_found": not no_face,
                        "raw_prob":   round(raw_prob, 6),
                        "smoothed":   round(smoothed, 6),
                        "label":      label,
                        "confidence": round(confidence, 2),
                        "inf_ms":     round(inf_ms, 2),
                    })

                self._det_render_and_show(
                    frame_bgr, bbox, label, cv_color,
                    confidence, raw_prob, smoothed, inf_ms
                )
                self._det_update_stats(
                    label, hex_color, confidence,
                    raw_prob, smoothed, inf_ms
                )

                self.det_progress_var.set(self.det_frame_idx)
                elapsed = self.det_frame_idx / max(self.det_video_fps, 1)
                total_s = self.det_total_frames / max(self.det_video_fps, 1)
                self.det_lbl_time.config(
                    text=f"{self._fmt_time(elapsed)} / {self._fmt_time(total_s)}"
                )

                delay = max(1, int((1000 / self.det_video_fps) / self.det_speed))
                self.det_after_id = self.root.after(delay, self._det_update_loop)
                return

        if self.det_loop_active:
            self.det_after_id = self.root.after(33, self._det_update_loop)

    def _det_render(self, frame_bgr):
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.det_canvas.imgtk = imgtk
        self.det_canvas.config(image=imgtk)

    def _det_render_and_show(self, frame_bgr, bbox=None, label="",
                              cv_color=COLOR_NO_FACE, confidence=0.0,
                              raw_prob=0.5, smoothed=0.5, inf_ms=0.0):
        frame = frame_bgr.copy() if bbox is not None else frame_bgr
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)
            tag = f"{label}  {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), cv_color, -1)
            cv2.putText(frame, tag, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            bar_w  = x2 - x1
            filled = int(bar_w * min(confidence, 100.0) / 100.0)
            cv2.rectangle(frame, (x1, y2+4), (x2, y2+16), (40,40,40), -1)
            cv2.rectangle(frame, (x1, y2+4), (x1+filled, y2+16), cv_color, -1)

        disp  = cv2.resize(frame, (self.det_display_w, self.det_display_h),
                           interpolation=cv2.INTER_LINEAR)
        self._det_render(disp)

    def _det_update_stats(self, label, hex_color, confidence,
                           raw_prob, smoothed, inf_ms):
        self.det_lbl_verdict.config(text=label, fg=hex_color)
        self.det_lbl_confidence.config(
            text=f"Confidence: {confidence:.1f}%", fg=hex_color
        )
        self.det_conf_bar["value"] = min(confidence, 100.0)
        self.det_stat_vars["frame"].set(
            f"{self.det_frame_idx} / {self.det_total_frames}"
        )
        self.det_stat_vars["fps"].set(f"{self.det_fps_display:.1f}")
        self.det_stat_vars["inf_ms"].set(f"{inf_ms:.1f} ms")
        self.det_stat_vars["raw"].set(f"{raw_prob:.4f}")
        self.det_stat_vars["smooth"].set(f"{smoothed:.4f}")

        total_inf = sum(self.det_verdict_counts.values())
        for verdict, var in self.det_tally_vars.items():
            count = self.det_verdict_counts[verdict]
            pct   = count / total_inf * 100 if total_inf > 0 else 0.0
            var.set(f"{count}  ({pct:.1f}%)")
        self._det_update_overall_verdict()

    def _det_update_overall_verdict(self):
        total = sum(self.det_verdict_counts.values())
        if total == 0:
            self.det_lbl_overall.config(text="—", fg=HEX_NEUTRAL)
            self.det_lbl_overall_sub.config(text="No inference yet.")
            return
        dominant = max(self.det_verdict_counts, key=self.det_verdict_counts.get)
        pct = self.det_verdict_counts[dominant] / total * 100
        hex_map = {"REAL": HEX_REAL, "FAKE": HEX_FAKE, "UNCERTAIN": HEX_UNC}
        self.det_lbl_overall.config(text=dominant, fg=hex_map[dominant])
        self.det_lbl_overall_sub.config(
            text=f"{dominant} in {pct:.1f}% of\n{total} inferred frames"
        )

    def _det_toggle_play(self):
        if self.det_cap is None:
            self._det_open_file()
            return
        self.det_playing = not self.det_playing
        self.det_btn_play.config(text="⏸" if self.det_playing else "▶")
        if self.det_playing:
            self.det_t_prev = time.time()

    def _det_stop(self):
        self.det_playing = False
        self.det_btn_play.config(text="▶")
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.det_frame_idx = 0
            self.det_progress_var.set(0)
            self.det_lbl_time.config(text="0:00 / 0:00")

    def _det_restart(self):
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.det_frame_idx = 0
            self.det_history.clear()
            self.det_verdict_counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
            self.det_frame_log = []
            self.det_progress_var.set(0)
            self._det_update_overall_verdict()
            self.det_playing = True
            self.det_btn_play.config(text="⏸")
            self.det_t_prev = time.time()

    def _det_on_seek(self, val):
        if self.det_cap:
            self.det_seek_pending = int(float(val))

    def _det_on_speed_change(self, event=None):
        val = self.det_speed_var.get().replace("×", "")
        try:
            self.det_speed = float(val)
        except ValueError:
            self.det_speed = 1.0

    def _det_export_csv(self):
        if not self.det_frame_log:
            messagebox.showinfo("Export",
                "No inference data yet.\nPlay the video first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"),("All files","*.*")],
            initialfile="deepfake_results.csv"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.det_frame_log[0].keys())
            writer.writeheader()
            writer.writerows(self.det_frame_log)
        messagebox.showinfo("Exported",
            f"Saved {len(self.det_frame_log)} rows to:\n{path}")

    def _det_save_frame(self):
        if self.det_cap is None:
            return
        pos = int(self.det_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos-1))
        ret, frame = self.det_cap.read()
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if not ret:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files","*.png"),("All files","*.*")],
            initialfile=f"frame_{self.det_frame_idx:05d}.png"
        )
        if path:
            cv2.imwrite(path, frame)
            messagebox.showinfo("Saved", f"Frame saved to:\n{path}")

    # ── Shared helpers ────────────────────────────────────────
    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    def on_close(self):
        self.det_loop_active = False
        self._det_cancel_loop()
        self.det_stop_event.set()
        self.det_playing = False
        if self.det_cap:
            self.det_cap.release()
        self.face_detector.close()
        plt.close("all")
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Defense Demo v2 — Enhanced FL Cycle"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to .tflite model (auto-detected if omitted)"
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"[INFO] Using model: {model_path}")

    root = tk.Tk()
    app  = DefenseDemo(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Center and size window
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = 1400, 820
    root.geometry(f"{ww}x{wh}+{max(0,(sw-ww)//2)}+{max(0,(sh-wh)//2)}")

    root.mainloop()


if __name__ == "__main__":
    main()
