"""
Deepfake Detector — Video File Analyser
========================================
GUI application that lets the user select a video file, extracts frames,
detects faces, and runs TFLite deepfake inference on each frame.

Preprocessing pipeline is IDENTICAL to validate_model.py (96%+ accuracy):
  BGR → RGB → tf.cast → tf.image.resize(260,260)
  → preprocess_input → TFLite inference → temporal smoothing

Features:
  • File browser (MP4, AVI, MOV, MKV, WEBM)
  • Real-time video playback with inference overlay
  • Per-frame stats: frame number, FPS, inference time, raw prob, smoothed prob
  • Bounding box + label banner on detected face
  • Confidence bar under face box
  • Live stats panel (top-left)
  • Overall verdict panel (bottom) — final REAL/FAKE/UNCERTAIN summary
  • Playback controls: Play / Pause / Stop / Restart
  • Speed control: 0.25×  0.5×  1×  2×  4×
  • Progress bar with seek-by-click
  • Export report as CSV (per-frame results)
  • Save current frame as PNG

Requirements:
    pip install -r requirements.txt
    (tkinter is included with Python — no extra install needed)

Usage:
    python deepfake_detector_video.py
    python deepfake_detector_video.py --model path/to/model.tflite
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
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─────────────────────────────────────────────────────────────
# TFLite runtime
# ─────────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
    print("[INFO] Using ai_edge_litert")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("[INFO] Using tflite_runtime")
    except ImportError:
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using tensorflow.lite.Interpreter")

# ─────────────────────────────────────────────────────────────
# Constants — match training / validate_model.py exactly
# ─────────────────────────────────────────────────────────────
INPUT_SIZE     = 260
FACE_PADDING   = 0.20
HISTORY_SIZE   = 8
REAL_THRESHOLD = 0.67
FAKE_THRESHOLD = 0.45
FRAME_SKIP = 4

DISPLAY_MAX_W  = 960
DISPLAY_MAX_H  = 540
DETECT_W       = 640
DETECT_H       = 360

# BGR colours for OpenCV overlay
COLOR_REAL      = (136, 255,  68)
COLOR_FAKE      = ( 68,  68, 255)
COLOR_UNCERTAIN = (  0, 255, 255)
COLOR_NO_FACE   = (  0, 200, 255)

# Hex colours for tkinter widgets
HEX_REAL      = "#44FF44"
HEX_FAKE      = "#FF4444"
HEX_UNCERTAIN = "#FFFF00"
HEX_NEUTRAL   = "#AAAAAA"
HEX_BG        = "#1E1E1E"
HEX_PANEL     = "#2A2A2A"
HEX_TEXT      = "#E0E0E0"
HEX_ACCENT    = "#3A6EA5"

SPEED_OPTIONS  = [0.25, 0.5, 1.0, 2.0, 4.0]

VIDEO_EXTS = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v"),
    ("All files",   "*.*"),
)


# ═════════════════════════════════════════════════════════════
# Preprocessing — identical to validate_model.py
# ═════════════════════════════════════════════════════════════
def preprocess_face_tflite(face_bgr: np.ndarray,
                            input_dtype,
                            input_scale: float,
                            input_zero_point: int) -> np.ndarray:
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


# ═════════════════════════════════════════════════════════════
# Classification
# ═════════════════════════════════════════════════════════════
def classify(smoothed: float):
    if smoothed >= REAL_THRESHOLD:
        return "REAL",      COLOR_REAL,      HEX_REAL,      smoothed * 100.0
    elif smoothed < FAKE_THRESHOLD:
        return "FAKE",      COLOR_FAKE,      HEX_FAKE,      (1.0 - smoothed) * 100.0
    else:
        conf = (abs(smoothed - 0.575) / 0.375) * 100.0
        return "UNCERTAIN", COLOR_UNCERTAIN, HEX_UNCERTAIN, min(conf, 100.0)


# ═════════════════════════════════════════════════════════════
# OpenCV overlay — drawn on each video frame
# ═════════════════════════════════════════════════════════════
def draw_overlay(frame, bbox, label, cv_color, confidence,
                 raw_prob, smoothed, fps, frame_idx,
                 total_frames, inf_ms, verdict_counts):
    h, w = frame.shape[:2]

    # ── Face bounding box ────────────────────────────────────
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)

        tag = f"{label}  {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), cv_color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # Confidence bar
        bar_w  = x2 - x1
        filled = int(bar_w * min(confidence, 100.0) / 100.0)
        cv2.rectangle(frame, (x1, y2 + 4), (x2,          y2 + 16), (40, 40, 40), -1)
        cv2.rectangle(frame, (x1, y2 + 4), (x1 + filled, y2 + 16), cv_color,     -1)

    # ── Stats panel — top left ───────────────────────────────
    progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
    stats = [
        f"Frame  : {frame_idx} / {total_frames}",
        f"Prog   : {progress_pct:.1f}%",
        f"FPS    : {fps:.1f}",
        f"Inf ms : {inf_ms:.1f}",
        f"Raw    : {raw_prob:.4f}",
        f"Smooth : {smoothed:.4f}",
        f"Label  : {label}",
    ]
    panel_h = len(stats) * 22 + 12
    cv2.rectangle(frame, (6, 6), (240, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (6, 6), (240, panel_h), (80, 80, 80), 1)
    for i, line in enumerate(stats):
        cv2.putText(frame, line, (12, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

    # ── Verdict tally — bottom right ────────────────────────
    total_inf = sum(verdict_counts.values())
    if total_inf > 0:
        real_pct = verdict_counts["REAL"]      / total_inf * 100
        fake_pct = verdict_counts["FAKE"]      / total_inf * 100
        unc_pct  = verdict_counts["UNCERTAIN"] / total_inf * 100
        tally = [
            ("REAL",      real_pct, COLOR_REAL),
            ("FAKE",      fake_pct, COLOR_FAKE),
            ("UNCERTAIN", unc_pct,  COLOR_UNCERTAIN),
        ]
        panel_w = 200
        px = w - panel_w - 6
        py = h - len(tally) * 22 - 14
        cv2.rectangle(frame, (px - 2, py - 2), (w - 4, h - 4), (0, 0, 0), -1)
        cv2.rectangle(frame, (px - 2, py - 2), (w - 4, h - 4), (80, 80, 80), 1)
        for i, (lbl, pct, col) in enumerate(tally):
            cv2.putText(frame, f"{lbl}: {pct:.1f}%",
                        (px, py + 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)


# ═════════════════════════════════════════════════════════════
# Model loader + sanity check
# ═════════════════════════════════════════════════════════════
def load_model(model_path: str):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"[INFO] Input  shape={inp['shape'].tolist()} dtype={inp['dtype'].__name__}")
    print(f"[INFO] Output shape={out['shape'].tolist()} dtype={out['dtype'].__name__}")

    print("[INFO] Sanity check...")
    results = {}
    for val, name in [(-1.0, "all-black"), (0.0, "mid-grey"), (1.0, "all-white")]:
        buf = np.full(inp["shape"], val, dtype=np.float32)
        if inp["dtype"] == np.uint8:
            sc, zp = inp.get("quantization", (1.0, 0))
            buf = np.clip(buf / sc + zp, 0, 255).astype(np.uint8)
        interpreter.set_tensor(inp["index"], buf)
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        sc2, zp2 = out.get("quantization", (1.0, 0))
        prob = float((raw[0][0] - zp2) * sc2) if out["dtype"] == np.uint8 \
               else float(raw[0][0])
        results[name] = prob
        print(f"  {name:12s}: {prob:.6f}")

    spread = max(results.values()) - min(results.values())
    print(f"  Spread: {spread:.4f}  "
          f"({'OK' if spread > 0.05 else 'WARN — model may not respond to input'})")
    return interpreter, inp, out


# ═════════════════════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════════════════════
class DeepfakeApp:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root       = root
        self.model_path = model_path

        # ── Model ────────────────────────────────────────────
        self.interpreter, self.inp, self.out = load_model(model_path)
        self.inp_dtype = self.inp["dtype"]
        self.inp_scale, self.inp_zp = self.inp.get("quantization", (1.0, 0))
        self.out_scale, self.out_zp = self.out.get("quantization", (1.0, 0))

        # ── MediaPipe ────────────────────────────────────────
        mp_fd = mp.solutions.face_detection
        self.face_detector = mp_fd.FaceDetection(
            model_selection=1,           # model 1 = full-range, better for video
            min_detection_confidence=0.5
        )

        # ── Playback state ───────────────────────────────────
        self.cap           = None
        self.video_path    = None
        self.total_frames  = 0
        self.video_fps     = 30.0
        self.frame_idx     = 0
        self.playing       = False
        self.speed         = 1.0
        self.seek_pending  = None   # frame number to seek to
        self._after_id     = None   # tkinter after() handle — cancel before rescheduling
        self._loop_active  = True   # set False only on app close

        # ── Inference state ──────────────────────────────────
        self.history        = collections.deque(maxlen=HISTORY_SIZE)
        self.raw_prob       = 0.5
        self.smoothed       = 0.5
        self.label          = "NO FACE"
        self.cv_color       = COLOR_NO_FACE
        self.hex_color      = HEX_NEUTRAL
        self.confidence     = 0.0
        self.inf_ms         = 0.0
        self.bbox           = None
        self.verdict_counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
        self.frame_log      = []   # list of dicts for CSV export
        self._last_detection = None

        # ── Display ──────────────────────────────────────────
        self.display_w = DISPLAY_MAX_W
        self.display_h = DISPLAY_MAX_H
        self.fps_display  = 0.0
        self.t_prev       = time.time()

        # ── Threading ────────────────────────────────────────
        self.infer_lock       = threading.Lock()
        self.infer_queue      = []
        self.infer_queue_lock = threading.Lock()
        self.stop_event       = threading.Event()

        self._build_ui()
        self._start_inference_thread()
        self._after_id = self.root.after(33, self._update_loop)

    # ─────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Deepfake Detector — Video Analyser")
        self.root.configure(bg=HEX_BG)
        self.root.resizable(True, True)

        # ── Top toolbar ──────────────────────────────────────
        toolbar = tk.Frame(self.root, bg=HEX_PANEL, pady=6)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Button(toolbar, text="📂  Open Video",
                  command=self._open_file,
                  bg=HEX_ACCENT, fg="white", font=("Helvetica", 11, "bold"),
                  relief=tk.FLAT, padx=14, pady=4
                  ).pack(side=tk.LEFT, padx=(10, 4))

        tk.Button(toolbar, text="💾  Export CSV",
                  command=self._export_csv,
                  bg="#4A4A4A", fg=HEX_TEXT, font=("Helvetica", 10),
                  relief=tk.FLAT, padx=10, pady=4
                  ).pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar, text="🖼  Save Frame",
                  command=self._save_frame,
                  bg="#4A4A4A", fg=HEX_TEXT, font=("Helvetica", 10),
                  relief=tk.FLAT, padx=10, pady=4
                  ).pack(side=tk.LEFT, padx=4)

        # Model label
        model_name = Path(self.model_path).name
        tk.Label(toolbar, text=f"Model: {model_name}",
                 bg=HEX_PANEL, fg="#888888", font=("Helvetica", 9)
                 ).pack(side=tk.RIGHT, padx=12)

        # ── Video canvas ─────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=HEX_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))

        self.canvas = tk.Label(canvas_frame, bg="black",
                               width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Right stats panel ─────────────────────────────────
        stats_frame = tk.Frame(canvas_frame, bg=HEX_PANEL,
                               width=200, relief=tk.FLAT)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        stats_frame.pack_propagate(False)

        tk.Label(stats_frame, text="INFERENCE STATS",
                 bg=HEX_PANEL, fg="#888888",
                 font=("Helvetica", 9, "bold")).pack(pady=(12, 4))

        # Verdict banner
        self.lbl_verdict = tk.Label(
            stats_frame, text="—",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 28, "bold")
        )
        self.lbl_verdict.pack(pady=(4, 2))

        self.lbl_confidence = tk.Label(
            stats_frame, text="Confidence: —",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 10)
        )
        self.lbl_confidence.pack()

        # Confidence progress bar
        self.conf_bar = ttk.Progressbar(
            stats_frame, orient="horizontal", length=160, mode="determinate"
        )
        self.conf_bar.pack(pady=6)

        ttk.Separator(stats_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        # Per-frame stats labels
        self.stat_vars = {}
        stat_rows = [
            ("frame",   "Frame"),
            ("fps",     "Display FPS"),
            ("inf_ms",  "Inference ms"),
            ("raw",     "Raw prob"),
            ("smooth",  "Smoothed prob"),
        ]
        for key, label in stat_rows:
            row = tk.Frame(stats_frame, bg=HEX_PANEL)
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=label + ":", bg=HEX_PANEL, fg="#888888",
                     font=("Helvetica", 9), anchor="w", width=14
                     ).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=HEX_PANEL, fg=HEX_TEXT,
                     font=("Helvetica", 9, "bold"), anchor="e"
                     ).pack(side=tk.RIGHT)

        ttk.Separator(stats_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        # Tally counts
        tk.Label(stats_frame, text="FRAME TALLY",
                 bg=HEX_PANEL, fg="#888888",
                 font=("Helvetica", 9, "bold")).pack()

        self.tally_vars = {}
        for verdict, hex_c in [("REAL", HEX_REAL), ("FAKE", HEX_FAKE),
                                ("UNCERTAIN", HEX_UNCERTAIN)]:
            row = tk.Frame(stats_frame, bg=HEX_PANEL)
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=verdict + ":", bg=HEX_PANEL, fg=hex_c,
                     font=("Helvetica", 9, "bold"), anchor="w", width=10
                     ).pack(side=tk.LEFT)
            var = tk.StringVar(value="0  (0.0%)")
            self.tally_vars[verdict] = var
            tk.Label(row, textvariable=var, bg=HEX_PANEL, fg=HEX_TEXT,
                     font=("Helvetica", 9), anchor="e"
                     ).pack(side=tk.RIGHT)

        ttk.Separator(stats_frame, orient="horizontal").pack(fill=tk.X, pady=8)

        # Overall verdict
        tk.Label(stats_frame, text="OVERALL VERDICT",
                 bg=HEX_PANEL, fg="#888888",
                 font=("Helvetica", 9, "bold")).pack()
        self.lbl_overall = tk.Label(
            stats_frame, text="—",
            bg=HEX_PANEL, fg=HEX_NEUTRAL,
            font=("Helvetica", 16, "bold")
        )
        self.lbl_overall.pack(pady=4)
        self.lbl_overall_sub = tk.Label(
            stats_frame, text="",
            bg=HEX_PANEL, fg="#888888",
            font=("Helvetica", 9), wraplength=180, justify="center"
        )
        self.lbl_overall_sub.pack(padx=8)

        # ── Progress bar ─────────────────────────────────────
        prog_frame = tk.Frame(self.root, bg=HEX_BG, pady=4)
        prog_frame.pack(fill=tk.X, padx=10)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Scale(
            prog_frame, from_=0, to=100,
            orient="horizontal", variable=self.progress_var,
            command=self._on_seek
        )
        self.progress_bar.pack(fill=tk.X)

        self.lbl_time = tk.Label(
            prog_frame, text="0:00 / 0:00",
            bg=HEX_BG, fg="#888888", font=("Helvetica", 9)
        )
        self.lbl_time.pack(anchor="e")

        # ── Playback controls ─────────────────────────────────
        ctrl_frame = tk.Frame(self.root, bg=HEX_PANEL, pady=8)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_cfg = dict(bg="#4A4A4A", fg=HEX_TEXT, font=("Helvetica", 12),
                       relief=tk.FLAT, width=4, pady=3)

        self.btn_play = tk.Button(ctrl_frame, text="▶",
                                   command=self._toggle_play, **btn_cfg)
        self.btn_play.pack(side=tk.LEFT, padx=(12, 2))

        tk.Button(ctrl_frame, text="⏹", command=self._stop,
                  **btn_cfg).pack(side=tk.LEFT, padx=2)

        tk.Button(ctrl_frame, text="⟳", command=self._restart,
                  **btn_cfg).pack(side=tk.LEFT, padx=2)

        # Speed selector
        tk.Label(ctrl_frame, text="Speed:",
                 bg=HEX_PANEL, fg=HEX_TEXT,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(16, 4))

        self.speed_var = tk.StringVar(value="1.0×")
        speed_menu = ttk.Combobox(
            ctrl_frame, textvariable=self.speed_var,
            values=[f"{s}×" for s in SPEED_OPTIONS],
            width=6, state="readonly"
        )
        speed_menu.pack(side=tk.LEFT, padx=4)
        speed_menu.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Status label
        self.lbl_status = tk.Label(
            ctrl_frame, text="Open a video file to begin.",
            bg=HEX_PANEL, fg="#888888", font=("Helvetica", 10)
        )
        self.lbl_status.pack(side=tk.LEFT, padx=16)

        # Keyboard shortcuts
        self.root.bind("<space>",   lambda e: self._toggle_play())
        self.root.bind("<Escape>",  lambda e: self._stop())
        self.root.bind("<r>",       lambda e: self._restart())
        self.root.bind("<s>",       lambda e: self._save_frame())

        # Placeholder image
        self._show_placeholder()

    # ─────────────────────────────────────────────────────────
    # Placeholder canvas
    # ─────────────────────────────────────────────────────────
    def _show_placeholder(self):
        img = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.putText(img, "No video loaded",
                    (DISPLAY_MAX_W // 2 - 130, DISPLAY_MAX_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.putText(img, "Click  Open Video  to begin",
                    (DISPLAY_MAX_W // 2 - 180, DISPLAY_MAX_H // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 1)
        self._display_frame(img)

    # ─────────────────────────────────────────────────────────
    # File open
    # ─────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=VIDEO_EXTS
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str):
        self._cancel_loop()
        self._stop()
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            return

        self.video_path   = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_idx    = 0
        self.history.clear()
        self.raw_prob       = 0.5
        self.smoothed       = 0.5
        self.label          = "NO FACE"
        self.cv_color       = COLOR_NO_FACE
        self.hex_color      = HEX_NEUTRAL
        self.confidence     = 0.0
        self.inf_ms         = 0.0
        self.bbox           = None
        self.verdict_counts  = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
        self.frame_log       = []
        self._last_detection = None
        self.fps_display     = 0.0
        self.t_prev          = time.time()

        # Compute display scale
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(DISPLAY_MAX_W / vid_w, DISPLAY_MAX_H / vid_h, 1.0)
        self.display_w = int(vid_w * scale)
        self.display_h = int(vid_h * scale)
        self.vid_w = vid_w
        self.vid_h = vid_h

        self.progress_bar.config(to=self.total_frames)
        fname = Path(path).name
        self.lbl_status.config(
            text=f"{fname}  |  {vid_w}×{vid_h}  |  {self.video_fps:.1f} fps  "
                 f"|  {self.total_frames} frames"
        )
        self.root.title(f"Deepfake Detector — {fname}")
        self._update_overall_verdict()

        # Read and show first frame without playing
        ret, first = self.cap.read()
        if ret:
            self._render_and_show(first)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Restart the update loop (was cancelled at top of this method)
        self._after_id = self.root.after(33, self._update_loop)

    # ─────────────────────────────────────────────────────────
    # Inference background thread
    # ─────────────────────────────────────────────────────────
    def _cancel_loop(self):
        """Cancel any pending root.after() callback to prevent duplicate loops."""
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _start_inference_thread(self):
        self.infer_result = {
            "raw_prob": 0.5, "smoothed": 0.5,
            "label": "NO FACE", "cv_color": COLOR_NO_FACE,
            "hex_color": HEX_NEUTRAL, "confidence": 0.0,
            "inf_ms": 0.0, "bbox": None,
        }

        def worker():
            while not self.stop_event.is_set():
                crop = bb = None
                with self.infer_queue_lock:
                    if self.infer_queue:
                        crop, bb = self.infer_queue.pop()
                        self.infer_queue.clear()
                if crop is None:
                    time.sleep(0.005)
                    continue

                t0 = time.perf_counter()
                inp_arr = preprocess_face_tflite(
                    crop, self.inp_dtype, self.inp_scale, self.inp_zp
                )
                self.interpreter.set_tensor(self.inp["index"], inp_arr)
                self.interpreter.invoke()
                raw_out = self.interpreter.get_tensor(self.out["index"])
                raw_prob = float((raw_out[0][0] - self.out_zp) * self.out_scale) \
                           if self.inp_dtype == np.uint8 else float(raw_out[0][0])
                inf_ms = (time.perf_counter() - t0) * 1000

                self.history.append(raw_prob)
                smoothed = float(np.mean(self.history))
                label, cv_color, hex_color, confidence = classify(smoothed)

                with self.infer_lock:
                    self.infer_result.update({
                        "raw_prob":   raw_prob,
                        "smoothed":   smoothed,
                        "label":      label,
                        "cv_color":   cv_color,
                        "hex_color":  hex_color,
                        "confidence": confidence,
                        "inf_ms":     inf_ms,
                        "bbox":       bb,
                    })

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # ─────────────────────────────────────────────────────────
    # Main update loop — reads frames, updates UI
    # ─────────────────────────────────────────────────────────
    def _update_loop(self):
        if self.playing and self.cap and self.cap.isOpened():
            # Handle pending seek
            if self.seek_pending is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_pending)
                self.history.clear()
                self.seek_pending = None

            ret, frame_bgr = self.cap.read()
            if not ret:
                # End of video
                self.playing = False
                self.btn_play.config(text="▶")
                self.lbl_status.config(text="Playback complete.")
                self._update_overall_verdict()
            else:
                self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # FPS
                t_now = time.time()
                self.fps_display = (0.9 * self.fps_display +
                                    0.1 * (1.0 / max(t_now - self.t_prev, 1e-6)))
                self.t_prev = t_now

                # Face detection on downscaled frame
                det_scale_x = self.vid_w / DETECT_W
                det_scale_y = self.vid_h / DETECT_H

                # small     = cv2.resize(frame_bgr, (DETECT_W, DETECT_H),
                #                        interpolation=cv2.INTER_LINEAR)
                # small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                # results   = self.face_detector.process(small_rgb)

                if self.frame_idx % FRAME_SKIP == 0:
                    small     = cv2.resize(frame_bgr, (DETECT_W, DETECT_H),
                                        interpolation=cv2.INTER_LINEAR)
                    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    results   = self.face_detector.process(small_rgb)
                    self._last_detection = results   # cache for skipped frames
                else:
                    results = self._last_detection   # reuse last result

                if results and results.detections:
                    largest = max(
                        results.detections,
                        key=lambda d: (
                            d.location_data.relative_bounding_box.width *
                            d.location_data.relative_bounding_box.height
                        )
                    )
                    rb  = largest.location_data.relative_bounding_box
                    x1  = int(rb.xmin   * DETECT_W * det_scale_x)
                    y1  = int(rb.ymin   * DETECT_H * det_scale_y)
                    bw  = int(rb.width  * DETECT_W * det_scale_x)
                    bh  = int(rb.height * DETECT_H * det_scale_y)
                    px  = int(bw * FACE_PADDING)
                    py  = int(bh * FACE_PADDING)
                    x1  = max(0, x1 - px);  y1 = max(0, y1 - py)
                    x2  = min(self.vid_w, x1 + bw + 2 * px)
                    y2  = min(self.vid_h, y1 + bh + 2 * py)

                    if x2 > x1 and y2 > y1:
                        face_crop = frame_bgr[y1:y2, x1:x2]
                        with self.infer_queue_lock:
                            self.infer_queue.clear()
                            self.infer_queue.append((face_crop.copy(), (x1, y1, x2, y2)))

                # Read latest inference result
                with self.infer_lock:
                    res = dict(self.infer_result)

                no_face = not (results and results.detections)
                label      = "NO FACE" if no_face else res["label"]
                cv_color   = COLOR_NO_FACE if no_face else res["cv_color"]
                hex_color  = HEX_NEUTRAL   if no_face else res["hex_color"]
                confidence = 0.0 if no_face else res["confidence"]
                raw_prob   = res["raw_prob"]
                smoothed   = res["smoothed"]
                inf_ms     = res["inf_ms"]
                bbox       = None if no_face else res["bbox"]

                # Tally (only when face detected)
                if not no_face and label in self.verdict_counts:
                    self.verdict_counts[label] += 1

                # Log — capped at 50,000 rows to prevent unbounded memory growth
                if len(self.frame_log) < 50_000:
                    self.frame_log.append({
                        "frame":      self.frame_idx,
                        "face_found": not no_face,
                        "raw_prob":   round(raw_prob, 6),
                        "smoothed":   round(smoothed, 6),
                        "label":      label,
                        "confidence": round(confidence, 2),
                        "inf_ms":     round(inf_ms, 2),
                    })

                # Render frame with overlay
                self._render_and_show(
                    frame_bgr, bbox, label, cv_color, confidence,
                    raw_prob, smoothed, inf_ms
                )

                # Update right-panel stats
                self._update_stats(label, hex_color, confidence,
                                   raw_prob, smoothed, inf_ms)

                # Progress bar + time
                self.progress_var.set(self.frame_idx)
                elapsed = self.frame_idx / max(self.video_fps, 1)
                total_s = self.total_frames / max(self.video_fps, 1)
                self.lbl_time.config(
                    text=f"{self._fmt_time(elapsed)} / {self._fmt_time(total_s)}"
                )

                # Schedule next frame based on speed
                delay = max(1, int((1000 / self.video_fps) / self.speed))
                self._after_id = self.root.after(delay, self._update_loop)
                return

        # Not playing — reschedule idle poll (only if loop is still active)
        if self._loop_active:
            self._after_id = self.root.after(33, self._update_loop)

    # ─────────────────────────────────────────────────────────
    # Render frame → tkinter canvas
    # ─────────────────────────────────────────────────────────
    def _render_and_show(self, frame_bgr, bbox=None, label="",
                         cv_color=COLOR_NO_FACE, confidence=0.0,
                         raw_prob=0.5, smoothed=0.5, inf_ms=0.0):
        # draw_overlay modifies the frame in place — copy only if we have overlay
        frame = frame_bgr.copy() if bbox is not None else frame_bgr

        if bbox is not None:
            draw_overlay(
                frame, bbox, label, cv_color, confidence,
                raw_prob, smoothed, self.fps_display,
                self.frame_idx, self.total_frames,
                inf_ms, self.verdict_counts
            )

        # Resize for display
        if hasattr(self, "display_w"):
            disp = cv2.resize(frame, (self.display_w, self.display_h),
                              interpolation=cv2.INTER_LINEAR)
        else:
            disp = cv2.resize(frame, (DISPLAY_MAX_W, DISPLAY_MAX_H))

        rgb   = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.canvas.imgtk = imgtk   # prevent GC
        self.canvas.config(image=imgtk)

    def _display_frame(self, frame_bgr):
        self._render_and_show(frame_bgr)

    # ─────────────────────────────────────────────────────────
    # Stats panel update
    # ─────────────────────────────────────────────────────────
    def _update_stats(self, label, hex_color, confidence,
                      raw_prob, smoothed, inf_ms):
        self.lbl_verdict.config(text=label, fg=hex_color)
        self.lbl_confidence.config(
            text=f"Confidence: {confidence:.1f}%", fg=hex_color
        )
        self.conf_bar["value"] = min(confidence, 100.0)

        self.stat_vars["frame"].set(
            f"{self.frame_idx} / {self.total_frames}"
        )
        self.stat_vars["fps"].set(f"{self.fps_display:.1f}")
        self.stat_vars["inf_ms"].set(f"{inf_ms:.1f} ms")
        self.stat_vars["raw"].set(f"{raw_prob:.4f}")
        self.stat_vars["smooth"].set(f"{smoothed:.4f}")

        total_inf = sum(self.verdict_counts.values())
        for verdict, var in self.tally_vars.items():
            count = self.verdict_counts[verdict]
            pct   = count / total_inf * 100 if total_inf > 0 else 0.0
            var.set(f"{count}  ({pct:.1f}%)")

        self._update_overall_verdict()

    def _update_overall_verdict(self):
        total = sum(self.verdict_counts.values())
        if total == 0:
            self.lbl_overall.config(text="—", fg=HEX_NEUTRAL)
            self.lbl_overall_sub.config(text="No inference yet.")
            return

        dominant = max(self.verdict_counts, key=self.verdict_counts.get)
        pct = self.verdict_counts[dominant] / total * 100
        hex_map = {"REAL": HEX_REAL, "FAKE": HEX_FAKE, "UNCERTAIN": HEX_UNCERTAIN}
        self.lbl_overall.config(text=dominant, fg=hex_map[dominant])
        self.lbl_overall_sub.config(
            text=f"{dominant} in {pct:.1f}% of\n{total} inferred frames"
        )

    # ─────────────────────────────────────────────────────────
    # Playback controls
    # ─────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.cap is None:
            self._open_file()
            return
        self.playing = not self.playing
        self.btn_play.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self.t_prev = time.time()
            # Do NOT call _update_loop() directly — it is already running
            # via root.after(). Just setting self.playing=True is enough.

    def _stop(self):
        self.playing = False
        self.btn_play.config(text="▶")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.progress_var.set(0)
            self.lbl_time.config(text="0:00 / 0:00")

    def _restart(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.history.clear()
            self.verdict_counts = {"REAL": 0, "FAKE": 0, "UNCERTAIN": 0}
            del self.frame_log          # explicitly free old list memory
            self.frame_log = []
            import gc; gc.collect()
            self.progress_var.set(0)
            self._update_overall_verdict()
            self.playing = True
            self.btn_play.config(text="⏸")
            self.t_prev = time.time()

    def _on_seek(self, val):
        if self.cap:
            frame_no = int(float(val))
            self.seek_pending = frame_no

    def _on_speed_change(self, event=None):
        val = self.speed_var.get().replace("×", "")
        try:
            self.speed = float(val)
        except ValueError:
            self.speed = 1.0

    # ─────────────────────────────────────────────────────────
    # Export / Save
    # ─────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self.frame_log:
            messagebox.showinfo("Export", "No inference data to export yet.\n"
                                "Play the video first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="deepfake_results.csv"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.frame_log[0].keys())
            writer.writeheader()
            writer.writerows(self.frame_log)
        messagebox.showinfo("Exported",
                            f"Saved {len(self.frame_log)} rows to:\n{path}")

    def _save_frame(self):
        if self.cap is None:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if not ret:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"frame_{self.frame_idx:05d}.png"
        )
        if path:
            cv2.imwrite(path, frame)
            messagebox.showinfo("Saved", f"Frame saved to:\n{path}")

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    def on_close(self):
        self._loop_active = False
        self._cancel_loop()
        self.stop_event.set()
        self.playing = False
        if self.cap:
            self.cap.release()
        self.face_detector.close()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Deepfake detector — video file mode")
    parser.add_argument(
        "--model",
        default="effnet_global_fl_final_quantised.tflite",
        help="Path to .tflite model file"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Use: python deepfake_detector_video.py --model path/to/model.tflite")
        sys.exit(1)

    root = tk.Tk()
    app  = DeepfakeApp(root, str(model_path))
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Center window
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = root.winfo_width()
    wh = root.winfo_height()
    root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

    root.mainloop()


if __name__ == "__main__":
    main()