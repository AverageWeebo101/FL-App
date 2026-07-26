"""
Defense Demo — Enhanced Federated Learning Cycle  (v6)
=======================================================
This script demonstrates an enhanced federated learning cycle with a focus on client selection, training, 
and reputation management. It includes a visual representation of the process using Tkinter and Matplotlib, 
showcasing the interactions between clients and the server throughout multiple rounds of training.
"""

import argparse
import collections
import csv
import math
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

try:
    from keras.applications.efficientnet import preprocess_input
except ImportError:
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

# ══════════════════════════════════════════════════════════════
# TRAINING DATA
# ══════════════════════════════════════════════════════════════
ROUND_DATA = [
    (0,  0.9435, 0.9435, 0.9901, None,   None,   []),
    (1,  0.9478, 0.9478, 0.9924, 0.9504, 0.9496, ["006","004","005"]),
    (2,  0.9543, 0.9543, 0.9963, 0.9558, 0.9558, ["000","003","008"]),
    (3,  0.9543, 0.9543, 0.9963, 0.9558, 0.9558, ["002","007","009"]),
    (4,  0.9630, 0.9630, 0.9964, 0.9639, 0.9643, ["006","004","005"]),
    (5,  0.9630, 0.9630, 0.9964, 0.9639, 0.9643, ["000","003","008"]),
    (6,  0.9652, 0.9652, 0.9964, 0.9656, 0.9662, ["002","007","009"]),
    (7,  0.9620, 0.9620, 0.9970, 0.9635, 0.9635, ["006","004","005"]),
    (8,  0.9587, 0.9587, 0.9981, 0.9602, 0.9602, ["000","003","008"]),
    (9,  0.9530, 0.9530, 0.9975, 0.9545, 0.9545, ["002","007","009"]),
    (10, 0.9478, 0.9478, 0.9962, 0.9507, 0.9497, ["006","004","005"]),
]

CLIENT_REP_INIT = 0.500
CLIENT_REP_FINAL = {
    "000": 0.465, "001": 0.452, "002": 0.469,
    "003": 0.465, "004": 0.467, "005": 0.470,
    "006": 0.467, "007": 0.465, "008": 0.465,
    "009": 0.465,
}
ALL_CLIENTS = ["000","001","002","003","004","005","006","007","008","009"]

def _build_rep_history():
    history = {}
    for cid in ALL_CLIENTS:
        history[cid] = {}
        history[cid][0] = CLIENT_REP_INIT
        final = CLIENT_REP_FINAL[cid]
        for rnd in range(1, 11):
            history[cid][rnd] = round(
                CLIENT_REP_INIT + (final - CLIENT_REP_INIT) * rnd / 10, 4)
    return history

REP_HISTORY = _build_rep_history()

CLIENT_ROUNDS = {
    "000": [2,5,8], "001": [],      "002": [3,6,9],
    "003": [2,5,8], "004": [1,4,7,10], "005": [1,4,7,10],
    "006": [1,4,7,10], "007": [3,6,9], "008": [2,5,8],
    "009": [3,6,9],
}

# ── Mid-light slate palette ────────────────────────────────────
# UI surfaces — charcoal-slate, not full dark
C_BG      = "#2B2D3E"   # main background
C_PANEL   = "#353748"   # panel / sidebar
C_CARD    = "#404360"   # card surfaces
C_CARD2   = "#3A3D56"   # alternate card
C_TEXT    = "#E8EAF0"   # primary text  (near-white on slate)
C_TEXT2   = "#B0B8D0"   # secondary text
C_ACCENT  = "#5B8DD9"   # accent blue (buttons)
C_GREEN   = "#3DDC84"   # success green — brighter for slate bg
C_GOLD    = "#F5C842"   # gold
C_REAL    = "#44FF66"
C_FAKE    = "#FF5555"
C_UNC     = "#FFD700"
C_NEUTRAL = "#C0C8DC"

# Canvas colours stay dark — animation reads better on dark bg
CV_BG       = "#0D0D1F"
CV_NODE_C   = "#2A3060"
CV_NODE_S   = "#1A4080"
CV_SEL      = "#00B4D8"
CV_TRAIN    = "#4EA8F0"
CV_ARROW_UP = "#F07850"
CV_ARROW_DN = "#44CC66"
CV_GOLD     = "#F5C842"
CV_TEXT     = "#DDEEFF"

COLOR_REAL      = (136, 255,  68)
COLOR_FAKE      = ( 68,  68, 255)
COLOR_UNCERTAIN = (  0, 255, 255)
COLOR_NO_FACE   = (  0, 200, 255)

INPUT_SIZE     = 260
FACE_PADDING   = 0.20
HISTORY_SIZE   = 8
REAL_THRESHOLD   = 0.67   # legacy, kept for reference/back-compat
FAKE_THRESHOLD   = 0.45   # legacy, kept for reference/back-compat
DECISION_THRESHOLD = (REAL_THRESHOLD + FAKE_THRESHOLD) / 2  # 0.56 — binary REAL/FAKE cut
FRAME_SKIP     = 4
DISPLAY_MAX_W  = 880
DISPLAY_MAX_H  = 500
DETECT_W       = 640
DETECT_H       = 360
SPEED_OPTIONS  = [0.25, 0.5, 1.0, 2.0, 4.0]
VIDEO_EXTS     = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v"),
    ("All files", "*.*"),
)

T_SELECT    = 1.0
T_TRAIN     = 1.8
T_SEND      = 1.0
T_VALIDATE  = 0.9
T_AGGREGATE = 1.0
T_REP       = 0.9
T_AUTO_GAP  = 0.6

STEPS_DEF = [
    ("select",    "1  Select"),
    ("train",     "2  Train"),
    ("send",      "3  Send"),
    ("validate",  "4  Validate"),
    ("aggregate", "5  Aggregate"),
    ("rep",       "6  Reputation"),
]

STEP_COLORS = {
    "select":    "#F5C842",
    "train":     "#4EA8F0",
    "send":      "#F07850",
    "validate":  "#B79BFF",
    "aggregate": "#3DDC84",
    "rep":       "#FF9A7A",
}

# Per-step infographic card content
STEP_CARDS = {
    "select": {
        "icon":     "🎯",
        "headline": "Smart Client Selection",
        "why":      "Not all devices are equal. The server picks the 3 best-scoring "
                    "clients using a formula that considers past accuracy, participation "
                    "history, and reputation — so only high-quality contributors train each round.",
        "detail":   "Score = V_i  ×  H_i  ×  R_i\n2-round cooldown prevents repeat selection",
    },
    "train": {
        "icon":     "🔒",
        "headline": "Private Local Training",
        "why":      "Each selected device trains the model on its own private data for "
                    "5 full epochs. The raw data never leaves the device — only the "
                    "resulting model changes will be shared.",
        "detail":   "5 epochs · batch 32 · 920 images/client\nData stays on device at all times",
    },
    "send": {
        "icon":     "📤",
        "headline": "Model Updates Only",
        "why":      "Instead of sharing private images, each client sends only the "
                    "changes it made to the model weights. This is what makes Federated "
                    "Learning privacy-preserving — no raw data ever travels to the server.",
        "detail":   "Weight delta sent → not images\nPrivacy preserved across all clients",
    },
    "validate": {
        "icon":     "🛡️",
        "headline": "Update Validation Gate",
        "why":      "Before merging anything, the server checks each update for safety. "
                    "An L2 norm bound catches unusually large updates, and a gain test "
                    "rejects updates that would hurt the global model. Bad updates are "
                    "discarded — not merged.",
        "detail":   "L2 norm check  +  gain test\nThis round: 3 / 3 updates accepted ✓",
    },
    "aggregate": {
        "icon":     "⚖️",
        "headline": "Weighted Aggregation",
        "why":      "Accepted updates are blended together — but not equally. "
                    "Clients that scored higher in selection get more influence over "
                    "the new global model. Higher quality contribution = higher weight.",
        "detail":   "Weights ∝ contribution scores\nGlobal model updated for all clients",
    },
    "rep": {
        "icon":     "📊",
        "headline": "Reputation Ledger Update",
        "why":      "Every client's reputation score is adjusted after each round. "
                    "Clients that submitted valid updates earn a small reward. "
                    "All scores decay slightly each round to keep the system responsive "
                    "to recent behaviour rather than distant history.",
        "detail":   "Valid update → +reward\nAll clients → ×0.99 decay per round",
    },
}

# Chart Y-axis ranges — tight around actual data
_ACC_VALS = [rd[1] for rd in ROUND_DATA]
_AUC_VALS = [rd[3] for rd in ROUND_DATA]
CHART_ACC_LO = round(min(_ACC_VALS) - 0.003, 4)
CHART_ACC_HI = round(max(_ACC_VALS) + 0.003, 4)
CHART_AUC_LO = round(min(_AUC_VALS) - 0.001, 4)
CHART_AUC_HI = round(max(_AUC_VALS) + 0.001, 4)


# ══════════════════════════════════════════════════════════════
# Model helpers
# ══════════════════════════════════════════════════════════════
def load_tflite_model(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]

def resolve_model_path(given):
    default = "effnet_global_fl_final_quantised.tflite"
    here    = Path(__file__).parent
    if given:
        p = Path(given)
        if p.exists(): return str(p)
    auto = here / default
    if auto.exists(): return str(auto)
    root_tmp = tk.Tk(); root_tmp.withdraw()
    path = filedialog.askopenfilename(
        title="Locate TFLite model",
        filetypes=[("TFLite model","*.tflite"),("All files","*.*")],
        initialdir=str(here))
    root_tmp.destroy()
    if not path: sys.exit(1)
    return path

def resolve_plain_model_path(given=None):
    """Resolves the baseline Plain FedAvg model, mirroring resolve_model_path
    but defaulting to the plain-FedAvg filename instead of the enhanced one."""
    default = "effnet_plain_fedavg_full_integer.tflite"
    here    = Path(__file__).parent
    if given:
        p = Path(given)
        if p.exists(): return str(p)
    auto = here / default
    if auto.exists(): return str(auto)
    root_tmp = tk.Tk(); root_tmp.withdraw()
    path = filedialog.askopenfilename(
        title="Locate the Plain FedAvg TFLite model",
        filetypes=[("TFLite model","*.tflite"),("All files","*.*")],
        initialdir=str(here))
    root_tmp.destroy()
    if not path: sys.exit(1)
    return path

def preprocess_face(face_bgr, dtype, scale, zp):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = tf.cast(face_rgb, tf.float32)
    img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    if dtype in (np.uint8, np.int8):
        img = preprocess_input(img)
        if scale: img = img / scale + zp
        lo, hi = (0, 255) if dtype == np.uint8 else (-128, 127)
        img = tf.clip_by_value(img, lo, hi)
        return np.expand_dims(img.numpy().astype(dtype), 0)
    img = preprocess_input(img)
    return np.expand_dims(img.numpy().astype(np.float32), 0)

def classify(s):
    """Binary classification — REAL vs FAKE only.
    Decision boundary sits at the midpoint of the old REAL/FAKE thresholds,
    so the former 'uncertain' band is folded into whichever side it's closer to."""
    if s >= DECISION_THRESHOLD:
        return "REAL", COLOR_REAL, C_REAL, s * 100
    return "FAKE", COLOR_FAKE, C_FAKE, (1 - s) * 100


# ══════════════════════════════════════════════════════════════
# Network Canvas
# ══════════════════════════════════════════════════════════════
class NetworkCanvas:
    R_CLIENT = 38
    R_SERVER = 56

    def __init__(self, parent, width, height):
        self.w = width
        self.h = height
        self._root = parent.winfo_toplevel()

        self._outer = tk.Frame(parent, bg=CV_BG)
        self._outer.pack(fill=tk.BOTH, expand=True)

        # ── Breadcrumb strip ─────────────────────────────────
        self._crumb_frame = tk.Frame(self._outer, bg="#14142A", pady=4)
        self._crumb_frame.pack(fill=tk.X)
        self._crumb_labels = {}
        for i, (key, label) in enumerate(STEPS_DEF):
            if i > 0:
                tk.Label(self._crumb_frame, text="›", bg="#14142A",
                         fg="#44446A", font=("Helvetica", 11)).pack(side=tk.LEFT)
            lbl = tk.Label(self._crumb_frame, text=label,
                           bg="#14142A", fg="#44446A",
                           font=("Helvetica", 9, "bold"), padx=8, pady=2)
            lbl.pack(side=tk.LEFT)
            self._crumb_labels[key] = lbl

        # ── Canvas ───────────────────────────────────────────
        self.canvas = tk.Canvas(self._outer, width=width, height=height,
                                bg=CV_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── Step status bar ──────────────────────────────────
        self._status_bar = tk.Frame(self._outer, bg="#14142A", pady=4)
        self._status_bar.pack(fill=tk.X)
        self._status_lbl = tk.Label(
            self._status_bar, text="Ready — click 'Next Round' to begin",
            bg="#14142A", fg="#8888BB",
            font=("Helvetica", 9, "italic"))
        self._status_lbl.pack(padx=10, anchor="w")

        self.cx = width  // 2
        self.cy = height // 2 + 20

        self._client_pos = {}
        top_ids    = ["000","001","002","003","004"]
        bottom_ids = ["005","006","007","008","009"]
        rx = int(width  * 0.40)
        ry = int(height * 0.38)

        for i, cid in enumerate(top_ids):
            angle = math.pi + (math.pi / (len(top_ids)+1)) * (i+1)
            self._client_pos[cid] = (
                self.cx + int(rx * math.cos(angle)),
                self.cy + int(ry * math.sin(angle)))

        for i, cid in enumerate(bottom_ids):
            angle = (math.pi / (len(bottom_ids)+1)) * (i+1)
            self._client_pos[cid] = (
                self.cx + int(rx * math.cos(angle)),
                self.cy + int(ry * math.sin(angle)))

        self._node_ovals   = {}
        self._node_labels  = {}
        self._rep_labels   = {}
        self._epoch_labels = {}
        self._server_oval  = None
        self._server_lbl   = None
        self._server_sub   = None
        self._server_rnd   = None
        self._anim_items   = []
        self._anim_after   = None

        self._draw_static()

    def _draw_static(self):
        c = self.canvas
        for x in range(0, self.w, 70):
            c.create_line(x, 0, x, self.h, fill="#111128", width=1)
        for y in range(0, self.h, 70):
            c.create_line(0, y, self.w, y, fill="#111128", width=1)

        for cid, (x, y) in self._client_pos.items():
            c.create_line(x, y, self.cx, self.cy,
                          fill="#1C1C3A", width=1, dash=(3,8),
                          tags="idle_line")
        c.tag_lower("idle_line")

        # Server
        r = self.R_SERVER
        self._server_oval = c.create_oval(
            self.cx-r, self.cy-r, self.cx+r, self.cy+r,
            fill=CV_NODE_S, outline="#4A80C0", width=2)
        self._server_lbl = c.create_text(
            self.cx, self.cy-14, text="SERVER",
            fill=CV_TEXT, font=("Helvetica", 11, "bold"))
        self._server_sub = c.create_text(
            self.cx, self.cy+4, text="Global Model",
            fill="#8888BB", font=("Helvetica", 8))
        self._server_rnd = c.create_text(
            self.cx, self.cy+20, text="",
            fill=CV_GOLD, font=("Helvetica", 8, "bold"))

        # Client nodes
        for cid, (x, y) in self._client_pos.items():
            r = self.R_CLIENT
            ov  = c.create_oval(x-r, y-r, x+r, y+r,
                                fill=CV_NODE_C, outline="#4A4A7A", width=1)
            lbl = c.create_text(x, y-10, text=f"C{cid}",
                                fill=CV_TEXT, font=("Helvetica", 9, "bold"))
            rep = c.create_text(x, y+6,
                                text=f"R:{CLIENT_REP_FINAL[cid]:.3f}",
                                fill="#9999BB", font=("Helvetica", 7))
            epoch = c.create_text(x, y+20, text="",
                                  fill=CV_TRAIN, font=("Helvetica", 7, "bold"))
            self._node_ovals[cid]   = ov
            self._node_labels[cid]  = lbl
            self._rep_labels[cid]   = rep
            self._epoch_labels[cid] = epoch

    # ── helpers ───────────────────────────────────────────────
    def _clear_anim(self):
        for item in self._anim_items:
            try: self.canvas.delete(item)
            except: pass
        self._anim_items.clear()

    def _cancel_anim(self):
        if self._anim_after:
            try: self._root.after_cancel(self._anim_after)
            except: pass
            self._anim_after = None

    def set_status(self, text, color="#9090CC"):
        self._status_lbl.config(text=text, fg=color)

    # Dim solid tint for breadcrumb active-step background
    _CRUMB_BG = {
        "select":    "#302810",
        "train":     "#102030",
        "send":      "#301810",
        "validate":  "#201030",
        "aggregate": "#0A2018",
        "rep":       "#301810",
    }

    def set_breadcrumb(self, active_key):
        for key, lbl in self._crumb_labels.items():
            if key == active_key:
                lbl.config(fg=STEP_COLORS[key],
                           bg=self._CRUMB_BG.get(key, "#1A1A30"))
            else:
                lbl.config(fg="#44446A", bg="#14142A")

    def clear_breadcrumb(self):
        for lbl in self._crumb_labels.values():
            lbl.config(fg="#44446A", bg="#14142A")

    def set_round_badge(self, rnd):
        text = f"Round {rnd} / 10" if rnd > 0 else ""
        self.canvas.itemconfig(self._server_rnd, text=text)

    def reset(self):
        self._cancel_anim()
        self._clear_anim()
        c = self.canvas
        for cid, ov in self._node_ovals.items():
            c.itemconfig(ov, fill=CV_NODE_C, outline="#4A4A7A", width=1)
            c.itemconfig(self._node_labels[cid], fill=CV_TEXT)
            c.itemconfig(self._rep_labels[cid], fill="#9999BB")
            c.itemconfig(self._epoch_labels[cid], text="")
        c.itemconfig(self._server_oval, fill=CV_NODE_S,
                     outline="#4A80C0", width=2)
        c.itemconfig(self._server_sub, text="Global Model", fill="#8888BB")
        c.itemconfig(self._server_rnd, text="")
        self.clear_breadcrumb()
        self.set_status("Ready — click 'Next Round' to begin")

    def round_done_reset(self):
        c = self.canvas
        to_del = []
        for item in self._anim_items:
            try:
                if c.type(item) in ("line","oval","rectangle"):
                    to_del.append(item)
                elif c.type(item) == "text":
                    txt = c.itemcget(item,"text") or ""
                    if any(k in txt for k in [
                        "score=","Training","private","Model\nUpdate","Selecting",
                        "Aggregating","Receiving","Validating","w=",
                        "reward","decay","inactive","OK","weight","epoch",
                        "All updates","C0","Epoch","Δ","never"]):
                        to_del.append(item)
            except: pass
        for item in to_del:
            try: c.delete(item)
            except: pass
        self._anim_items = [i for i in self._anim_items if i not in to_del]
        for cid in ALL_CLIENTS:
            try: c.itemconfig(self._epoch_labels[cid], text="")
            except: pass

    # ── Step 1: Select ────────────────────────────────────────
    def animate_select(self, selected, rnd, on_done):
        c = self.canvas
        self._clear_anim()
        self.set_breadcrumb("select")
        self.set_status("Scoring all clients… selecting top 3 by V_i × H_i × R_i",
                        STEP_COLORS["select"])
        self.set_round_badge(rnd)

        for cid, ov in self._node_ovals.items():
            if cid in selected:
                c.itemconfig(ov, fill=CV_SEL, outline=CV_GOLD, width=3)
                c.itemconfig(self._node_labels[cid], fill="#001A22")
            else:
                c.itemconfig(ov, fill="#181830", outline="#2A2A4A", width=1)
                c.itemconfig(self._node_labels[cid], fill="#383860")

        scores = [0.38, 0.35, 0.27]
        lines  = []
        for i, cid in enumerate(selected):
            x, y = self._client_pos[cid]
            lid  = c.create_line(x, y, x, y, fill=CV_GOLD, width=2,
                                 dash=(6,3), arrow=tk.LAST,
                                 arrowshape=(10,12,4))
            sid  = c.create_text((x+self.cx)//2, (y+self.cy)//2,
                                 text=f"score={scores[i]:.2f}",
                                 fill=CV_GOLD, font=("Helvetica",8,"bold"),
                                 state=tk.HIDDEN)
            self._anim_items += [lid, sid]
            lines.append((lid, sid, x, y))

        slbl = c.create_text(
            self.cx, self.cy + self.R_SERVER + 18,
            text=f"Selected: C{selected[0]}  C{selected[1]}  C{selected[2]}",
            fill=CV_GOLD, font=("Helvetica",9,"bold"))
        self._anim_items.append(slbl)

        steps = 22
        def draw(step=0):
            frac = step / steps
            for lid, sid, x, y in lines:
                nx = x + (self.cx-x)*frac
                ny = y + (self.cy-y)*frac
                c.coords(lid, x, y, nx, ny)
            if step < steps:
                self._anim_after = self._root.after(
                    int(T_SELECT*1000/steps), lambda s=step+1: draw(s))
            else:
                for _, sid, *_ in lines:
                    c.itemconfig(sid, state=tk.NORMAL)
                self._anim_after = self._root.after(350, on_done)
        draw()

    # ── Step 2: Train ─────────────────────────────────────────
    def animate_train(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("train")
        self.set_status("Clients training on private local data — 5 epochs each",
                        STEP_COLORS["train"])
        c.itemconfig(self._server_sub, text="Awaiting\nupdates…", fill="#8888BB")

        bars   = {}
        epochs = 5
        for cid in selected:
            x, y = self._client_pos[cid]
            r    = self.R_CLIENT
            lock = c.create_text(x, y-r-12, text="🔒 Private Data",
                                 fill="#AAAACC", font=("Helvetica",7))
            bg   = c.create_rectangle(x-r+5, y+16, x+r-5, y+26,
                                      fill="#111130", outline="#333355")
            fill = c.create_rectangle(x-r+5, y+16, x-r+5, y+26,
                                      fill=CV_TRAIN, outline="")
            self._anim_items += [lock, bg, fill]
            c.itemconfig(self._epoch_labels[cid], text="Epoch 1 / 5")
            bars[cid] = (fill, x-r+5, y+16, x+r-5, y+26)

        total_steps = epochs * 8
        def grow(step=0):
            frac = step / total_steps
            ep   = min(int(step / 8) + 1, epochs)
            for cid, (fill, bx1, by1, bx2, by2) in bars.items():
                nx = bx1 + (bx2-bx1)*frac
                c.coords(fill, bx1, by1, nx, by2)
                c.itemconfig(self._epoch_labels[cid],
                             text=f"Epoch {ep} / {epochs}")
            if step < total_steps:
                self._anim_after = self._root.after(
                    int(T_TRAIN*1000/total_steps), lambda s=step+1: grow(s))
            else:
                for cid, (fill, *_) in bars.items():
                    c.itemconfig(fill, fill=C_GREEN)
                    c.itemconfig(self._epoch_labels[cid],
                                 text="Done ✓", fill=C_GREEN)
                self._anim_after = self._root.after(350, on_done)
        grow()

    # ── Step 3: Send ──────────────────────────────────────────
    def animate_send(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("send")
        self.set_status("Sending model weight updates to server — no raw data transmitted",
                        STEP_COLORS["send"])
        c.itemconfig(self._server_sub, text="Receiving\nupdates…", fill=CV_ARROW_UP)

        arrows = []
        for cid in selected:
            x, y = self._client_pos[cid]
            dot  = c.create_oval(x-7, y-7, x+7, y+7,
                                 fill=CV_ARROW_UP, outline="")
            lbl  = c.create_text(x, y-18, text="Model\nUpdate",
                                 fill=CV_ARROW_UP, font=("Helvetica",7,"bold"))
            self._anim_items += [dot, lbl]
            arrows.append((dot, lbl, x, y))

        steps = 26
        def travel(step=0):
            frac = step / steps
            ease = frac*frac*(3-2*frac)
            for dot, lbl, x1, y1 in arrows:
                nx = x1 + (self.cx-x1)*ease
                ny = y1 + (self.cy-y1)*ease
                c.coords(dot, nx-7, ny-7, nx+7, ny+7)
                c.coords(lbl, nx, ny-18)
            if step < steps:
                self._anim_after = self._root.after(
                    int(T_SEND*1000/steps), lambda s=step+1: travel(s))
            else:
                for dot, lbl, *_ in arrows:
                    c.delete(dot); c.delete(lbl)
                on_done()
        travel()

    # ── Step 4: Validate ──────────────────────────────────────
    def animate_validate(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("validate")
        self.set_status("Server checking each update: L2 norm bound + gain test",
                        STEP_COLORS["validate"])
        c.itemconfig(self._server_sub, text="Validating\nupdates…", fill="#B79BFF")
        c.itemconfig(self._server_oval, outline="#B79BFF", width=3)

        checks  = []
        offsets = [(-62,-28),(0,-58),(62,-28)]
        for i, cid in enumerate(selected):
            ox, oy = offsets[i]
            bx, by = self.cx+ox, self.cy+oy
            badge  = c.create_oval(bx-16, by-16, bx+16, by+16,
                                   fill="#003A00", outline=C_GREEN, width=2,
                                   state=tk.HIDDEN)
            chk    = c.create_text(bx, by, text="✓", fill=C_GREEN,
                                   font=("Helvetica",13,"bold"), state=tk.HIDDEN)
            clbl   = c.create_text(bx, by+26, text=f"C{cid} OK",
                                   fill=C_GREEN, font=("Helvetica",7),
                                   state=tk.HIDDEN)
            self._anim_items += [badge, chk, clbl]
            checks.append((badge, chk, clbl))

        def show(i=0):
            if i < len(checks):
                for item in checks[i]:
                    c.itemconfig(item, state=tk.NORMAL)
                self._anim_after = self._root.after(
                    int(T_VALIDATE*1000/len(checks)), lambda: show(i+1))
            else:
                c.itemconfig(self._server_sub,
                             text="All updates\naccepted ✓", fill=C_GREEN)
                self._anim_after = self._root.after(400, on_done)
        show()

    # ── Step 5: Aggregate ─────────────────────────────────────
    def animate_aggregate(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("aggregate")
        self.set_status("Merging updates with adaptive weights — higher score = more influence",
                        STEP_COLORS["aggregate"])
        c.itemconfig(self._server_sub, text="Aggregating\nweights…", fill=CV_GOLD)

        weights = ["w=0.38", "w=0.35", "w=0.27"]
        for i, cid in enumerate(selected):
            x, y = self._client_pos[cid]
            al   = c.create_line(x, y, self.cx, self.cy, fill=C_GREEN,
                                 width=2, arrow=tk.LAST, arrowshape=(10,12,4))
            wl   = c.create_text((x+self.cx)//2, (y+self.cy)//2,
                                 text=weights[i], fill=C_GREEN,
                                 font=("Helvetica",8,"bold"))
            self._anim_items += [al, wl]

        pulses = [CV_GOLD, "#B08010", CV_GOLD, "#D0A020", CV_GOLD]
        def pulse(i=0):
            if i < len(pulses):
                c.itemconfig(self._server_oval, fill=pulses[i], width=3)
                self._anim_after = self._root.after(
                    int(T_AGGREGATE*1000/len(pulses)), lambda: pulse(i+1))
            else:
                c.itemconfig(self._server_oval, fill=CV_NODE_S,
                             outline=CV_GOLD, width=3)
                c.itemconfig(self._server_sub,
                             text="Global Model\nUpdated ✓", fill=CV_GOLD)
                for cid2 in ALL_CLIENTS:
                    x2, y2 = self._client_pos[cid2]
                    bl = c.create_line(self.cx, self.cy, x2, y2,
                                      fill=CV_ARROW_DN, width=1, dash=(4,5))
                    self._anim_items.append(bl)
                self._anim_after = self._root.after(500, on_done)
        pulse()

    # ── Step 6: Reputation (with Δ pop-ups) ───────────────────
    def animate_reputation(self, selected, rnd, on_done):
        c = self.canvas
        self.set_breadcrumb("rep")
        self.set_status("Updating reputation ledger — valid clients rewarded, all scores decay ×0.99",
                        STEP_COLORS["rep"])

        badges = []
        delta_labels = []
        for cid in ALL_CLIENTS:
            x, y  = self._client_pos[cid]

            # Compute numeric delta for pop-up
            rep_now  = REP_HISTORY[cid].get(rnd, CLIENT_REP_FINAL[cid])
            rep_prev = REP_HISTORY[cid].get(max(rnd-1, 0), CLIENT_REP_INIT)
            delta    = rep_now - rep_prev
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"

            if cid in selected:
                txt   = "+reward"
                color = C_GREEN
                d_col = "#55FF88"
                c.itemconfig(self._node_ovals[cid],
                             fill=CV_SEL, outline=C_GREEN, width=2)
            elif cid == "001":
                txt   = "never selected"
                color = "#777799"
                d_col = "#888888"
                c.itemconfig(self._node_ovals[cid],
                             fill=CV_NODE_C, outline="#4A4A7A", width=1)
            else:
                txt   = "×0.99 decay"
                color = "#888888"
                d_col = "#AAAAAA"
                c.itemconfig(self._node_ovals[cid],
                             fill=CV_NODE_C, outline="#4A4A7A", width=1)

            bid = c.create_text(x, y-self.R_CLIENT-14, text=txt,
                                fill=color, font=("Helvetica",7,"bold"))
            # Δ value shown just above the status text
            did = c.create_text(x, y-self.R_CLIENT-26, text=f"Δ {delta_str}",
                                fill=d_col, font=("Helvetica",7,"bold"))
            self._anim_items += [bid, did]
            badges.append(bid)
            delta_labels.append(did)

            rep_val = REP_HISTORY[cid].get(rnd, CLIENT_REP_FINAL[cid])
            c.itemconfig(self._rep_labels[cid],
                         text=f"R:{rep_val:.3f}", fill="#BBBBDD")

        def fade():
            for b in badges + delta_labels:
                try: c.delete(b)
                except: pass
            on_done()
        self._anim_after = self._root.after(int(T_REP*1000), fade)

    # ── Round 6 special ───────────────────────────────────────
    def show_best_checkpoint(self):
        c = self.canvas
        c.itemconfig(self._server_oval, fill="#5A4000", outline=CV_GOLD, width=4)
        c.itemconfig(self._server_lbl, fill=CV_GOLD)
        c.itemconfig(self._server_sub, text="★ Best\nCheckpoint!", fill=CV_GOLD)
        glow = c.create_oval(
            self.cx-self.R_SERVER-14, self.cy-self.R_SERVER-14,
            self.cx+self.R_SERVER+14, self.cy+self.R_SERVER+14,
            outline=CV_GOLD, width=3, fill="")
        self._anim_items.append(glow)
        self.set_status("★  Round 6 — Best checkpoint saved! Peak: Acc 96.52%  F1 96.52%  AUC 0.9964",
                        CV_GOLD)


# ══════════════════════════════════════════════════════════════
# Reputation Ledger Strip — two columns of 5 clients
# ══════════════════════════════════════════════════════════════
class LedgerStrip:
    """
    Two side-by-side mini-tables of 5 clients each.
    Fits in ~105 px without clipping. Prev/Next buttons navigate rounds.
    """
    # Three columns: C000-C003 | C004-C006 | C007-C009
    _COL1 = ["000","001","002","003"]
    _COL2 = ["004","005","006"]
    _COL3 = ["007","008","009"]

    def __init__(self, parent):
        self._current_rnd = 0
        self._max_rnd     = 0

        # Outer container — auto height to fit 3 cols of 4/3/3 rows
        outer = tk.Frame(parent, bg=C_PANEL, pady=4)
        outer.pack(fill=tk.X)

        # ── Header ───────────────────────────────────────────
        hdr = tk.Frame(outer, bg=C_PANEL)
        hdr.pack(fill=tk.X, padx=8)

        tk.Label(hdr, text="REPUTATION LEDGER",
                 bg=C_PANEL, fg=C_GOLD,
                 font=("Helvetica",8,"bold")).pack(side=tk.LEFT)

        self._nav_lbl = tk.Label(hdr, text="Round 0  (Pre-training)",
                                 bg=C_PANEL, fg=C_TEXT2,
                                 font=("Helvetica",8))
        self._nav_lbl.pack(side=tk.LEFT, padx=(10,0))

        btn_cfg = dict(bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                       relief=tk.FLAT, padx=6, pady=1,
                       activebackground=C_ACCENT, activeforeground="white")
        self._btn_next = tk.Button(hdr, text="▶", command=self._go_next, **btn_cfg)
        self._btn_next.pack(side=tk.RIGHT, padx=(2,0))
        self._btn_prev = tk.Button(hdr, text="◀", command=self._go_prev, **btn_cfg)
        self._btn_prev.pack(side=tk.RIGHT, padx=(0,2))

        # ── Three-column table area ───────────────────────────
        tables = tk.Frame(outer, bg=C_PANEL)
        tables.pack(fill=tk.X, padx=8, pady=(3,0))

        self._row_vars   = {}   # cid → [StringVar×5, label_ref]
        col_hdrs = ["Client","Sel. Rnds","Status","Rep","Δ Rep"]
        col_w    = [6, 11, 11, 7, 7]

        for side_idx, group in enumerate([self._COL1, self._COL2, self._COL3]):
            tbl = tk.Frame(tables, bg=C_PANEL)
            tbl.pack(side=tk.LEFT, fill=tk.X, expand=True,
                     padx=(0, 8 if side_idx < 2 else 0))

            # Column headers
            for j, (h, w) in enumerate(zip(col_hdrs, col_w)):
                tk.Label(tbl, text=h, bg=C_PANEL, fg="#888AAA",
                         font=("Helvetica",7,"bold"),
                         width=w, anchor="w").grid(row=0, column=j, padx=(0,3), pady=(0,1))

            # Data rows
            for i, cid in enumerate(group):
                row_vars = [tk.StringVar() for _ in range(5)]
                colors   = [C_TEXT, C_TEXT2, C_TEXT, C_GREEN, C_TEXT]
                self._row_vars[cid] = row_vars

                for j, (var, w, fg) in enumerate(zip(row_vars, col_w, colors)):
                    lbl = tk.Label(tbl, textvariable=var,
                                   bg=C_PANEL, fg=fg,
                                   font=("Helvetica",7),
                                   width=w, anchor="w")
                    lbl.grid(row=i+1, column=j, padx=(0,3), pady=0)
                    if j == 2:
                        row_vars.append(lbl)   # store label ref at [5]

        self._refresh(0)

    def _refresh(self, rnd):
        self._current_rnd = rnd
        suffix = ""
        if rnd == 0: suffix = "  (Pre-training)"
        if rnd == 6: suffix = "  ★ Best"
        self._nav_lbl.config(text=f"Round {rnd}{suffix}")

        for cid in ALL_CLIENTS:
            rounds_sel = CLIENT_ROUNDS[cid]
            rep_now    = REP_HISTORY[cid].get(rnd, CLIENT_REP_FINAL[cid])
            rep_prev   = REP_HISTORY[cid].get(max(rnd-1,0), CLIENT_REP_INIT)
            delta      = rep_now - rep_prev

            if rnd == 0:
                status  = "—"
                s_color = "#888888"
            elif rnd in rounds_sel:
                status  = "Selected ✓"
                s_color = C_GREEN
            elif cid == "001":
                status  = "Never sel."
                s_color = "#777799"
            else:
                status  = "Idle (decay)"
                s_color = C_TEXT2

            delta_str  = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            rounds_str = ", ".join(str(r) for r in rounds_sel) if rounds_sel else "None"

            vars_ = self._row_vars[cid]
            vars_[0].set(f"C{cid}")
            vars_[1].set(rounds_str)
            vars_[2].set(status)
            vars_[3].set(f"{rep_now:.4f}")
            vars_[4].set(delta_str)
            if len(vars_) > 5:
                vars_[5].config(fg=s_color)

        self._btn_prev.config(state=tk.NORMAL if rnd > 0 else tk.DISABLED)
        self._btn_next.config(state=tk.NORMAL if rnd < self._max_rnd else tk.DISABLED)

    def advance_to(self, rnd):
        self._max_rnd = rnd
        self._refresh(rnd)

    def reset(self):
        self._max_rnd = 0
        self._refresh(0)

    def _go_prev(self):
        if self._current_rnd > 0:
            self._refresh(self._current_rnd - 1)

    def _go_next(self):
        if self._current_rnd < self._max_rnd:
            self._refresh(self._current_rnd + 1)


# ══════════════════════════════════════════════════════════════
# Step Illustration Card — now lives in the centre column
# ══════════════════════════════════════════════════════════════
class StepCard:
    """
    Infographic card: icon · headline · plain-English why · technical detail.
    Wider now (sits in centre column beside canvas).
    """
    def __init__(self, parent):
        self._frame = tk.Frame(parent, bg=C_CARD2,
                               relief=tk.FLAT, pady=10, padx=14)
        self._frame.pack(fill=tk.BOTH, expand=True)

        top = tk.Frame(self._frame, bg=C_CARD2)
        top.pack(fill=tk.X, pady=(0,4))

        self._icon_lbl = tk.Label(top, text="", bg=C_CARD2, fg=C_TEXT,
                                   font=("Helvetica",22))
        self._icon_lbl.pack(side=tk.LEFT, padx=(0,10))

        right_hdr = tk.Frame(top, bg=C_CARD2)
        right_hdr.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._headline = tk.Label(right_hdr, text="", bg=C_CARD2, fg=C_TEXT,
                                   font=("Helvetica",13,"bold"), anchor="w")
        self._headline.pack(fill=tk.X)

        self._step_tag = tk.Label(right_hdr, text="",
                                   bg=C_CARD2, fg=C_TEXT2,
                                   font=("Helvetica",9), anchor="w")
        self._step_tag.pack(fill=tk.X)

        self._color_bar = tk.Frame(self._frame, bg="#444466", height=2)
        self._color_bar.pack(fill=tk.X, pady=(2,8))

        self._why_lbl = tk.Label(self._frame, text="",
                                  bg=C_CARD2, fg=C_TEXT,
                                  font=("Helvetica",10),
                                  wraplength=260, justify="left", anchor="nw")
        self._why_lbl.pack(fill=tk.X)

        self._detail_lbl = tk.Label(self._frame, text="",
                                     bg=C_CARD, fg=C_TEXT2,
                                     font=("Helvetica",9,"italic"),
                                     wraplength=260, justify="left",
                                     anchor="w", pady=5, padx=8)
        self._detail_lbl.pack(fill=tk.X, pady=(8,0))

    def show(self, step_key):
        info  = STEP_CARDS.get(step_key, {})
        color = STEP_COLORS.get(step_key, "#888888")
        # find step number
        idx   = next((i+1 for i,(k,_) in enumerate(STEPS_DEF) if k==step_key), "")
        self._icon_lbl.config(text=info.get("icon",""))
        self._headline.config(text=info.get("headline",""), fg=color)
        self._step_tag.config(text=f"Step {idx} of 6")
        self._color_bar.config(bg=color)
        self._why_lbl.config(text=info.get("why",""))
        self._detail_lbl.config(text=info.get("detail",""))

    def clear(self):
        for lbl in (self._icon_lbl, self._headline, self._step_tag,
                    self._why_lbl, self._detail_lbl):
            lbl.config(text="")
        self._color_bar.config(bg="#444466")


# ══════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PLAIN FEDAVG DATA  (from fl_plain_run-2.log — real training run)
# round, acc, f1, auc, mean_local_acc, selected(3), local_accs(3), seconds
# ══════════════════════════════════════════════════════════════
PLAIN_ROUND_DATA = [
    (0,  0.9478, 0.9475, 0.9882, None,   None,                        None,                          None),
    (1,  0.9486, 0.9493, 0.9880, 0.9658, ("000","004","002"), (0.9627,0.9752,0.9596), 1162),
    (2,  0.9464, 0.9461, 0.9868, 0.9633, ("007","005","001"), (0.9674,0.9534,0.9689),  764),
    (3,  None,   None,   None,   0.9705, ("007","009","000"), (0.9643,0.9783,0.9689),  786),
    (4,  0.9500, 0.9502, 0.9878, 0.9622, ("008","006","005"), (0.9581,0.9705,0.9581),  837),
    (5,  None,   None,   None,   0.9679, ("009","008","006"), (0.9783,0.9565,0.9689),  891),
    (6,  0.9486, 0.9491, 0.9878, 0.9710, ("006","002","000"), (0.9752,0.9643,0.9736),  870),
    (7,  None,   None,   None,   0.9731, ("001","009","006"), (0.9736,0.9798,0.9658), 1043),
    (8,  0.9442, 0.9458, 0.9865, 0.9829, ("002","001","009"), (0.9876,0.9736,0.9876),  997),
    (9,  None,   None,   None,   0.9741, ("008","007","002"), (0.9658,0.9798,0.9767), 1002),
    (10, 0.9478, 0.9489, 0.9871, 0.9752, ("006","008","004"), (0.9736,0.9705,0.9814), 1287),
]
PLAIN_BEST_ROUND = 8
PLAIN_BEST_METRICS = {"acc": 0.9442, "f1": 0.9458, "auc": 0.9865}
PLAIN_ALL_CLIENTS = [f"{i:03d}" for i in range(10)]

def _plain_fill_eval(data):
    """Carry the last known eval forward for rounds without a fresh EVAL line,
    so charts/metric panels never show a blank round."""
    out = []
    last = (data[0][1], data[0][2], data[0][3])
    for rd in data:
        acc, f1, auc = rd[1], rd[2], rd[3]
        if acc is None:
            acc, f1, auc = last
        else:
            last = (acc, f1, auc)
        out.append((rd[0], acc, f1, auc) + rd[4:])
    return out

PLAIN_ROUND_DATA_FILLED = _plain_fill_eval(PLAIN_ROUND_DATA)

PLAIN_ACC_VALS = [r[1] for r in PLAIN_ROUND_DATA_FILLED]
PLAIN_AUC_VALS = [r[3] for r in PLAIN_ROUND_DATA_FILLED]
CHART_PLAIN_ACC_LO = round(min(PLAIN_ACC_VALS) - 0.003, 4)
CHART_PLAIN_ACC_HI = round(max(PLAIN_ACC_VALS) + 0.003, 4)
CHART_PLAIN_AUC_LO = round(min(PLAIN_AUC_VALS) - 0.001, 4)
CHART_PLAIN_AUC_HI = round(max(PLAIN_AUC_VALS) + 0.001, 4)

PLAIN_STEPS_DEF = [
    ("select",  "1 Select"),
    ("train",   "2 Train"),
    ("send",    "3 Send"),
    ("average", "4 Average"),
]
PLAIN_STEP_COLORS = {
    "select":  "#8A8FA3",   # neutral grey-blue — no scoring involved
    "train":   "#4EA8F0",
    "send":    "#F07850",
    "average": "#7A7E96",   # muted vs enhanced's gold — plain mean, not weighted
}
PLAIN_STEP_CARDS = {
    "select": {
        "icon": "🎲",
        "headline": "Random Client Selection",
        "why": "Plain FedAvg has no scoring system. Each round the server picks "
               "3 of the 10 clients completely at random, with no regard for past "
               "accuracy, participation, or reliability.",
        "detail": "Uniform random draw · no reputation · no cooldown",
    },
    "train": {
        "icon": "💻",
        "headline": "Local Training",
        "why": "Each selected client trains the shared model on its own private "
               "data for 2 local epochs, batch size 8. As in the enhanced version, "
               "raw data never leaves the device.",
        "detail": "2 epochs · batch 8 · lr 1e-05 · 644 images/client",
    },
    "send": {
        "icon": "📤",
        "headline": "Model Updates Sent",
        "why": "Clients send their locally updated weights back to the server. "
               "There is no validation gate waiting on the other side — every "
               "update that arrives gets used.",
        "detail": "Full weights returned · no L2 / gain-test screening",
    },
    "average": {
        "icon": "⚖️",
        "headline": "Simple Averaging (FedAvg)",
        "why": "The server blends the 3 incoming updates with a plain, equal-weight "
               "mean. A client with a poor update has exactly the same influence as "
               "one with a great update — quality is not considered.",
        "detail": "Equal weights (1/3 each) · no reputation ledger · no decay",
    },
}


class SimpleNetworkCanvas:
    """Stripped-down animated pipeline for Plain FedAvg — visually consistent
    with NetworkCanvas (same node layout / colour language) but only 4 steps,
    no scores, no reputation numbers, no validation badges."""

    R_CLIENT = 34
    R_SERVER = 50

    def __init__(self, parent, width, height):
        self.w = width; self.h = height
        self.root = parent.winfo_toplevel()
        self.outer = tk.Frame(parent, bg=CV_BG)
        self.outer.pack(fill=tk.BOTH, expand=True)

        self.crumb_frame = tk.Frame(self.outer, bg="#14142A", pady=4)
        self.crumb_frame.pack(fill=tk.X)
        self.crumb_labels = {}
        for i, (key, label) in enumerate(PLAIN_STEPS_DEF):
            if i > 0:
                tk.Label(self.crumb_frame, text="›", bg="#14142A",
                         fg="#44446A", font=("Helvetica", 11)).pack(side=tk.LEFT)
            lbl = tk.Label(self.crumb_frame, text=label, bg="#14142A",
                            fg="#44446A", font=("Helvetica", 9, "bold"),
                            padx=8, pady=2)
            lbl.pack(side=tk.LEFT)
            self.crumb_labels[key] = lbl

        self.canvas = tk.Canvas(self.outer, width=width, height=height,
                                 bg=CV_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_bar = tk.Frame(self.outer, bg="#14142A", pady=4)
        self.status_bar.pack(fill=tk.X)
        self.status_lbl = tk.Label(
            self.status_bar, text="Ready — click Next Round to begin",
            bg="#14142A", fg="#8888BB", font=("Helvetica", 9, "italic"))
        self.status_lbl.pack(padx=10, anchor="w")

        self.cx = width // 2; self.cy = height // 2 - 20
        self.client_pos = {}
        top_ids = ["000","001","002","003","004"]
        bottom_ids = ["005","006","007","008","009"]
        rx, ry = int(width * 0.40), int(height * 0.38)
        for i, cid in enumerate(top_ids):
            angle = math.pi + (math.pi / (len(top_ids)-1)) * i
            self.client_pos[cid] = (self.cx + int(rx*math.cos(angle)),
                                     self.cy + int(ry*math.sin(angle)))
        for i, cid in enumerate(bottom_ids):
            angle = (math.pi / (len(bottom_ids)-1)) * i
            self.client_pos[cid] = (self.cx + int(rx*math.cos(angle)),
                                     self.cy + int(ry*math.sin(angle)))

        self.node_ovals = {}; self.node_labels = {}; self.acc_labels = {}
        self.server_oval = None; self.server_lbl = None
        self.server_sub = None; self.server_rnd = None
        self.anim_items = []; self.anim_after = None
        self._draw_static()

    def _draw_static(self):
        c = self.canvas
        for x in range(0, self.w, 70):
            c.create_line(x, 0, x, self.h, fill="#111128", width=1)
        for y in range(0, self.h, 70):
            c.create_line(0, y, self.w, y, fill="#111128", width=1)
        for cid, (x, y) in self.client_pos.items():
            c.create_line(x, y, self.cx, self.cy, fill="#1C1C3A", width=1,
                          dash=(3,8), tags="idleline")
        c.tag_lower("idleline")

        r = self.R_SERVER
        self.server_oval = c.create_oval(self.cx-r, self.cy-r, self.cx+r, self.cy+r,
                                          fill=CV_NODE_S, outline="#4A80C0", width=2)
        self.server_lbl = c.create_text(self.cx, self.cy-12, text="SERVER",
                                         fill=CV_TEXT, font=("Helvetica", 10, "bold"))
        self.server_sub = c.create_text(self.cx, self.cy+6, text="Global Model",
                                         fill="#8888BB", font=("Helvetica", 8))
        self.server_rnd = c.create_text(self.cx, self.cy+20, text="",
                                         fill="#B0B8D0", font=("Helvetica", 8, "bold"))

        for cid, (x, y) in self.client_pos.items():
            r = self.R_CLIENT
            ov = c.create_oval(x-r, y-r, x+r, y+r, fill=CV_NODE_C,
                                outline="#4A4A7A", width=1)
            lbl = c.create_text(x, y-8, text=f"C{cid}", fill=CV_TEXT,
                                 font=("Helvetica", 9, "bold"))
            acc = c.create_text(x, y+18, text="", fill=CV_TRAIN,
                                 font=("Helvetica", 7, "bold"))
            self.node_ovals[cid] = ov; self.node_labels[cid] = lbl
            self.acc_labels[cid] = acc

    def _clear_anim(self):
        for item in self.anim_items:
            try: self.canvas.delete(item)
            except: pass
        self.anim_items.clear()

    def _cancel_anim(self):
        if self.anim_after:
            try: self.root.after_cancel(self.anim_after)
            except: pass
            self.anim_after = None

    def set_status(self, text, color="#9090CC"):
        self.status_lbl.config(text=text, fg=color)

    def set_breadcrumb(self, active_key):
        for key, lbl in self.crumb_labels.items():
            if key == active_key:
                lbl.config(fg=PLAIN_STEP_COLORS[key], bg="#20223A")
            else:
                lbl.config(fg="#44446A", bg="#14142A")

    def clear_breadcrumb(self):
        for lbl in self.crumb_labels.values():
            lbl.config(fg="#44446A", bg="#14142A")

    def set_round_badge(self, rnd):
        text = f"Round {rnd}/10" if rnd > 0 else ""
        self.canvas.itemconfig(self.server_rnd, text=text)

    def reset(self):
        self._cancel_anim(); self._clear_anim()
        c = self.canvas
        for cid, ov in self.node_ovals.items():
            c.itemconfig(ov, fill=CV_NODE_C, outline="#4A4A7A", width=1)
            c.itemconfig(self.node_labels[cid], fill=CV_TEXT)
            c.itemconfig(self.acc_labels[cid], text="")
        c.itemconfig(self.server_oval, fill=CV_NODE_S, outline="#4A80C0", width=2)
        c.itemconfig(self.server_sub, text="Global Model", fill="#8888BB")
        c.itemconfig(self.server_rnd, text="")
        self.clear_breadcrumb()
        self.set_status("Ready — click Next Round to begin")

    def round_done_reset(self):
        c = self.canvas
        todel = []
        for item in self.anim_items:
            try:
                if c.type(item) in ("line", "oval", "rectangle"):
                    todel.append(item)
                elif c.type(item) == "text":
                    todel.append(item)
            except: pass
        for item in todel:
            try: c.delete(item)
            except: pass
        self.anim_items = [i for i in self.anim_items if i not in todel]
        for cid in PLAIN_ALL_CLIENTS:
            try: c.itemconfig(self.acc_labels[cid], text="")
            except: pass

    def animate_select(self, selected, rnd, on_done):
        c = self.canvas
        self._clear_anim()
        self.set_breadcrumb("select")
        self.set_status("Picking 3 random clients — no scoring, no history used",
                        PLAIN_STEP_COLORS["select"])
        self.set_round_badge(rnd)
        for cid, ov in self.node_ovals.items():
            if cid in selected:
                c.itemconfig(ov, fill="#4A5070", outline="#B0B8D0", width=2)
                c.itemconfig(self.node_labels[cid], fill="#FFFFFF")
            else:
                c.itemconfig(ov, fill="#181830", outline="#2A2A4A", width=1)
                c.itemconfig(self.node_labels[cid], fill="#383860")
        slbl = c.create_text(self.cx, self.cy + self.R_SERVER + 18,
                              text=f"Randomly drew C{selected[0]} · C{selected[1]} · C{selected[2]}",
                              fill="#C0C8DC", font=("Helvetica", 9, "bold"))
        self.anim_items.append(slbl)
        self.anim_after = self.root.after(900, on_done)

    def animate_train(self, selected, local_accs, on_done):
        c = self.canvas
        self.set_breadcrumb("train")
        self.set_status("Clients training locally — 2 epochs each, no epoch bar (plain run has no per-epoch log)",
                        PLAIN_STEP_COLORS["train"])
        c.itemconfig(self.server_sub, text="Awaiting", fill="#8888BB")
        bars = {}
        for cid in selected:
            x, y = self.client_pos[cid]
            r = self.R_CLIENT
            bg = c.create_rectangle(x-r+5, y+22, x+r-5, y+30, fill="#111130", outline="#333355")
            fill = c.create_rectangle(x-r+5, y+22, x-r+5, y+30, fill=CV_TRAIN, outline="")
            self.anim_items += [bg, fill]
            bars[cid] = (fill, x-r+5, y+22, x+r-5, y+30)
        total_steps = 16
        def grow(step=0):
            frac = step/total_steps
            for cid, (fill, bx1, by1, bx2, by2) in bars.items():
                nx = bx1 + (bx2-bx1)*frac
                c.coords(fill, bx1, by1, nx, by2)
            if step < total_steps:
                self.anim_after = self.root.after(int(1000/total_steps), lambda: grow(step+1))
            else:
                for cid, acc in zip(selected, local_accs):
                    c.itemconfig(self.acc_labels[cid], text=f"acc {acc:.4f}", fill="#8DE0A8")
                for fill, *_ in bars.values():
                    c.itemconfig(fill, fill="#5FBF8C")
                self.anim_after = self.root.after(350, on_done)
        grow()

    def animate_send(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("send")
        self.set_status("Sending updated weights to server — no raw data transmitted",
                        PLAIN_STEP_COLORS["send"])
        c.itemconfig(self.server_sub, text="Receiving", fill=CV_ARROW_UP)
        arrows = []
        for cid in selected:
            x, y = self.client_pos[cid]
            dot = c.create_oval(x-6, y-6, x+6, y+6, fill=CV_ARROW_UP, outline="")
            self.anim_items.append(dot)
            arrows.append((dot, x, y))
        steps = 20
        def travel(step=0):
            frac = step/steps
            ease = frac*frac*(3-2*frac)
            for dot, x1, y1 in arrows:
                nx = x1 + (self.cx-x1)*ease
                ny = y1 + (self.cy-y1)*ease
                c.coords(dot, nx-6, ny-6, nx+6, ny+6)
            if step < steps:
                self.anim_after = self.root.after(int(1000/steps), lambda: travel(step+1))
            else:
                for dot, *_ in arrows: c.delete(dot)
                on_done()
        travel()

    def animate_average(self, selected, on_done):
        c = self.canvas
        self.set_breadcrumb("average")
        self.set_status("Averaging updates equally — every client counts the same, regardless of quality",
                        PLAIN_STEP_COLORS["average"])
        c.itemconfig(self.server_sub, text="Averaging", fill="#B0B8D0")
        for cid in selected:
            x, y = self.client_pos[cid]
            al = c.create_line(x, y, self.cx, self.cy, fill="#8A8FA3", width=2,
                                arrow=tk.LAST, arrowshape=(8,10,3))
            wl = c.create_text((x+self.cx)//2, (y+self.cy)//2, text="1/3",
                                fill="#C0C8DC", font=("Helvetica", 8, "bold"))
            self.anim_items += [al, wl]
        def pulse(i=0):
            pulses = ["#8A8FA3","#6A6E82","#8A8FA3"]
            if i < len(pulses):
                c.itemconfig(self.server_oval, fill=pulses[i], width=3)
                self.anim_after = self.root.after(220, lambda: pulse(i+1))
            else:
                c.itemconfig(self.server_oval, fill=CV_NODE_S, outline="#B0B8D0", width=2)
                c.itemconfig(self.server_sub, text="Global Model ✓", fill="#B0B8D0")
                for cid2 in PLAIN_ALL_CLIENTS:
                    x2, y2 = self.client_pos[cid2]
                    bl = c.create_line(self.cx, self.cy, x2, y2, fill="#666A80",
                                        width=1, dash=(4,5))
                    self.anim_items.append(bl)
                self.anim_after = self.root.after(500, on_done)
        pulse()

    def show_best_checkpoint(self):
        c = self.canvas
        c.itemconfig(self.server_oval, fill="#4A4E60", outline="#B0B8D0", width=4)
        c.itemconfig(self.server_sub, text="★ Best (no rollback)", fill="#B0B8D0")
        glow = c.create_oval(self.cx-self.R_SERVER-12, self.cy-self.R_SERVER-12,
                              self.cx+self.R_SERVER+12, self.cy+self.R_SERVER+12,
                              outline="#B0B8D0", width=3, fill="")
        self.anim_items.append(glow)
        self.set_status(
            "Round 8 saved as best (highest test accuracy so far) — Acc 94.42%  F1 94.58%  AUC 0.9865",
            "#B0B8D0")


class SimpleStepCard:
    """Plain-tab counterpart to StepCard — same visual language, 4 steps."""
    def __init__(self, parent):
        self._frame = tk.Frame(parent, bg=C_CARD2, relief=tk.FLAT, pady=10, padx=14)
        self._frame.pack(fill=tk.BOTH, expand=True)
        top = tk.Frame(self._frame, bg=C_CARD2)
        top.pack(fill=tk.X, pady=(0,4))
        self._icon_lbl = tk.Label(top, text="", bg=C_CARD2, fg=C_TEXT, font=("Helvetica",22))
        self._icon_lbl.pack(side=tk.LEFT, padx=(0,10))
        right_hdr = tk.Frame(top, bg=C_CARD2)
        right_hdr.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._headline = tk.Label(right_hdr, text="", bg=C_CARD2, fg=C_TEXT,
                                   font=("Helvetica",13,"bold"), anchor="w")
        self._headline.pack(fill=tk.X)
        self._step_tag = tk.Label(right_hdr, text="", bg=C_CARD2, fg=C_TEXT2,
                                   font=("Helvetica",9), anchor="w")
        self._step_tag.pack(fill=tk.X)
        self._color_bar = tk.Frame(self._frame, bg="#444466", height=2)
        self._color_bar.pack(fill=tk.X, pady=(2,8))
        self._why_lbl = tk.Label(self._frame, text="", bg=C_CARD2, fg=C_TEXT,
                                  font=("Helvetica",10), wraplength=260, justify="left", anchor="nw")
        self._why_lbl.pack(fill=tk.X)
        self._detail_lbl = tk.Label(self._frame, text="", bg=C_CARD, fg=C_TEXT2,
                                     font=("Helvetica",9,"italic"), wraplength=260,
                                     justify="left", anchor="w", pady=5, padx=8)
        self._detail_lbl.pack(fill=tk.X, pady=(8,0))

    def show(self, step_key):
        info = PLAIN_STEP_CARDS.get(step_key, {})
        color = PLAIN_STEP_COLORS.get(step_key, "#888888")
        idx = next((i+1 for i,(k,_) in enumerate(PLAIN_STEPS_DEF) if k==step_key), "")
        self._icon_lbl.config(text=info.get("icon",""))
        self._headline.config(text=info.get("headline",""), fg=color)
        self._step_tag.config(text=f"Step {idx} of 4")
        self._color_bar.config(bg=color)
        self._why_lbl.config(text=info.get("why",""))
        self._detail_lbl.config(text=info.get("detail",""))

    def clear(self):
        for lbl in (self._icon_lbl, self._headline, self._step_tag,
                    self._why_lbl, self._detail_lbl):
            lbl.config(text="")
        self._color_bar.config(bg="#444466")


class DefenseDemo:
    def __init__(self, root, model_path, plain_model_path=None):
        self.root       = root
        self.model_path = model_path
        self.root.title("Enhanced FL Cycle — Defense Demo v8.3")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)
        self.root.minsize(1400, 820)

        self.model_lock = threading.Lock()
        # known_models maps a friendly display name -> resolved file path.
        # The default/enhanced model is always index 0; additional models
        # can be added via "Load Model...".
        self.known_models = {Path(model_path).name: model_path}
        self.active_model_name = Path(model_path).name
        self.model_cache = {}
        self.det_model_loading = False
        self.interpreter, self.inp, self.out = load_tflite_model(model_path)
        self.inp_dtype = self.inp["dtype"]
        self.inp_scale, self.inp_zp = self.inp.get("quantization",(1.0,0))
        self.out_scale, self.out_zp = self.out.get("quantization",(1.0,0))
        self.model_cache[model_path] = (self.interpreter, self.inp, self.out)

        mp_fd = mp.solutions.face_detection
        self.face_detector = mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        # ── Second, fully independent model/state for the Plain FedAvg tab.
        # A separate interpreter, lock, cache and face detector ensure a
        # crash or hang in one tab can never affect the other.
        self.pf_model_path = plain_model_path or resolve_plain_model_path()
        self.pf_model_lock = threading.Lock()
        self.pf_known_models = {Path(self.pf_model_path).name: self.pf_model_path}
        self.pf_active_model_name = Path(self.pf_model_path).name
        self.pf_model_cache = {}
        self.pfdet_model_loading = False
        self.pf_interpreter, self.pf_inp, self.pf_out = load_tflite_model(self.pf_model_path)
        self.pf_inp_dtype = self.pf_inp["dtype"]
        self.pf_inp_scale, self.pf_inp_zp = self.pf_inp.get("quantization",(1.0,0))
        self.pf_out_scale, self.pf_out_zp = self.pf_out.get("quantization",(1.0,0))
        self.pf_model_cache[self.pf_model_path] = (self.pf_interpreter, self.pf_inp, self.pf_out)
        self.pf_face_detector = mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        self._build_ui()

    # ── Top-level UI ──────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=C_PANEL, pady=9)
        hdr.pack(fill=tk.X)
        tk.Label(hdr,
                 text="An Enhanced Federated Cycle for DeepFake Detection  ·  Defense Demo",
                 bg=C_PANEL, fg=C_TEXT,
                 font=("Helvetica",13,"bold")).pack()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT,
                        padding=[22,7],  font=("Helvetica",11,"bold"))
        style.map("TNotebook.Tab",
                  background=[("selected",C_ACCENT)],
                  foreground=[("selected","white")])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_detection_tab()
        self._build_plain_detection_tab()
        self._build_fl_tab()
        self._build_plain_fl_tab()
        self._build_comparison_tab()
        self._build_sync_playback_tab()

        # Detector tabs are always index 0 (Enhanced) and 1 (Plain FedAvg).
        # Keyboard shortcuts route to whichever detector tab is active so
        # both tabs can use the same keys without clobbering each other.
        self.root.bind("<space>", self._detector_kb_dispatch("toggle_play"))
        self.root.bind("<Escape>", self._detector_kb_dispatch("stop"))
        self.root.bind("<r>", self._detector_kb_dispatch("restart"))
        self.root.bind("<s>", self._detector_kb_dispatch("save_frame"))

    def _detector_kb_dispatch(self, action):
        actions_tab0 = {
            "toggle_play": self._det_toggle_play,
            "stop": self._det_stop,
            "restart": self._det_restart,
            "save_frame": self._det_save_frame,
        }
        actions_tab1 = {
            "toggle_play": self._pfdet_toggle_play,
            "stop": self._pfdet_stop,
            "restart": self._pfdet_restart,
            "save_frame": self._pfdet_save_frame,
        }
        def handler(event=None):
            idx = self.nb.index(self.nb.select())
            if idx == 0:
                actions_tab0[action]()
            elif idx == 1:
                actions_tab1[action]()
        return handler

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Deepfake Detector
    # ══════════════════════════════════════════════════════════
    def _build_detection_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Deepfake Detector  ")

        tb = tk.Frame(frame, bg=C_PANEL, pady=7)
        tb.pack(fill=tk.X)
        tk.Button(tb, text="📂  Open Video", command=self._det_open_file,
                  bg=C_ACCENT, fg="white", font=("Helvetica",11,"bold"),
                  relief=tk.FLAT, padx=14, pady=5).pack(side=tk.LEFT, padx=(12,6))
        tk.Button(tb, text="💾  Export CSV", command=self._det_export_csv,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="🖼  Save Frame", command=self._det_save_frame,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        badge = tk.Frame(tb, bg=C_CARD2, padx=10, pady=4)
        badge.pack(side=tk.RIGHT, padx=12)
        tk.Label(badge, text="Model:", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",8)).pack(side=tk.LEFT)
        self.det_model_var = tk.StringVar(value=self.active_model_name)
        self.det_model_combo = ttk.Combobox(
            badge, textvariable=self.det_model_var,
            values=list(self.known_models.keys()),
            width=26, state="readonly", font=("Helvetica",8))
        self.det_model_combo.pack(side=tk.LEFT, padx=(4,4))
        self.det_model_combo.bind("<<ComboboxSelected>>", self._det_on_model_selected)
        tk.Button(badge, text="📁 Load Model…", command=self._det_browse_model,
                  bg=C_ACCENT, fg="white", font=("Helvetica",8,"bold"),
                  relief=tk.FLAT, padx=6, pady=2).pack(side=tk.LEFT)
        self.det_model_status = tk.Label(badge, text="", bg=C_CARD2, fg=C_GREEN,
                                          font=("Helvetica",8,"bold"))
        self.det_model_status.pack(side=tk.LEFT, padx=(6,0))

        main = tk.Frame(frame, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8,0))

        self.det_canvas = tk.Label(main, bg="#0A0A0A",
                                    width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.det_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sp = tk.Frame(main, bg=C_PANEL, width=220)
        sp.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))
        sp.pack_propagate(False)

        # ── Overall Verdict — dominant element at top ────────────
        tk.Label(sp, text="OVERALL VERDICT", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(14,2))
        self.det_lbl_overall = tk.Label(sp, text="—", bg=C_PANEL,
                                         fg=C_NEUTRAL, font=("Helvetica",32,"bold"))
        self.det_lbl_overall.pack(pady=(2,0))
        self.det_lbl_overall_sub = tk.Label(sp, text="", bg=C_PANEL,
                                             fg=C_TEXT2, font=("Helvetica",9),
                                             wraplength=200, justify="center")
        self.det_lbl_overall_sub.pack(padx=8, pady=(0,6))
        self._sep(sp)

        tk.Label(sp, text="INFERENCE STATS", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.det_stat_vars = {}
        for key, lbl in [("frame","Frame"),("fps","Display FPS"),
                          ("inf_ms","Inference ms"),("raw","Raw prob"),
                          ("smooth","Smoothed prob")]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=lbl+":", bg=C_PANEL, fg=C_TEXT2,
                     font=("Helvetica",9), anchor="w", width=14).pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            self.det_stat_vars[key] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9,"bold"), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        tk.Label(sp, text="FRAME TALLY", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.det_tally_vars = {}
        for verdict, hc in [("REAL",C_REAL),("FAKE",C_FAKE)]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=verdict+":", bg=C_PANEL, fg=hc,
                     font=("Helvetica",9,"bold"), anchor="w",
                     width=10).pack(side=tk.LEFT)
            v = tk.StringVar(value="0  (0.0%)")
            self.det_tally_vars[verdict] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        # (Overall Verdict shown at top of panel)

        pf = tk.Frame(frame, bg=C_BG, pady=4)
        pf.pack(fill=tk.X, padx=10)
        self.det_progress_var = tk.DoubleVar(value=0)
        self.det_progress_bar = ttk.Scale(pf, from_=0, to=100,
                                           orient="horizontal",
                                           variable=self.det_progress_var,
                                           command=self._det_on_seek)
        self.det_progress_bar.pack(fill=tk.X)
        self.det_lbl_time = tk.Label(pf, text="0:00 / 0:00",
                                      bg=C_BG, fg=C_TEXT2,
                                      font=("Helvetica",9))
        self.det_lbl_time.pack(anchor="e")

        cf = tk.Frame(frame, bg=C_PANEL, pady=8)
        cf.pack(fill=tk.X, side=tk.BOTTOM)
        bc = dict(bg=C_CARD, fg=C_TEXT, font=("Helvetica",12),
                  relief=tk.FLAT, width=4, pady=4)
        self.det_btn_play = tk.Button(cf, text="▶",
                                       command=self._det_toggle_play, **bc)
        self.det_btn_play.pack(side=tk.LEFT, padx=(14,2))
        tk.Button(cf, text="⏹", command=self._det_stop, **bc).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="⟳", command=self._det_restart, **bc).pack(side=tk.LEFT, padx=2)
        tk.Label(cf, text="Speed:", bg=C_PANEL, fg=C_TEXT,
                 font=("Helvetica",10)).pack(side=tk.LEFT, padx=(16,4))
        self.det_speed_var = tk.StringVar(value="1.0×")
        sm = ttk.Combobox(cf, textvariable=self.det_speed_var,
                          values=[f"{s}×" for s in SPEED_OPTIONS],
                          width=6, state="readonly")
        sm.pack(side=tk.LEFT, padx=4)
        sm.bind("<<ComboboxSelected>>", self._det_on_speed_change)
        self.det_lbl_status = tk.Label(
            cf, text="Open a video file to begin.  [Space=play/pause · Esc=stop · R=restart · S=save frame]",
            bg=C_PANEL, fg=C_TEXT2, font=("Helvetica",9))
        self.det_lbl_status.pack(side=tk.LEFT, padx=16)

        self.det_cap=None; self.det_video_path=None
        self.det_total_frames=0; self.det_video_fps=30.0
        self.det_frame_idx=0; self.det_playing=False
        self.det_speed=1.0; self.det_seek_pending=None
        self.det_after_id=None; self.det_loop_active=True
        self.det_history=collections.deque(maxlen=HISTORY_SIZE)
        self.det_raw_prob=0.5; self.det_smoothed=0.5
        self.det_label="NO FACE"; self.det_cv_color=COLOR_NO_FACE
        self.det_hex_color=C_NEUTRAL; self.det_confidence=0.0
        self.det_inf_ms=0.0; self.det_bbox=None
        self.det_verdict_counts={"REAL":0,"FAKE":0}
        self.det_frame_log=[]; self.det_last_det=None
        self.det_fps_display=0.0; self.det_t_prev=time.time()
        self.det_vid_w=DISPLAY_MAX_W; self.det_vid_h=DISPLAY_MAX_H
        self.det_display_w=DISPLAY_MAX_W; self.det_display_h=DISPLAY_MAX_H
        self.det_infer_lock=threading.Lock()
        self.det_infer_queue=[]; self.det_infer_queue_lock=threading.Lock()
        self.det_stop_event=threading.Event()
        self.det_infer_error_shown=False

        self._det_show_placeholder()
        self._det_start_inference_thread()
        self.det_after_id = self.root.after(33, self._det_update_loop)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — FL Simulation
    # New layout:
    #   left (canvas 940×560 + ledger)  |  centre (step card)  |  right (metrics + chart)
    # ══════════════════════════════════════════════════════════
    def _build_fl_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  FL Simulation  ")

        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_mode   = tk.BooleanVar(value=False)
        self.fl_auto_paused = False
        self.fl_auto_after  = None
        self.fl_step_pending_cb = None  # callback waiting for Next Step press

        # ── Three-column layout ───────────────────────────────
        left = tk.Frame(frame, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,4), pady=6)

        # Centre column: step card + controls
        centre = tk.Frame(frame, bg=C_BG, width=300)
        centre.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4), pady=6)
        centre.pack_propagate(False)

        # Right column: round header + metrics + best banner + chart
        right = tk.Frame(frame, bg=C_BG, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,8), pady=6)
        right.pack_propagate(False)

        # ── Network canvas (larger) ────────────────────────────
        canvas_host = tk.Frame(left, bg=CV_BG)
        canvas_host.pack(fill=tk.BOTH, expand=True)
        self.net_canvas = NetworkCanvas(canvas_host, width=940, height=560)

        # ── Ledger strip (two columns) ────────────────────────
        self.ledger = LedgerStrip(left)

        # ══ Centre column contents ════════════════════════════
        # Step card label at top
        tk.Label(centre, text="WHAT'S HAPPENING", bg=C_BG, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(fill=tk.X, padx=6, pady=(0,4))

        # Step card occupies most of centre column
        card_host = tk.Frame(centre, bg=C_CARD2)
        card_host.pack(fill=tk.BOTH, expand=True, padx=0)
        self.step_card = StepCard(card_host)
        self.step_card.clear()

        # (Controls moved to right panel)

        # ══ Right column contents ════════════════════════════

        # ── FL Controls (top of right panel) ────────────────────
        ctrl_outer = tk.Frame(right, bg=C_PANEL, pady=6, padx=8)
        ctrl_outer.pack(fill=tk.X, pady=(0,5))

        ctrl_row1 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row1.pack(fill=tk.X, pady=(0,3))

        self.btn_next = tk.Button(ctrl_row1, text="▶  Next Round",
                                   command=self._fl_next_round,
                                   bg=C_ACCENT, fg="white",
                                   font=("Helvetica",10,"bold"),
                                   relief=tk.FLAT, padx=10, pady=5)
        self.btn_next.pack(side=tk.LEFT, padx=(0,5))

        tk.Button(ctrl_row1, text="⟳  Reset", command=self._fl_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)

        # Next Step button — enabled only while a round is mid-animation
        self.btn_next_step = tk.Button(ctrl_row1, text="Step ▶",
                                        command=self._fl_advance_step,
                                        bg="#3A4060", fg=C_TEXT2,
                                        font=("Helvetica",9),
                                        relief=tk.FLAT, padx=8, pady=5,
                                        state=tk.DISABLED)
        self.btn_next_step.pack(side=tk.LEFT, padx=(8,0))

        ctrl_row2 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row2.pack(fill=tk.X)
        self.auto_chk = tk.Checkbutton(
            ctrl_row2, text="Auto-run  (pauses at Round 6)",
            variable=self.fl_auto_mode,
            command=self._fl_on_auto_toggle,
            bg=C_PANEL, fg=C_TEXT, selectcolor=C_CARD,
            font=("Helvetica",9),
            activebackground=C_PANEL, activeforeground=C_TEXT)
        self.auto_chk.pack(side=tk.LEFT)
        self.lbl_auto_status = tk.Label(ctrl_row2, text="", bg=C_PANEL, fg=C_GOLD,
                                         font=("Helvetica",8,"bold"))
        self.lbl_auto_status.pack(side=tk.LEFT, padx=(8,0))

        # Round header
        hcard = tk.Frame(right, bg=C_CARD, pady=8, padx=12)
        hcard.pack(fill=tk.X, pady=(0,5))
        self.lbl_round = tk.Label(hcard, text="Pre-Training Baseline",
                                   bg=C_CARD, fg=C_TEXT,
                                   font=("Helvetica",12,"bold"), anchor="w")
        self.lbl_round.pack(fill=tk.X)
        self.lbl_phase = tk.Label(hcard,
                                   text="Global model initialised. No FL rounds applied yet.",
                                   bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8),
                                   anchor="w", wraplength=300, justify="left")
        self.lbl_phase.pack(fill=tk.X, pady=(3,0))

        # Metrics
        mf = tk.Frame(right, bg=C_CARD)
        mf.pack(fill=tk.X, pady=(0,5))
        self.metric_vars   = {}
        self.metric_frames = {}
        for col, (key, lbl) in enumerate([("acc","Accuracy"),
                                           ("f1","F1 Score"),
                                           ("auc","ROC-AUC")]):
            cell = tk.Frame(mf, bg=C_CARD, padx=8, pady=7)
            cell.grid(row=0, column=col, sticky="nsew")
            mf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg=C_CARD, fg=C_TEXT2,
                     font=("Helvetica",7,"bold")).pack()
            v  = tk.StringVar(value="—")
            self.metric_vars[key] = v
            ml = tk.Label(cell, textvariable=v, bg=C_CARD, fg=C_GREEN,
                          font=("Helvetica",16,"bold"))
            ml.pack()
            self.metric_frames[key] = (cell, ml)

        # Best checkpoint banner (hidden until Round 6)
        self.best_banner = tk.Frame(right, bg=C_GOLD)
        tk.Label(self.best_banner,
                 text="★  BEST CHECKPOINT  —  ROUND 6  ★",
                 bg=C_GOLD, fg="#1A1000",
                 font=("Helvetica",9,"bold")).pack(pady=(5,1))
        bmf = tk.Frame(self.best_banner, bg=C_GOLD)
        bmf.pack(fill=tk.X, padx=6, pady=(0,5))
        for col, (lbl, val) in enumerate([("Accuracy","96.52%"),
                                           ("F1 Score","96.52%"),
                                           ("ROC-AUC","0.9964")]):
            cell = tk.Frame(bmf, bg="#9A6800", padx=8, pady=3)
            cell.grid(row=0, column=col, padx=2, sticky="nsew")
            bmf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg="#9A6800", fg="#FFE8A0",
                     font=("Helvetica",7,"bold")).pack()
            tk.Label(cell, text=val, bg="#9A6800", fg="white",
                     font=("Helvetica",12,"bold")).pack()

        # Chart
        self._build_fl_chart(right)
        self._fl_update_display(0)

    def _build_fl_chart(self, parent):
        # Match the mid-light theme in chart surroundings
        fig_bg = C_BG   # slate — matches right panel
        ax_bg  = "#22243A"
        self.fl_fig, self.fl_axes = plt.subplots(
            2, 1, figsize=(3.4, 4.2),
            facecolor=fig_bg, gridspec_kw={"hspace":0.60})
        self.fl_fig.patch.set_facecolor(fig_bg)

        for ax in self.fl_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = self.fl_axes
        ax_acc.set_title("Accuracy & F1 per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_acc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_acc.set_ylabel("Score",  color=C_TEXT2, fontsize=7)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(CHART_ACC_LO, CHART_ACC_HI)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        # Tight Y tick labels
        ax_acc.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.3f"))

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_auc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_auc.set_ylabel("AUC",   color=C_TEXT2, fontsize=7)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(CHART_AUC_LO, CHART_AUC_HI)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_auc.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.4f"))

        self.line_acc, = ax_acc.plot([], [], color="#6AB4FF", linewidth=1.8,
                                      marker="o", markersize=4, label="Accuracy", zorder=3)
        self.line_f1,  = ax_acc.plot([], [], color="#FF9966", linewidth=1.2,
                                      linestyle="--", marker="s", markersize=3,
                                      label="F1", zorder=3)
        ax_acc.legend(fontsize=6, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        self.line_auc, = ax_auc.plot([], [], color=C_GREEN, linewidth=1.8,
                                      marker="o", markersize=4, zorder=3)

        self.vline_acc = ax_acc.axvline(x=6, color=C_GOLD, linewidth=1.8,
                                         linestyle=":", alpha=0.0)
        self.vline_auc = ax_auc.axvline(x=6, color=C_GOLD, linewidth=1.8,
                                         linestyle=":", alpha=0.0)
        self.ann_r6_acc = ax_acc.annotate(
            "★ R6\n96.52%", xy=(6,0.9652), xytext=(7.5, CHART_ACC_LO+0.003),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)
        self.ann_r6_auc = ax_auc.annotate(
            "★ R6\n0.9964", xy=(6,0.9964), xytext=(7.5, CHART_AUC_LO+0.0005),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)

        self.fl_canvas_widget = FigureCanvasTkAgg(self.fl_fig, master=parent)
        self.fl_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fl_canvas_widget.draw()

    # ── FL orchestration ──────────────────────────────────────
    def _fl_next_round(self):
        if self.fl_animating: return
        if self.fl_auto_paused:
            self.fl_auto_paused = False
            self.lbl_auto_status.config(
                text="▶ Auto running…" if self.fl_auto_mode.get() else "")
        next_idx = self.fl_round_index + 1
        if next_idx >= len(ROUND_DATA):
            self.net_canvas.set_status(
                "All 10 rounds complete. Best-weight tracking deployed Round 6 model.",
                C_GOLD)
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto()
            return
        self.fl_animating = True
        self.btn_next.config(state=tk.DISABLED)
        self.auto_chk.config(state=tk.DISABLED)
        self._fl_run_round(next_idx)

    def _fl_run_round(self, idx):
        """Pure callback chain — no blocking thread.
        Each step animates, then waits for either auto-advance or manual Next Step."""
        rd       = ROUND_DATA[idx]
        selected = rd[6]
        rnd      = rd[0]

        self._fl_set_header(idx)

        # Enable Next Step button while round is in progress
        self.btn_next_step.config(state=tk.NORMAL, bg=C_ACCENT, fg="white")

        def _gate(next_cb):
            """After an animation finishes, either auto-advance or wait for button."""
            if self.fl_auto_mode.get():
                self.root.after(int(T_AUTO_GAP * 600), next_cb)
            else:
                # Park the callback — Next Step button will fire it
                self.fl_step_pending_cb = next_cb

        def step_select():
            self.step_card.show("select")
            self.net_canvas.animate_select(selected, rnd, lambda: _gate(step_train))

        def step_train():
            self.step_card.show("train")
            self.net_canvas.animate_train(selected, lambda: _gate(step_send))

        def step_send():
            self.step_card.show("send")
            self.net_canvas.animate_send(selected, lambda: _gate(step_validate))

        def step_validate():
            self.step_card.show("validate")
            self.net_canvas.animate_validate(selected, lambda: _gate(step_aggregate))

        def step_aggregate():
            self.step_card.show("aggregate")
            self.net_canvas.animate_aggregate(selected, lambda: _gate(step_rep))

        def step_rep():
            self.step_card.show("rep")
            self.net_canvas.animate_reputation(selected, rnd, step_finish)

        def step_finish():
            # Disable Next Step — round done
            self.btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
            self.fl_step_pending_cb = None
            self._fl_update_metrics(idx)
            self._fl_update_chart(idx)
            self.ledger.advance_to(rnd)
            self.step_card.clear()
            self.net_canvas.clear_breadcrumb()
            if rnd == 6:
                self._fl_show_best_checkpoint()
                self._fl_pause_auto_at_r6()
            else:
                self._fl_finish_round(idx)

        self.root.after(0, step_select)

    def _fl_advance_step(self):
        """Called by the Next Step button to fire the pending inter-step callback."""
        cb = self.fl_step_pending_cb
        if cb is not None:
            self.fl_step_pending_cb = None
            cb()

    def _fl_set_header(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        if rnd == 0:
            self.lbl_round.config(text="Pre-Training Baseline")
            self.lbl_phase.config(
                text="Global model initialised with pre-trained EfficientNetB4 weights. "
                     "No federated rounds applied yet.")
        else:
            self.lbl_round.config(text=f"Round {rnd}  /  10")
            self.lbl_phase.config(
                text=f"Selected: C{rd[6][0]}  ·  C{rd[6][1]}  ·  C{rd[6][2]}     "
                     f"Updates accepted: 3 / 3")

    def _fl_update_metrics(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        for key, val in [("acc", f"{rd[1]*100:.2f}%"),
                          ("f1",  f"{rd[2]*100:.2f}%"),
                          ("auc", f"{rd[3]:.4f}")]:
            self.metric_vars[key].set(val)
            cell, lbl = self.metric_frames[key]
            color = C_GOLD if rnd == 6 else C_GREEN
            bg    = "#1A1500" if rnd == 6 else C_CARD
            lbl.config(fg=color); cell.config(bg=bg)

        # Flash metrics gold three times at Round 6
        if rnd == 6:
            self._flash_metrics(3)

    def _flash_metrics(self, remaining):
        """Toggle metric labels bright-white/gold to draw the eye."""
        if remaining <= 0:
            for key in ("acc","f1","auc"):
                _, lbl = self.metric_frames[key]
                lbl.config(fg=C_GOLD)
            return
        flash_on = (remaining % 2 == 1)
        for key in ("acc","f1","auc"):
            cell, lbl = self.metric_frames[key]
            lbl.config(fg="white" if flash_on else C_GOLD)
            cell.config(bg="#3A2800" if flash_on else "#1A1500")
        self.root.after(220, lambda: self._flash_metrics(remaining - 1))

    def _fl_update_chart(self, idx):
        xs   = [ROUND_DATA[i][0] for i in range(idx+1)]
        accs = [ROUND_DATA[i][1] for i in range(idx+1)]
        f1s  = [ROUND_DATA[i][2] for i in range(idx+1)]
        aucs = [ROUND_DATA[i][3] for i in range(idx+1)]
        self.line_acc.set_data(xs, accs)
        self.line_f1.set_data(xs, f1s)
        self.line_auc.set_data(xs, aucs)
        if ROUND_DATA[idx][0] >= 6:
            self.vline_acc.set_alpha(0.9); self.vline_auc.set_alpha(0.9)
            self.ann_r6_acc.set_alpha(1.0); self.ann_r6_auc.set_alpha(1.0)
        self.fl_canvas_widget.draw_idle()

    def _fl_show_best_checkpoint(self):
        self.best_banner.pack(fill=tk.X, pady=(0,5),
                              after=self.metric_frames["auc"][0].master)
        self.net_canvas.show_best_checkpoint()

    def _fl_finish_round(self, idx):
        self.net_canvas.round_done_reset()
        self.fl_round_index = idx
        self.fl_animating   = False
        self.auto_chk.config(state=tk.NORMAL)

        if idx < len(ROUND_DATA) - 1:
            self.btn_next.config(state=tk.NORMAL)
        else:
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto(); return

        if self.fl_auto_mode.get() and not self.fl_auto_paused:
            self.fl_auto_after = self.root.after(
                int(T_AUTO_GAP*1000), self._fl_next_round)

    def _fl_pause_auto_at_r6(self):
        self.net_canvas.round_done_reset()
        self.fl_round_index = 6
        self.fl_animating   = False
        self.fl_auto_paused = True
        self.lbl_auto_status.config(text="⏸ Paused at Round 6")
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)

    def _fl_on_auto_toggle(self):
        if self.fl_auto_mode.get():
            self.lbl_auto_status.config(text="▶ Auto running…")
            if self.fl_auto_paused:
                self.fl_auto_paused = False
            if not self.fl_animating and self.fl_round_index < len(ROUND_DATA)-1:
                self.fl_auto_after = self.root.after(
                    int(T_AUTO_GAP*1000), self._fl_next_round)
        else:
            self.lbl_auto_status.config(text="")
            if self.fl_auto_after:
                try: self.root.after_cancel(self.fl_auto_after)
                except: pass
                self.fl_auto_after = None

    def _fl_disable_auto(self):
        self.fl_auto_mode.set(False)
        self.lbl_auto_status.config(text="")
        if self.fl_auto_after:
            try: self.root.after_cancel(self.fl_auto_after)
            except: pass
            self.fl_auto_after = None

    def _fl_reset(self):
        self._fl_disable_auto()
        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_paused = False
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)
        self.btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
        self.fl_step_pending_cb = None
        self.net_canvas.reset()
        self.ledger.reset()
        self.step_card.clear()

        self.line_acc.set_data([],[]); self.line_f1.set_data([],[])
        self.line_auc.set_data([],[])
        self.vline_acc.set_alpha(0.0); self.vline_auc.set_alpha(0.0)
        self.ann_r6_acc.set_alpha(0.0); self.ann_r6_auc.set_alpha(0.0)
        self.fl_canvas_widget.draw_idle()

        for key in ("acc","f1","auc"):
            self.metric_vars[key].set("—")
            cell, lbl = self.metric_frames[key]
            lbl.config(fg=C_GREEN); cell.config(bg=C_CARD)

        try: self.best_banner.pack_forget()
        except: pass

        self.lbl_round.config(text="Pre-Training Baseline")
        self.lbl_phase.config(
            text="Global model initialised. No FL rounds applied yet.")
        self._fl_update_display(0)

    def _fl_update_display(self, idx):
        rd = ROUND_DATA[idx]
        self.metric_vars["acc"].set(f"{rd[1]*100:.2f}%")
        self.metric_vars["f1"].set(f"{rd[2]*100:.2f}%")
        self.metric_vars["auc"].set(f"{rd[3]:.4f}")


    # ══════════════════════════════════════════════════════════
    # TAB 2 — Plain FedAvg (baseline)
    # ══════════════════════════════════════════════════════════
    def _build_plain_fl_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Plain FedAvg (Baseline)  ")

        self.pf_round_index = 0
        self.pf_animating = False
        self.pf_auto_mode = tk.BooleanVar(value=False)
        self.pf_auto_after = None
        self.pf_step_pending_cb = None

        left = tk.Frame(frame, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,4), pady=6)

        centre = tk.Frame(frame, bg=C_BG, width=300)
        centre.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4), pady=6)
        centre.pack_propagate(False)

        right = tk.Frame(frame, bg=C_BG, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,8), pady=6)
        right.pack_propagate(False)

        canvas_host = tk.Frame(left, bg=CV_BG)
        canvas_host.pack(fill=tk.BOTH, expand=True)
        self.pf_canvas = SimpleNetworkCanvas(canvas_host, width=940, height=560)

        # No reputation ledger for plain FedAvg — replaced with a note strip
        note = tk.Frame(left, bg=C_PANEL, pady=6, padx=8)
        note.pack(fill=tk.X, pady=(6,0))
        tk.Label(note, text="ℹ  Plain FedAvg keeps no reputation ledger — client "
                             "selection is uniform-random every round, with no "
                             "memory of past performance.",
                 bg=C_PANEL, fg=C_TEXT2, font=("Helvetica", 9), anchor="w",
                 wraplength=900, justify="left").pack(fill=tk.X)

        tk.Label(centre, text="WHAT'S HAPPENING", bg=C_BG, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(fill=tk.X, padx=6, pady=(0,4))
        card_host = tk.Frame(centre, bg=C_CARD2)
        card_host.pack(fill=tk.BOTH, expand=True, padx=0)
        self.pf_step_card = SimpleStepCard(card_host)
        self.pf_step_card.clear()

        ctrl_outer = tk.Frame(right, bg=C_PANEL, pady=6, padx=8)
        ctrl_outer.pack(fill=tk.X, pady=(0,5))
        ctrl_row1 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row1.pack(fill=tk.X, pady=(0,3))
        self.pf_btn_next = tk.Button(ctrl_row1, text="▶  Next Round",
                                      command=self._pf_next_round,
                                      bg=C_ACCENT, fg="white",
                                      font=("Helvetica",10,"bold"),
                                      relief=tk.FLAT, padx=10, pady=5)
        self.pf_btn_next.pack(side=tk.LEFT, padx=(0,5))
        tk.Button(ctrl_row1, text="⟳  Reset", command=self._pf_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)
        self.pf_btn_next_step = tk.Button(ctrl_row1, text="Step ▶",
                                           command=self._pf_advance_step,
                                           bg="#3A4060", fg=C_TEXT2,
                                           font=("Helvetica",9),
                                           relief=tk.FLAT, padx=8, pady=5,
                                           state=tk.DISABLED)
        self.pf_btn_next_step.pack(side=tk.LEFT, padx=(8,0))

        ctrl_row2 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row2.pack(fill=tk.X)
        self.pf_auto_chk = tk.Checkbutton(
            ctrl_row2, text="Auto-run  (stops after Round 10)",
            variable=self.pf_auto_mode, command=self._pf_on_auto_toggle,
            bg=C_PANEL, fg=C_TEXT, selectcolor=C_CARD, font=("Helvetica",9),
            activebackground=C_PANEL, activeforeground=C_TEXT)
        self.pf_auto_chk.pack(side=tk.LEFT)
        self.pf_lbl_auto_status = tk.Label(ctrl_row2, text="", bg=C_PANEL,
                                            fg="#B0B8D0", font=("Helvetica",8,"bold"))
        self.pf_lbl_auto_status.pack(side=tk.LEFT, padx=(8,0))

        hcard = tk.Frame(right, bg=C_CARD, pady=8, padx=12)
        hcard.pack(fill=tk.X, pady=(0,5))
        self.pf_lbl_round = tk.Label(hcard, text="Pre-Training Baseline",
                                      bg=C_CARD, fg=C_TEXT,
                                      font=("Helvetica",12,"bold"), anchor="w")
        self.pf_lbl_round.pack(fill=tk.X)
        self.pf_lbl_phase = tk.Label(hcard,
                                      text="Global model initialised. No FL rounds applied yet.",
                                      bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8),
                                      anchor="w", wraplength=300, justify="left")
        self.pf_lbl_phase.pack(fill=tk.X, pady=(3,0))

        mf = tk.Frame(right, bg=C_CARD)
        mf.pack(fill=tk.X, pady=(0,5))
        self.pf_metric_vars = {}
        self.pf_metric_frames = {}
        for col, (key, lbl) in enumerate([("acc","Accuracy"),("f1","F1 Score"),("auc","ROC-AUC")]):
            cell = tk.Frame(mf, bg=C_CARD, padx=8, pady=7)
            cell.grid(row=0, column=col, sticky="nsew")
            mf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg=C_CARD, fg=C_TEXT2,
                     font=("Helvetica",7,"bold")).pack()
            v = tk.StringVar(value="—")
            self.pf_metric_vars[key] = v
            ml = tk.Label(cell, textvariable=v, bg=C_CARD, fg="#B0B8D0",
                          font=("Helvetica",16,"bold"))
            ml.pack()
            self.pf_metric_frames[key] = (cell, ml)

        self.pf_best_banner = tk.Frame(right, bg="#5A5E70")
        tk.Label(self.pf_best_banner, text="★  BEST CHECKPOINT  —  ROUND 8  ★",
                 bg="#5A5E70", fg="white",
                 font=("Helvetica",9,"bold")).pack(pady=(5,1))
        bmf = tk.Frame(self.pf_best_banner, bg="#5A5E70")
        bmf.pack(fill=tk.X, padx=6, pady=(0,5))
        for col, (lbl, val) in enumerate([("Accuracy","94.42%"),
                                           ("F1 Score","94.58%"),
                                           ("ROC-AUC","0.9865")]):
            cell = tk.Frame(bmf, bg="#42465A", padx=8, pady=3)
            cell.grid(row=0, column=col, padx=2, sticky="nsew")
            bmf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg="#42465A", fg="#D8DCE8",
                     font=("Helvetica",7,"bold")).pack()
            tk.Label(cell, text=val, bg="#42465A", fg="white",
                     font=("Helvetica",12,"bold")).pack()

        self._build_pf_chart(right)
        self._pf_update_display(0)

    def _build_pf_chart(self, parent):
        fig_bg = C_BG; ax_bg = "#22243A"
        self.pf_fig, self.pf_axes = plt.subplots(
            2, 1, figsize=(3.4, 4.2), facecolor=fig_bg,
            gridspec_kw={"hspace":0.60})
        self.pf_fig.patch.set_facecolor(fig_bg)
        for ax in self.pf_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = self.pf_axes
        ax_acc.set_title("Accuracy & F1 per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_acc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_acc.set_ylabel("Score", color=C_TEXT2, fontsize=7)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(CHART_PLAIN_ACC_LO, CHART_PLAIN_ACC_HI)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_acc.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_auc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_auc.set_ylabel("AUC", color=C_TEXT2, fontsize=7)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(CHART_PLAIN_AUC_LO, CHART_PLAIN_AUC_HI)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_auc.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))

        self.pf_line_acc, = ax_acc.plot([], [], color="#8FA8D0", linewidth=1.8,
                                         marker="o", markersize=4, label="Accuracy", zorder=3)
        self.pf_line_f1, = ax_acc.plot([], [], color="#C9A0DC", linewidth=1.2,
                                        linestyle="--", marker="s", markersize=3,
                                        label="F1", zorder=3)
        ax_acc.legend(fontsize=6, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)
        self.pf_line_auc, = ax_auc.plot([], [], color="#8FD0B0", linewidth=1.8,
                                         marker="o", markersize=4, zorder=3)
        self.pf_vline_acc = ax_acc.axvline(x=8, color="#B0B8D0", linewidth=1.8,
                                            linestyle=":", alpha=0.0)
        self.pf_vline_auc = ax_auc.axvline(x=8, color="#B0B8D0", linewidth=1.8,
                                            linestyle=":", alpha=0.0)
        self.pf_ann_r8_acc = ax_acc.annotate(
            "★ R8\n94.42%", xy=(8,0.9442), xytext=(6.0, CHART_PLAIN_ACC_LO+0.003),
            color="#B0B8D0", fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B0B8D0", lw=1.0), alpha=0.0)
        self.pf_ann_r8_auc = ax_auc.annotate(
            "★ R8\n0.9865", xy=(8,0.9865), xytext=(6.0, CHART_PLAIN_AUC_LO+0.0005),
            color="#B0B8D0", fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B0B8D0", lw=1.0), alpha=0.0)

        self.pf_canvas_widget = FigureCanvasTkAgg(self.pf_fig, master=parent)
        self.pf_canvas_widget.get_tk_widget().pack(fill=tk.X, pady=(5,0))
        self.pf_canvas_widget.draw()

    def _pf_next_round(self):
        if self.pf_animating: return
        next_idx = self.pf_round_index + 1
        if next_idx >= len(PLAIN_ROUND_DATA_FILLED):
            self.pf_canvas.set_status(
                "All 10 rounds complete. Round 8 kept as best (highest test accuracy).",
                "#B0B8D0")
            self.pf_btn_next.config(state=tk.DISABLED)
            self._pf_disable_auto()
            return
        self.pf_animating = True
        self.pf_btn_next.config(state=tk.DISABLED)
        self.pf_auto_chk.config(state=tk.DISABLED)
        self._pf_run_round(next_idx)

    def _pf_run_round(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd, selected, local_accs = rd[0], rd[5], rd[6]
        self._pf_set_header(idx)
        self.pf_btn_next_step.config(state=tk.NORMAL, bg=C_ACCENT, fg="white")

        def _gate(next_cb):
            if self.pf_auto_mode.get():
                self.root.after(int(T_AUTO_GAP * 600), next_cb)
            else:
                self.pf_step_pending_cb = next_cb

        def step_select():
            self.pf_step_card.show("select")
            self.pf_canvas.animate_select(selected, rnd, lambda: _gate(step_train))

        def step_train():
            self.pf_step_card.show("train")
            self.pf_canvas.animate_train(selected, local_accs, lambda: _gate(step_send))

        def step_send():
            self.pf_step_card.show("send")
            self.pf_canvas.animate_send(selected, lambda: _gate(step_average))

        def step_average():
            self.pf_step_card.show("average")
            self.pf_canvas.animate_average(selected, step_finish)

        def step_finish():
            self.pf_btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
            self.pf_step_pending_cb = None
            self._pf_update_metrics(idx)
            self._pf_update_chart(idx)
            self.pf_step_card.clear()
            self.pf_canvas.clear_breadcrumb()
            if rnd == PLAIN_BEST_ROUND:
                self._pf_show_best_checkpoint()
            self._pf_finish_round(idx)

        self.root.after(0, step_select)

    def _pf_advance_step(self):
        cb = self.pf_step_pending_cb
        if cb is not None:
            self.pf_step_pending_cb = None
            cb()

    def _pf_set_header(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd, selected = rd[0], rd[5]
        if rnd == 0:
            self.pf_lbl_round.config(text="Pre-Training Baseline")
            self.pf_lbl_phase.config(
                text="Global model initialised with pre-trained weights. "
                     "No federated rounds applied yet.")
        else:
            self.pf_lbl_round.config(text=f"Round {rnd}  /  10")
            self.pf_lbl_phase.config(
                text=f"Randomly selected: C{selected[0]}  ·  C{selected[1]}  ·  C{selected[2]}     "
                     f"All updates used (no validation gate)")

    def _pf_update_metrics(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd = rd[0]
        for key, val in [("acc", f"{rd[1]*100:.2f}%"), ("f1", f"{rd[2]*100:.2f}%"),
                          ("auc", f"{rd[3]:.4f}")]:
            self.pf_metric_vars[key].set(val)
            cell, lbl = self.pf_metric_frames[key]
            color = "#B0B8D0" if rnd == PLAIN_BEST_ROUND else "#8FA8D0"
            bg = "#33364A" if rnd == PLAIN_BEST_ROUND else C_CARD
            lbl.config(fg=color); cell.config(bg=bg)

    def _pf_update_chart(self, idx):
        xs = [PLAIN_ROUND_DATA_FILLED[i][0] for i in range(idx+1)]
        accs = [PLAIN_ROUND_DATA_FILLED[i][1] for i in range(idx+1)]
        f1s = [PLAIN_ROUND_DATA_FILLED[i][2] for i in range(idx+1)]
        aucs = [PLAIN_ROUND_DATA_FILLED[i][3] for i in range(idx+1)]
        self.pf_line_acc.set_data(xs, accs)
        self.pf_line_f1.set_data(xs, f1s)
        self.pf_line_auc.set_data(xs, aucs)
        if PLAIN_ROUND_DATA_FILLED[idx][0] >= PLAIN_BEST_ROUND:
            self.pf_vline_acc.set_alpha(0.9); self.pf_vline_auc.set_alpha(0.9)
            self.pf_ann_r8_acc.set_alpha(1.0); self.pf_ann_r8_auc.set_alpha(1.0)
        self.pf_canvas_widget.draw_idle()

    def _pf_show_best_checkpoint(self):
        self.pf_best_banner.pack(fill=tk.X, pady=(0,5),
                                  after=self.pf_metric_frames["auc"][0].master)
        self.pf_canvas.show_best_checkpoint()

    def _pf_finish_round(self, idx):
        self.pf_canvas.round_done_reset()
        self.pf_round_index = idx
        self.pf_animating = False
        self.pf_auto_chk.config(state=tk.NORMAL)
        if idx < len(PLAIN_ROUND_DATA_FILLED) - 1:
            self.pf_btn_next.config(state=tk.NORMAL)
        else:
            self.pf_btn_next.config(state=tk.DISABLED)
            self._pf_disable_auto(); return
        if self.pf_auto_mode.get():
            self.pf_auto_after = self.root.after(int(T_AUTO_GAP*1000), self._pf_next_round)

    def _pf_on_auto_toggle(self):
        if self.pf_auto_mode.get():
            self.pf_lbl_auto_status.config(text="▶ Auto running…")
            if not self.pf_animating and self.pf_round_index < len(PLAIN_ROUND_DATA_FILLED)-1:
                self.pf_auto_after = self.root.after(int(T_AUTO_GAP*1000), self._pf_next_round)
        else:
            self.pf_lbl_auto_status.config(text="")
            if self.pf_auto_after:
                try: self.root.after_cancel(self.pf_auto_after)
                except: pass
                self.pf_auto_after = None

    def _pf_disable_auto(self):
        self.pf_auto_mode.set(False)
        self.pf_lbl_auto_status.config(text="")
        if self.pf_auto_after:
            try: self.root.after_cancel(self.pf_auto_after)
            except: pass
            self.pf_auto_after = None

    def _pf_reset(self):
        self._pf_disable_auto()
        self.pf_round_index = 0
        self.pf_animating = False
        self.pf_btn_next.config(state=tk.NORMAL)
        self.pf_auto_chk.config(state=tk.NORMAL)
        self.pf_btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
        self.pf_step_pending_cb = None
        self.pf_canvas.reset()
        self.pf_step_card.clear()
        self.pf_line_acc.set_data([],[]); self.pf_line_f1.set_data([],[])
        self.pf_line_auc.set_data([],[])
        self.pf_vline_acc.set_alpha(0.0); self.pf_vline_auc.set_alpha(0.0)
        self.pf_ann_r8_acc.set_alpha(0.0); self.pf_ann_r8_auc.set_alpha(0.0)
        self.pf_canvas_widget.draw_idle()
        for key in ("acc","f1","auc"):
            self.pf_metric_vars[key].set("—")
            cell, lbl = self.pf_metric_frames[key]
            lbl.config(fg="#8FA8D0"); cell.config(bg=C_CARD)
        try: self.pf_best_banner.pack_forget()
        except: pass
        self.pf_lbl_round.config(text="Pre-Training Baseline")
        self.pf_lbl_phase.config(text="Global model initialised. No FL rounds applied yet.")
        self._pf_update_display(0)

    def _pf_update_display(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        self.pf_metric_vars["acc"].set(f"{rd[1]*100:.2f}%")
        self.pf_metric_vars["f1"].set(f"{rd[2]*100:.2f}%")
        self.pf_metric_vars["auc"].set(f"{rd[3]:.4f}")


    # ══════════════════════════════════════════════════════════
    # TAB 3 — Comparison (Enhanced vs Plain)
    # ══════════════════════════════════════════════════════════
    def _build_comparison_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Comparison  ")

        top = tk.Frame(frame, bg=C_PANEL, pady=9)
        top.pack(fill=tk.X)
        tk.Label(top, text="Enhanced (Reputation-Weighted) vs Plain FedAvg (Baseline)",
                 bg=C_PANEL, fg=C_TEXT, font=("Helvetica",12,"bold")).pack()

        body = tk.Frame(frame, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ── Left: pipeline step-count comparison ──────────────
        left = tk.Frame(body, bg=C_CARD2, padx=14, pady=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,6))
        tk.Label(left, text="PIPELINE STEPS", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(anchor="w", pady=(0,8))

        enh_row = tk.Frame(left, bg=C_CARD2)
        enh_row.pack(fill=tk.X, pady=(0,4))
        tk.Label(enh_row, text="Enhanced (6 steps):", bg=C_CARD2, fg=C_TEXT,
                 font=("Helvetica",9,"bold"), width=18, anchor="w").pack(side=tk.LEFT)
        for key, label in STEPS_DEF:
            tk.Label(enh_row, text=label.strip(), bg=STEP_COLORS.get(key,"#888"),
                     fg="#111", font=("Helvetica",8,"bold"), padx=6, pady=2
                     ).pack(side=tk.LEFT, padx=2)

        plain_row = tk.Frame(left, bg=C_CARD2)
        plain_row.pack(fill=tk.X, pady=(4,10))
        tk.Label(plain_row, text="Plain (4 steps):", bg=C_CARD2, fg=C_TEXT,
                 font=("Helvetica",9,"bold"), width=18, anchor="w").pack(side=tk.LEFT)
        for key, label in PLAIN_STEPS_DEF:
            tk.Label(plain_row, text=label.strip(), bg=PLAIN_STEP_COLORS.get(key,"#888"),
                     fg="#111", font=("Helvetica",8,"bold"), padx=6, pady=2
                     ).pack(side=tk.LEFT, padx=2)

        tk.Label(left, text="Enhanced adds Validate (safety checks) and Reputation "
                             "(ledger + decay) — steps that plain FedAvg skips entirely. "
                             "Enhanced also selects by score; plain selects at random, "
                             "and enhanced weights contributions by quality where plain "
                             "averages every update equally.",
                 bg=C_CARD2, fg=C_TEXT2, font=("Helvetica",9), wraplength=420,
                 justify="left", anchor="w").pack(fill=tk.X, pady=(4,10))

        # Feature comparison table
        table = tk.Frame(left, bg=C_CARD2)
        table.pack(fill=tk.X, pady=(4,0))
        rows = [
            ("Selection", "Score-based (Vᵢ·Hᵢ·Rᵢ)", "Uniform random"),
            ("Validation gate", "L2 norm + gain test", "None"),
            ("Aggregation", "Weighted by score", "Equal mean (1/3 each)"),
            ("Reputation ledger", "Yes, 0.99 decay/round", "None"),
            ("Best round", "Round 6", "Round 8"),
            ("Best Accuracy", "96.52%", "94.42%"),
            ("Best F1", "96.52%", "94.58%"),
            ("Best ROC-AUC", "0.9964", "0.9865"),
        ]
        hdr = tk.Frame(table, bg=C_CARD)
        hdr.pack(fill=tk.X)
        for i, h in enumerate(["Aspect","Enhanced","Plain"]):
            tk.Label(hdr, text=h, bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8,"bold"),
                     width=[16,20,20][i], anchor="w", padx=4, pady=3
                     ).grid(row=0, column=i, sticky="w")
        for r, (a, e, p) in enumerate(rows):
            rowbg = C_CARD2 if r % 2 == 0 else "#33364E"
            rf = tk.Frame(table, bg=rowbg)
            rf.pack(fill=tk.X)
            tk.Label(rf, text=a, bg=rowbg, fg=C_TEXT, font=("Helvetica",8),
                     width=16, anchor="w", padx=4, pady=2).grid(row=0, column=0, sticky="w")
            tk.Label(rf, text=e, bg=rowbg, fg=C_GOLD, font=("Helvetica",8,"bold"),
                     width=20, anchor="w", padx=4, pady=2).grid(row=0, column=1, sticky="w")
            tk.Label(rf, text=p, bg=rowbg, fg="#B0B8D0", font=("Helvetica",8,"bold"),
                     width=20, anchor="w", padx=4, pady=2).grid(row=0, column=2, sticky="w")

        # ── Right: overlaid metric charts ──────────────────────
        right = tk.Frame(body, bg=C_CARD2, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6,0))
        tk.Label(right, text="METRICS OVERLAY (ALL 10 ROUNDS)", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(anchor="w", pady=(0,6))
        self._build_cmp_chart(right)


    def _build_cmp_chart(self, parent):
        fig_bg = C_BG; ax_bg = "#22243A"
        fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.6), facecolor=fig_bg,
                                  gridspec_kw={"hspace":0.55})
        fig.patch.set_facecolor(fig_bg)
        for ax in axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = axes
        enh_x = [rd[0] for rd in ROUND_DATA]
        enh_acc = [rd[1] for rd in ROUND_DATA]
        enh_auc = [rd[3] for rd in ROUND_DATA]
        plain_x = [rd[0] for rd in PLAIN_ROUND_DATA_FILLED]
        plain_acc = [rd[1] for rd in PLAIN_ROUND_DATA_FILLED]
        plain_auc = [rd[3] for rd in PLAIN_ROUND_DATA_FILLED]

        ax_acc.set_title("Accuracy per Round", color=C_TEXT, fontsize=9,
                          pad=5, fontweight="bold")
        ax_acc.plot(enh_x, enh_acc, color=C_GOLD, linewidth=2.0, marker="o",
                    markersize=4, label="Enhanced", zorder=3)
        ax_acc.plot(plain_x, plain_acc, color="#8FA8D0", linewidth=2.0,
                    linestyle="--", marker="s", markersize=4, label="Plain", zorder=3)
        ax_acc.axvline(x=6, color=C_GOLD, linewidth=1.2, linestyle=":", alpha=0.6)
        ax_acc.axvline(x=8, color="#8FA8D0", linewidth=1.2, linestyle=":", alpha=0.6)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)], color=C_TEXT2, fontsize=6)
        ax_acc.set_ylabel("Accuracy", color=C_TEXT2, fontsize=7)
        ax_acc.legend(fontsize=7, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT, fontsize=9,
                          pad=5, fontweight="bold")
        ax_auc.plot(enh_x, enh_auc, color=C_GOLD, linewidth=2.0, marker="o",
                    markersize=4, label="Enhanced", zorder=3)
        ax_auc.plot(plain_x, plain_auc, color="#8FD0B0", linewidth=2.0,
                    linestyle="--", marker="s", markersize=4, label="Plain", zorder=3)
        ax_auc.axvline(x=6, color=C_GOLD, linewidth=1.2, linestyle=":", alpha=0.6)
        ax_auc.axvline(x=8, color="#8FD0B0", linewidth=1.2, linestyle=":", alpha=0.6)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)], color=C_TEXT2, fontsize=6)
        ax_auc.set_ylabel("ROC-AUC", color=C_TEXT2, fontsize=7)
        ax_auc.legend(fontsize=7, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        canvas_widget = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_widget.draw()


    def _build_sync_playback_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Synced Playback  ")

        top = tk.Frame(frame, bg=C_PANEL, pady=9)
        top.pack(fill=tk.X)
        tk.Label(top, text="Side-by-Side Process Playback — Same Round, Both Pipelines",
                 bg=C_PANEL, fg=C_TEXT, font=("Helvetica",12,"bold")).pack()

        # ── Synced Side-by-Side Process Playback ───────────────
        sync_outer = tk.Frame(frame, bg=C_PANEL, pady=8, padx=10)
        sync_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        sync_hdr = tk.Frame(sync_outer, bg=C_PANEL)
        sync_hdr.pack(fill=tk.X, pady=(0,6))
        tk.Label(sync_hdr, text="SYNCED PROCESS PLAYBACK", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(side=tk.LEFT)
        self.cmp_lbl_round = tk.Label(sync_hdr, text="Pre-Training", bg=C_PANEL,
                                       fg=C_TEXT, font=("Helvetica",9,"bold"))
        self.cmp_lbl_round.pack(side=tk.LEFT, padx=(14,0))

        sync_ctrl = tk.Frame(sync_outer, bg=C_PANEL)
        sync_ctrl.pack(fill=tk.X, pady=(0,6))
        self.cmp_btn_play = tk.Button(sync_ctrl, text="▶  Play Round (Both)",
                                       command=self._cmp_play_round,
                                       bg=C_ACCENT, fg="white",
                                       font=("Helvetica",10,"bold"),
                                       relief=tk.FLAT, padx=10, pady=5)
        self.cmp_btn_play.pack(side=tk.LEFT, padx=(0,5))
        tk.Button(sync_ctrl, text="⟳  Reset", command=self._cmp_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)
        self.cmp_lbl_status = tk.Label(sync_ctrl, text="", bg=C_PANEL, fg=C_TEXT2,
                                        font=("Helvetica",8,"italic"))
        self.cmp_lbl_status.pack(side=tk.LEFT, padx=(12,0))

        dual = tk.Frame(sync_outer, bg=C_PANEL)
        dual.pack(fill=tk.BOTH, expand=True)

        left_host = tk.Frame(dual, bg=CV_BG)
        left_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,4))
        tk.Label(left_host, text="ENHANCED — 6 steps", bg="#14142A", fg=C_GOLD,
                 font=("Helvetica",9,"bold"), pady=3).pack(fill=tk.X)
        self.cmp_enh_canvas = NetworkCanvas(left_host, width=560, height=380)

        right_host = tk.Frame(dual, bg=CV_BG)
        right_host.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4,0))
        tk.Label(right_host, text="PLAIN FEDAVG — 4 steps", bg="#14142A", fg="#B0B8D0",
                 font=("Helvetica",9,"bold"), pady=3).pack(fill=tk.X)
        self.cmp_pf_canvas = SimpleNetworkCanvas(right_host, width=560, height=380)

        self.cmp_round_index = 0
        self.cmp_animating = False
        self.cmp_enh_done = False
        self.cmp_pf_done = False



    def _cmp_play_round(self):
        if self.cmp_animating:
            return
        next_idx = self.cmp_round_index + 1
        max_idx = min(len(ROUND_DATA), len(PLAIN_ROUND_DATA_FILLED)) - 1
        if next_idx > max_idx:
            self.cmp_lbl_status.config(text="All synced rounds complete.")
            self.cmp_btn_play.config(state=tk.DISABLED)
            return
        self.cmp_animating = True
        self.cmp_enh_done = False
        self.cmp_pf_done = False
        self.cmp_btn_play.config(state=tk.DISABLED)
        self.cmp_lbl_round.config(text=f"Round {next_idx} / 10  (same round, both sides)")
        self.cmp_lbl_status.config(text="Playing both pipelines for the same round…")

        enh_rd = ROUND_DATA[next_idx]
        enh_selected = enh_rd[6]
        enh_rnd = enh_rd[0]

        pf_rd = PLAIN_ROUND_DATA_FILLED[next_idx]
        pf_selected = pf_rd[5]
        pf_local_accs = pf_rd[6]
        pf_rnd = pf_rd[0]

        def enh_finished():
            self.cmp_enh_canvas.round_done_reset()
            self.cmp_enh_canvas.clear_breadcrumb()
            self.cmp_enh_done = True
            self._cmp_check_both_done(next_idx)

        def pf_finished():
            self.cmp_pf_canvas.round_done_reset()
            self.cmp_pf_canvas.clear_breadcrumb()
            self.cmp_pf_done = True
            self._cmp_check_both_done(next_idx)

        def enh_select():
            self.cmp_enh_canvas.animate_select(enh_selected, enh_rnd, enh_train)
        def enh_train():
            self.cmp_enh_canvas.animate_train(enh_selected, enh_send)
        def enh_send():
            self.cmp_enh_canvas.animate_send(enh_selected, enh_validate)
        def enh_validate():
            self.cmp_enh_canvas.animate_validate(enh_selected, enh_aggregate)
        def enh_aggregate():
            self.cmp_enh_canvas.animate_aggregate(enh_selected, enh_rep)
        def enh_rep():
            self.cmp_enh_canvas.animate_reputation(enh_selected, enh_rnd, enh_finished)

        def pf_select():
            self.cmp_pf_canvas.animate_select(pf_selected, pf_rnd, pf_train)
        def pf_train():
            self.cmp_pf_canvas.animate_train(pf_selected, pf_local_accs, pf_send)
        def pf_send():
            self.cmp_pf_canvas.animate_send(pf_selected, pf_average)
        def pf_average():
            self.cmp_pf_canvas.animate_average(pf_selected, pf_finished)

        # Start both pipelines at the same moment — Plain will naturally
        # finish first since it has fewer steps, visually showing the gap.
        self.root.after(0, enh_select)
        self.root.after(0, pf_select)

    def _cmp_check_both_done(self, idx):
        if self.cmp_enh_done and self.cmp_pf_done:
            self.cmp_round_index = idx
            self.cmp_animating = False
            self.cmp_btn_play.config(state=tk.NORMAL)
            self.cmp_lbl_status.config(
                text="Both sides finished this round. Notice Plain reached "
                     "'done' after only 4 steps.")

    def _cmp_reset(self):
        self.cmp_round_index = 0
        self.cmp_animating = False
        self.cmp_enh_done = False
        self.cmp_pf_done = False
        self.cmp_enh_canvas.reset()
        self.cmp_pf_canvas.reset()
        self.cmp_btn_play.config(state=tk.NORMAL)
        self.cmp_lbl_round.config(text="Pre-Training")
        self.cmp_lbl_status.config(text="")


    # ── Detection helpers ─────────────────────────────────────
    @staticmethod
    def _sep(p):
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

    def _det_show_placeholder(self):
        img = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), dtype=np.uint8)
        img[:] = (30, 32, 50)
        cv2.putText(img,"No video loaded",
                    (DISPLAY_MAX_W//2-140,DISPLAY_MAX_H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(100,110,160),2)
        cv2.putText(img,"Click  Open Video  to begin",
                    (DISPLAY_MAX_W//2-200,DISPLAY_MAX_H//2+36),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(70,80,120),1)
        self._det_render(img)

    def _det_open_file(self):
        p = filedialog.askopenfilename(title="Select a video file",
                                        filetypes=VIDEO_EXTS)
        if p: self._det_load_video(p)

    def _det_browse_model(self):
        """Let the user pick any .tflite model file from disk and register it
        in the dropdown, without having to edit any code."""
        p = filedialog.askopenfilename(
            title="Select a TFLite model to load",
            filetypes=[("TFLite model","*.tflite"),("All files","*.*")],
            initialdir=str(Path(self.model_path).parent))
        if not p:
            return
        name = Path(p).name
        # Avoid clobbering an existing entry with the same filename but a
        # different path — disambiguate by appending a counter if needed.
        base_name = name
        i = 2
        while name in self.known_models and self.known_models[name] != p:
            name = f"{base_name} ({i})"
            i += 1
        self.known_models[name] = p
        self.det_model_combo.config(values=list(self.known_models.keys()))
        self.det_model_var.set(name)
        self._det_load_model(name)

    def _det_on_model_selected(self, event=None):
        name = self.det_model_var.get()
        self._det_load_model(name)

    def _det_load_model(self, name):
        """Swap the active TFLite model at runtime. The interpreter build
        (allocate_tensors etc.) can take several seconds for larger models,
        so it is done on a background thread to avoid freezing the UI.
        Already-loaded interpreters are cached so re-selecting a model you
        already loaded this session is instant."""
        path = self.known_models.get(name)
        if not path:
            return
        if name == self.active_model_name:
            return
        if self.det_model_loading:
            # A load is already in flight — ignore extra clicks/selections
            # rather than queuing overlapping background loads.
            self.det_model_var.set(self.active_model_name)
            return

        cached = self.model_cache.get(path)
        if cached is not None:
            self._det_apply_loaded_model(name, path, cached)
            return

        self.det_model_loading = True
        self.det_model_combo.config(state="disabled")
        self.det_model_status.config(text="⏳ Loading…", fg=C_TEXT2)

        def worker():
            try:
                result = load_tflite_model(path)
                error = None
            except Exception as e:
                result = None
                error = e
            self.root.after(0, lambda: self._det_on_model_loaded(name, path, result, error))

        threading.Thread(target=worker, daemon=True).start()

    def _det_on_model_loaded(self, name, path, result, error):
        self.det_model_loading = False
        self.det_model_combo.config(state="readonly")
        if error is not None:
            messagebox.showerror("Model Load Failed",
                                  f"Could not load model:\n{path}\n\n{error}")
            self.det_model_var.set(self.active_model_name)
            self.det_model_status.config(text="")
            return
        self.model_cache[path] = result
        self._det_apply_loaded_model(name, path, result)

    def _det_apply_loaded_model(self, name, path, loaded):
        new_interp, new_inp, new_out = loaded
        with self.model_lock:
            self.interpreter = new_interp
            self.inp = new_inp
            self.out = new_out
            self.inp_dtype = self.inp["dtype"]
            self.inp_scale, self.inp_zp = self.inp.get("quantization",(1.0,0))
            self.out_scale, self.out_zp = self.out.get("quantization",(1.0,0))
            self.model_path = path
            self.active_model_name = name

        # Reset the smoothing history so old-model predictions don't bleed
        # into the new model's rolling average, and clear any prior error
        # flag so a fresh model gets a fresh chance to report problems.
        self.det_history.clear()
        self.det_infer_error_shown = False
        self.det_model_status.config(text="✓ Switched", fg=C_GREEN)
        self.root.after(1500, lambda: self.det_model_status.config(text=""))

    def _det_load_video(self, path):
        self._det_cancel_loop(); self._det_stop()
        if self.det_cap: self.det_cap.release()
        self.det_cap = cv2.VideoCapture(path)
        if not self.det_cap.isOpened():
            messagebox.showerror("Error",f"Cannot open video:\n{path}"); return
        self.det_video_path=path
        self.det_total_frames=int(self.det_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.det_video_fps=self.det_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.det_frame_idx=0; self.det_history.clear()
        self.det_raw_prob=0.5; self.det_smoothed=0.5
        self.det_label="NO FACE"; self.det_cv_color=COLOR_NO_FACE
        self.det_hex_color=C_NEUTRAL; self.det_confidence=0.0
        self.det_inf_ms=0.0; self.det_bbox=None
        self.det_verdict_counts={"REAL":0,"FAKE":0}
        self.det_frame_log=[]; self.det_last_det=None
        self.det_fps_display=0.0; self.det_t_prev=time.time()
        vw=int(self.det_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh=int(self.det_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sc=min(DISPLAY_MAX_W/vw,DISPLAY_MAX_H/vh,1.0)
        self.det_display_w=int(vw*sc); self.det_display_h=int(vh*sc)
        self.det_vid_w=vw; self.det_vid_h=vh
        self.det_progress_bar.config(to=self.det_total_frames)
        fname=Path(path).name
        self.det_lbl_status.config(
            text=f"{fname}  |  {vw}×{vh}  |  {self.det_video_fps:.1f} fps  |  {self.det_total_frames} frames")
        self.root.title(f"Enhanced FL Demo — {fname}")
        self._det_update_overall_verdict()
        ret,first=self.det_cap.read()
        if ret: self._det_render_and_show(first)
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.det_after_id=self.root.after(33,self._det_update_loop)

    def _det_start_inference_thread(self):
        self.det_infer_result={
            "raw_prob":0.5,"smoothed":0.5,"label":"NO FACE",
            "cv_color":COLOR_NO_FACE,"hex_color":C_NEUTRAL,
            "confidence":0.0,"inf_ms":0.0,"bbox":None}
        def worker():
            while not self.det_stop_event.is_set():
                crop=bb=None
                with self.det_infer_queue_lock:
                    if self.det_infer_queue:
                        crop,bb=self.det_infer_queue.pop()
                        self.det_infer_queue.clear()
                if crop is None: time.sleep(0.005); continue
                # Snapshot the active model under lock so a live model swap
                # (triggered from the UI thread) can never be read half-updated.
                with self.model_lock:
                    interp, inp, out = self.interpreter, self.inp, self.out
                    dtype, scale, zp = self.inp_dtype, self.inp_scale, self.inp_zp
                    o_scale, o_zp = self.out_scale, self.out_zp
                try:
                    t0=time.perf_counter()
                    arr=preprocess_face(crop,dtype,scale,zp)
                    interp.set_tensor(inp["index"],arr)
                    interp.invoke()
                    ro=interp.get_tensor(out["index"])
                    rp=(float((ro[0][0]-o_zp)*o_scale)
                        if dtype in (np.uint8, np.int8) else float(ro[0][0]))
                    ms=(time.perf_counter()-t0)*1000
                    self.det_history.append(rp)
                    sm=float(np.mean(self.det_history))
                    lb,cc,hc,cf=classify(sm)
                    with self.det_infer_lock:
                        self.det_infer_result.update({
                            "raw_prob":rp,"smoothed":sm,"label":lb,
                            "cv_color":cc,"hex_color":hc,"confidence":cf,
                            "inf_ms":ms,"bbox":bb})
                except Exception as e:
                    # Never let a bad frame/model mismatch silently kill this
                    # thread — surface the error once and keep the loop alive
                    # so a subsequent model swap or frame can recover.
                    with self.det_infer_lock:
                        self.det_infer_result["label"] = "INFER ERROR"
                    if not self.det_infer_error_shown:
                        self.det_infer_error_shown = True
                        err_text = str(e)
                        self.root.after(0, lambda: messagebox.showerror(
                            "Inference Error",
                            f"The active model raised an error during inference:\n\n{err_text}"))
        threading.Thread(target=worker,daemon=True).start()

    def _det_cancel_loop(self):
        if self.det_after_id:
            try: self.root.after_cancel(self.det_after_id)
            except: pass
            self.det_after_id=None

    def _det_update_loop(self):
        if self.det_playing and self.det_cap and self.det_cap.isOpened():
            if self.det_seek_pending is not None:
                self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,self.det_seek_pending)
                self.det_history.clear(); self.det_seek_pending=None
            ret,frame=self.det_cap.read()
            if not ret:
                self.det_playing=False; self.det_btn_play.config(text="▶")
                self.det_lbl_status.config(text="Playback complete.")
                self._det_update_overall_verdict()
            else:
                self.det_frame_idx=int(self.det_cap.get(cv2.CAP_PROP_POS_FRAMES))
                tn=time.time()
                self.det_fps_display=(0.9*self.det_fps_display+
                                      0.1/(max(tn-self.det_t_prev,1e-6)))
                self.det_t_prev=tn
                if self.det_frame_idx%FRAME_SKIP==0:
                    sm=cv2.resize(frame,(DETECT_W,DETECT_H),
                                  interpolation=cv2.INTER_LINEAR)
                    results=self.face_detector.process(
                        cv2.cvtColor(sm,cv2.COLOR_BGR2RGB))
                    self.det_last_det=results
                else:
                    results=self.det_last_det
                dsx=self.det_vid_w/DETECT_W; dsy=self.det_vid_h/DETECT_H
                if results and results.detections:
                    lg=max(results.detections,
                           key=lambda d:(d.location_data.relative_bounding_box.width*
                                         d.location_data.relative_bounding_box.height))
                    rb=lg.location_data.relative_bounding_box
                    x1=int(rb.xmin*DETECT_W*dsx); y1=int(rb.ymin*DETECT_H*dsy)
                    bw=int(rb.width*DETECT_W*dsx); bh=int(rb.height*DETECT_H*dsy)
                    px=int(bw*FACE_PADDING); py=int(bh*FACE_PADDING)
                    x1=max(0,x1-px); y1=max(0,y1-py)
                    x2=min(self.det_vid_w,x1+bw+2*px)
                    y2=min(self.det_vid_h,y1+bh+2*py)
                    if x2>x1 and y2>y1:
                        with self.det_infer_queue_lock:
                            self.det_infer_queue.clear()
                            self.det_infer_queue.append(
                                (frame[y1:y2,x1:x2].copy(),(x1,y1,x2,y2)))
                with self.det_infer_lock: res=dict(self.det_infer_result)
                nf=not(results and results.detections)
                lb="NO FACE" if nf else res["label"]
                cc=COLOR_NO_FACE if nf else res["cv_color"]
                hc=C_NEUTRAL if nf else res["hex_color"]
                cf=0.0 if nf else res["confidence"]
                rp=res["raw_prob"]; sm=res["smoothed"]; ms=res["inf_ms"]
                bb=None if nf else res["bbox"]
                if not nf and lb in self.det_verdict_counts:
                    self.det_verdict_counts[lb]+=1
                if len(self.det_frame_log)<50_000:
                    self.det_frame_log.append({
                        "frame":self.det_frame_idx,"face_found":not nf,
                        "raw_prob":round(rp,6),"smoothed":round(sm,6),
                        "label":lb,"confidence":round(cf,2),"inf_ms":round(ms,2)})
                self._det_render_and_show(frame,bb,lb,cc,cf,rp,sm,ms)
                self._det_update_stats(lb,hc,cf,rp,sm,ms)
                self.det_progress_var.set(self.det_frame_idx)
                el=self.det_frame_idx/max(self.det_video_fps,1)
                ts=self.det_total_frames/max(self.det_video_fps,1)
                self.det_lbl_time.config(
                    text=f"{self._fmt_time(el)} / {self._fmt_time(ts)}")
                dl=max(1,int((1000/self.det_video_fps)/self.det_speed))
                self.det_after_id=self.root.after(dl,self._det_update_loop)
                return
        if self.det_loop_active:
            self.det_after_id=self.root.after(33,self._det_update_loop)

    def _det_render(self,frame_bgr):
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        pil=Image.fromarray(rgb)
        imgtk=ImageTk.PhotoImage(image=pil)
        self.det_canvas.imgtk=imgtk
        self.det_canvas.config(image=imgtk)

    def _det_render_and_show(self,frame_bgr,bbox=None,label="",
                              cv_color=COLOR_NO_FACE,confidence=0.0,
                              raw_prob=0.5,smoothed=0.5,inf_ms=0.0):
        frame=frame_bgr.copy() if bbox else frame_bgr
        if bbox:
            x1,y1,x2,y2=bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),cv_color,2)
            tag=f"{label}  {confidence:.1f}%"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+8,y1),cv_color,-1)
            cv2.putText(frame,tag,(x1+4,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0),2)
            bw=x2-x1; fl=int(bw*min(confidence,100)/100)
            cv2.rectangle(frame,(x1,y2+4),(x2,y2+16),(40,40,40),-1)
            cv2.rectangle(frame,(x1,y2+4),(x1+fl,y2+16),cv_color,-1)
        disp=cv2.resize(frame,(self.det_display_w,self.det_display_h),
                        interpolation=cv2.INTER_LINEAR)
        self._det_render(disp)

    def _det_update_stats(self,label,hc,cf,rp,sm,ms):
        self.det_stat_vars["frame"].set(f"{self.det_frame_idx} / {self.det_total_frames}")
        self.det_stat_vars["fps"].set(f"{self.det_fps_display:.1f}")
        self.det_stat_vars["inf_ms"].set(f"{ms:.1f} ms")
        self.det_stat_vars["raw"].set(f"{rp:.4f}")
        self.det_stat_vars["smooth"].set(f"{sm:.4f}")
        total=sum(self.det_verdict_counts.values())
        for v,var in self.det_tally_vars.items():
            n=self.det_verdict_counts[v]
            p=n/total*100 if total>0 else 0
            var.set(f"{n}  ({p:.1f}%)")
        self._det_update_overall_verdict()

    def _det_update_overall_verdict(self):
        total=sum(self.det_verdict_counts.values())
        if total==0:
            self.det_lbl_overall.config(text="—",fg=C_NEUTRAL)
            self.det_lbl_overall_sub.config(text="No inference yet."); return
        dom=max(self.det_verdict_counts,key=self.det_verdict_counts.get)
        pct=self.det_verdict_counts[dom]/total*100
        hm={"REAL":C_REAL,"FAKE":C_FAKE}
        self.det_lbl_overall.config(text=dom,fg=hm[dom])
        self.det_lbl_overall_sub.config(
            text=f"{dom} in {pct:.1f}% of\n{total} inferred frames")

    def _det_toggle_play(self):
        if self.det_cap is None: self._det_open_file(); return
        self.det_playing=not self.det_playing
        self.det_btn_play.config(text="⏸" if self.det_playing else "▶")
        if self.det_playing: self.det_t_prev=time.time()

    def _det_stop(self):
        self.det_playing=False; self.det_btn_play.config(text="▶")
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.det_frame_idx=0; self.det_progress_var.set(0)
            self.det_lbl_time.config(text="0:00 / 0:00")

    def _det_restart(self):
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.det_frame_idx=0; self.det_history.clear()
            self.det_verdict_counts={"REAL":0,"FAKE":0}
            self.det_frame_log=[]; self.det_progress_var.set(0)
            self._det_update_overall_verdict()
            self.det_playing=True; self.det_btn_play.config(text="⏸")
            self.det_t_prev=time.time()

    def _det_on_seek(self,val):
        if self.det_cap: self.det_seek_pending=int(float(val))

    def _det_on_speed_change(self,event=None):
        val=self.det_speed_var.get().replace("×","")
        try: self.det_speed=float(val)
        except: self.det_speed=1.0

    def _det_export_csv(self):
        if not self.det_frame_log:
            messagebox.showinfo("Export","No inference data yet.\nPlay the video first."); return
        path=filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"),("All files","*.*")],
            initialfile="deepfake_results.csv")
        if not path: return
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=self.det_frame_log[0].keys())
            w.writeheader(); w.writerows(self.det_frame_log)
        messagebox.showinfo("Exported",f"Saved {len(self.det_frame_log)} rows to:\n{path}")

    def _det_save_frame(self):
        if self.det_cap is None: return
        pos=int(self.det_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,pos-1))
        ret,frame=self.det_cap.read()
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        if not ret: return
        path=filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files","*.png"),("All files","*.*")],
            initialfile=f"frame_{self.det_frame_idx:05d}.png")
        if path:
            cv2.imwrite(path,frame)
            messagebox.showinfo("Saved",f"Frame saved to:\n{path}")

    def _build_plain_detection_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Plain FedAvg Detector  ")

        tb = tk.Frame(frame, bg=C_PANEL, pady=7)
        tb.pack(fill=tk.X)
        tk.Button(tb, text="📂  Open Video", command=self._pfdet_open_file,
                  bg=C_ACCENT, fg="white", font=("Helvetica",11,"bold"),
                  relief=tk.FLAT, padx=14, pady=5).pack(side=tk.LEFT, padx=(12,6))
        tk.Button(tb, text="💾  Export CSV", command=self._pfdet_export_csv,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="🖼  Save Frame", command=self._pfdet_save_frame,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        badge = tk.Frame(tb, bg=C_CARD2, padx=10, pady=4)
        badge.pack(side=tk.RIGHT, padx=12)
        tk.Label(badge, text="Model:", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",8)).pack(side=tk.LEFT)
        self.pfdet_model_var = tk.StringVar(value=self.pf_active_model_name)
        self.pfdet_model_combo = ttk.Combobox(
            badge, textvariable=self.pfdet_model_var,
            values=list(self.pf_known_models.keys()),
            width=26, state="readonly", font=("Helvetica",8))
        self.pfdet_model_combo.pack(side=tk.LEFT, padx=(4,4))
        self.pfdet_model_combo.bind("<<ComboboxSelected>>", self._pfdet_on_model_selected)
        tk.Button(badge, text="📁 Load Model…", command=self._pfdet_browse_model,
                  bg=C_ACCENT, fg="white", font=("Helvetica",8,"bold"),
                  relief=tk.FLAT, padx=6, pady=2).pack(side=tk.LEFT)
        self.pfdet_model_status = tk.Label(badge, text="", bg=C_CARD2, fg=C_GREEN,
                                          font=("Helvetica",8,"bold"))
        self.pfdet_model_status.pack(side=tk.LEFT, padx=(6,0))

        main = tk.Frame(frame, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8,0))

        self.pfdet_canvas = tk.Label(main, bg="#0A0A0A",
                                    width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.pfdet_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sp = tk.Frame(main, bg=C_PANEL, width=220)
        sp.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))
        sp.pack_propagate(False)

        # ── Overall Verdict — dominant element at top ────────────
        tk.Label(sp, text="OVERALL VERDICT", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(14,2))
        self.pfdet_lbl_overall = tk.Label(sp, text="—", bg=C_PANEL,
                                         fg=C_NEUTRAL, font=("Helvetica",32,"bold"))
        self.pfdet_lbl_overall.pack(pady=(2,0))
        self.pfdet_lbl_overall_sub = tk.Label(sp, text="", bg=C_PANEL,
                                             fg=C_TEXT2, font=("Helvetica",9),
                                             wraplength=200, justify="center")
        self.pfdet_lbl_overall_sub.pack(padx=8, pady=(0,6))
        self._sep(sp)

        tk.Label(sp, text="INFERENCE STATS", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.pfdet_stat_vars = {}
        for key, lbl in [("frame","Frame"),("fps","Display FPS"),
                          ("inf_ms","Inference ms"),("raw","Raw prob"),
                          ("smooth","Smoothed prob")]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=lbl+":", bg=C_PANEL, fg=C_TEXT2,
                     font=("Helvetica",9), anchor="w", width=14).pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            self.pfdet_stat_vars[key] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9,"bold"), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        tk.Label(sp, text="FRAME TALLY", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.pfdet_tally_vars = {}
        for verdict, hc in [("REAL",C_REAL),("FAKE",C_FAKE)]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=verdict+":", bg=C_PANEL, fg=hc,
                     font=("Helvetica",9,"bold"), anchor="w",
                     width=10).pack(side=tk.LEFT)
            v = tk.StringVar(value="0  (0.0%)")
            self.pfdet_tally_vars[verdict] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        # (Overall Verdict shown at top of panel)

        pf = tk.Frame(frame, bg=C_BG, pady=4)
        pf.pack(fill=tk.X, padx=10)
        self.pfdet_progress_var = tk.DoubleVar(value=0)
        self.pfdet_progress_bar = ttk.Scale(pf, from_=0, to=100,
                                           orient="horizontal",
                                           variable=self.pfdet_progress_var,
                                           command=self._pfdet_on_seek)
        self.pfdet_progress_bar.pack(fill=tk.X)
        self.pfdet_lbl_time = tk.Label(pf, text="0:00 / 0:00",
                                      bg=C_BG, fg=C_TEXT2,
                                      font=("Helvetica",9))
        self.pfdet_lbl_time.pack(anchor="e")

        cf = tk.Frame(frame, bg=C_PANEL, pady=8)
        cf.pack(fill=tk.X, side=tk.BOTTOM)
        bc = dict(bg=C_CARD, fg=C_TEXT, font=("Helvetica",12),
                  relief=tk.FLAT, width=4, pady=4)
        self.pfdet_btn_play = tk.Button(cf, text="▶",
                                       command=self._pfdet_toggle_play, **bc)
        self.pfdet_btn_play.pack(side=tk.LEFT, padx=(14,2))
        tk.Button(cf, text="⏹", command=self._pfdet_stop, **bc).pack(side=tk.LEFT, padx=2)
        tk.Button(cf, text="⟳", command=self._pfdet_restart, **bc).pack(side=tk.LEFT, padx=2)
        tk.Label(cf, text="Speed:", bg=C_PANEL, fg=C_TEXT,
                 font=("Helvetica",10)).pack(side=tk.LEFT, padx=(16,4))
        self.pfdet_speed_var = tk.StringVar(value="1.0×")
        sm = ttk.Combobox(cf, textvariable=self.pfdet_speed_var,
                          values=[f"{s}×" for s in SPEED_OPTIONS],
                          width=6, state="readonly")
        sm.pack(side=tk.LEFT, padx=4)
        sm.bind("<<ComboboxSelected>>", self._pfdet_on_speed_change)
        self.pfdet_lbl_status = tk.Label(
            cf, text="Open a video file to begin.  [Space=play/pause · Esc=stop · R=restart · S=save frame]",
            bg=C_PANEL, fg=C_TEXT2, font=("Helvetica",9))
        self.pfdet_lbl_status.pack(side=tk.LEFT, padx=16)

        self.pfdet_cap=None; self.pfdet_video_path=None
        self.pfdet_total_frames=0; self.pfdet_video_fps=30.0
        self.pfdet_frame_idx=0; self.pfdet_playing=False
        self.pfdet_speed=1.0; self.pfdet_seek_pending=None
        self.pfdet_after_id=None; self.pfdet_loop_active=True
        self.pfdet_history=collections.deque(maxlen=HISTORY_SIZE)
        self.pfdet_raw_prob=0.5; self.pfdet_smoothed=0.5
        self.pfdet_label="NO FACE"; self.pfdet_cv_color=COLOR_NO_FACE
        self.pfdet_hex_color=C_NEUTRAL; self.pfdet_confidence=0.0
        self.pfdet_inf_ms=0.0; self.pfdet_bbox=None
        self.pfdet_verdict_counts={"REAL":0,"FAKE":0}
        self.pfdet_frame_log=[]; self.pfdet_last_det=None
        self.pfdet_fps_display=0.0; self.pfdet_t_prev=time.time()
        self.pfdet_vid_w=DISPLAY_MAX_W; self.pfdet_vid_h=DISPLAY_MAX_H
        self.pfdet_display_w=DISPLAY_MAX_W; self.pfdet_display_h=DISPLAY_MAX_H
        self.pfdet_infer_lock=threading.Lock()
        self.pfdet_infer_queue=[]; self.pfdet_infer_queue_lock=threading.Lock()
        self.pfdet_stop_event=threading.Event()
        self.pfdet_infer_error_shown=False

        self._pfdet_show_placeholder()
        self._pfdet_start_inference_thread()
        self.pfdet_after_id = self.root.after(33, self._pfdet_update_loop)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — FL Simulation
    # New layout:
    #   left (canvas 940×560 + ledger)  |  centre (step card)  |  right (metrics + chart)
    # ══════════════════════════════════════════════════════════
    def _build_fl_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  FL Simulation  ")

        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_mode   = tk.BooleanVar(value=False)
        self.fl_auto_paused = False
        self.fl_auto_after  = None
        self.fl_step_pending_cb = None  # callback waiting for Next Step press

        # ── Three-column layout ───────────────────────────────
        left = tk.Frame(frame, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,4), pady=6)

        # Centre column: step card + controls
        centre = tk.Frame(frame, bg=C_BG, width=300)
        centre.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4), pady=6)
        centre.pack_propagate(False)

        # Right column: round header + metrics + best banner + chart
        right = tk.Frame(frame, bg=C_BG, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,8), pady=6)
        right.pack_propagate(False)

        # ── Network canvas (larger) ────────────────────────────
        canvas_host = tk.Frame(left, bg=CV_BG)
        canvas_host.pack(fill=tk.BOTH, expand=True)
        self.net_canvas = NetworkCanvas(canvas_host, width=940, height=560)

        # ── Ledger strip (two columns) ────────────────────────
        self.ledger = LedgerStrip(left)

        # ══ Centre column contents ════════════════════════════
        # Step card label at top
        tk.Label(centre, text="WHAT'S HAPPENING", bg=C_BG, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(fill=tk.X, padx=6, pady=(0,4))

        # Step card occupies most of centre column
        card_host = tk.Frame(centre, bg=C_CARD2)
        card_host.pack(fill=tk.BOTH, expand=True, padx=0)
        self.step_card = StepCard(card_host)
        self.step_card.clear()

        # (Controls moved to right panel)

        # ══ Right column contents ════════════════════════════

        # ── FL Controls (top of right panel) ────────────────────
        ctrl_outer = tk.Frame(right, bg=C_PANEL, pady=6, padx=8)
        ctrl_outer.pack(fill=tk.X, pady=(0,5))

        ctrl_row1 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row1.pack(fill=tk.X, pady=(0,3))

        self.btn_next = tk.Button(ctrl_row1, text="▶  Next Round",
                                   command=self._fl_next_round,
                                   bg=C_ACCENT, fg="white",
                                   font=("Helvetica",10,"bold"),
                                   relief=tk.FLAT, padx=10, pady=5)
        self.btn_next.pack(side=tk.LEFT, padx=(0,5))

        tk.Button(ctrl_row1, text="⟳  Reset", command=self._fl_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)

        # Next Step button — enabled only while a round is mid-animation
        self.btn_next_step = tk.Button(ctrl_row1, text="Step ▶",
                                        command=self._fl_advance_step,
                                        bg="#3A4060", fg=C_TEXT2,
                                        font=("Helvetica",9),
                                        relief=tk.FLAT, padx=8, pady=5,
                                        state=tk.DISABLED)
        self.btn_next_step.pack(side=tk.LEFT, padx=(8,0))

        ctrl_row2 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row2.pack(fill=tk.X)
        self.auto_chk = tk.Checkbutton(
            ctrl_row2, text="Auto-run  (pauses at Round 6)",
            variable=self.fl_auto_mode,
            command=self._fl_on_auto_toggle,
            bg=C_PANEL, fg=C_TEXT, selectcolor=C_CARD,
            font=("Helvetica",9),
            activebackground=C_PANEL, activeforeground=C_TEXT)
        self.auto_chk.pack(side=tk.LEFT)
        self.lbl_auto_status = tk.Label(ctrl_row2, text="", bg=C_PANEL, fg=C_GOLD,
                                         font=("Helvetica",8,"bold"))
        self.lbl_auto_status.pack(side=tk.LEFT, padx=(8,0))

        # Round header
        hcard = tk.Frame(right, bg=C_CARD, pady=8, padx=12)
        hcard.pack(fill=tk.X, pady=(0,5))
        self.lbl_round = tk.Label(hcard, text="Pre-Training Baseline",
                                   bg=C_CARD, fg=C_TEXT,
                                   font=("Helvetica",12,"bold"), anchor="w")
        self.lbl_round.pack(fill=tk.X)
        self.lbl_phase = tk.Label(hcard,
                                   text="Global model initialised. No FL rounds applied yet.",
                                   bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8),
                                   anchor="w", wraplength=300, justify="left")
        self.lbl_phase.pack(fill=tk.X, pady=(3,0))

        # Metrics
        mf = tk.Frame(right, bg=C_CARD)
        mf.pack(fill=tk.X, pady=(0,5))
        self.metric_vars   = {}
        self.metric_frames = {}
        for col, (key, lbl) in enumerate([("acc","Accuracy"),
                                           ("f1","F1 Score"),
                                           ("auc","ROC-AUC")]):
            cell = tk.Frame(mf, bg=C_CARD, padx=8, pady=7)
            cell.grid(row=0, column=col, sticky="nsew")
            mf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg=C_CARD, fg=C_TEXT2,
                     font=("Helvetica",7,"bold")).pack()
            v  = tk.StringVar(value="—")
            self.metric_vars[key] = v
            ml = tk.Label(cell, textvariable=v, bg=C_CARD, fg=C_GREEN,
                          font=("Helvetica",16,"bold"))
            ml.pack()
            self.metric_frames[key] = (cell, ml)

        # Best checkpoint banner (hidden until Round 6)
        self.best_banner = tk.Frame(right, bg=C_GOLD)
        tk.Label(self.best_banner,
                 text="★  BEST CHECKPOINT  —  ROUND 6  ★",
                 bg=C_GOLD, fg="#1A1000",
                 font=("Helvetica",9,"bold")).pack(pady=(5,1))
        bmf = tk.Frame(self.best_banner, bg=C_GOLD)
        bmf.pack(fill=tk.X, padx=6, pady=(0,5))
        for col, (lbl, val) in enumerate([("Accuracy","96.52%"),
                                           ("F1 Score","96.52%"),
                                           ("ROC-AUC","0.9964")]):
            cell = tk.Frame(bmf, bg="#9A6800", padx=8, pady=3)
            cell.grid(row=0, column=col, padx=2, sticky="nsew")
            bmf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg="#9A6800", fg="#FFE8A0",
                     font=("Helvetica",7,"bold")).pack()
            tk.Label(cell, text=val, bg="#9A6800", fg="white",
                     font=("Helvetica",12,"bold")).pack()

        # Chart
        self._build_fl_chart(right)
        self._fl_update_display(0)

    def _build_fl_chart(self, parent):
        # Match the mid-light theme in chart surroundings
        fig_bg = C_BG   # slate — matches right panel
        ax_bg  = "#22243A"
        self.fl_fig, self.fl_axes = plt.subplots(
            2, 1, figsize=(3.4, 4.2),
            facecolor=fig_bg, gridspec_kw={"hspace":0.60})
        self.fl_fig.patch.set_facecolor(fig_bg)

        for ax in self.fl_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = self.fl_axes
        ax_acc.set_title("Accuracy & F1 per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_acc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_acc.set_ylabel("Score",  color=C_TEXT2, fontsize=7)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(CHART_ACC_LO, CHART_ACC_HI)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        # Tight Y tick labels
        ax_acc.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.3f"))

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_auc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_auc.set_ylabel("AUC",   color=C_TEXT2, fontsize=7)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(CHART_AUC_LO, CHART_AUC_HI)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_auc.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.4f"))

        self.line_acc, = ax_acc.plot([], [], color="#6AB4FF", linewidth=1.8,
                                      marker="o", markersize=4, label="Accuracy", zorder=3)
        self.line_f1,  = ax_acc.plot([], [], color="#FF9966", linewidth=1.2,
                                      linestyle="--", marker="s", markersize=3,
                                      label="F1", zorder=3)
        ax_acc.legend(fontsize=6, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        self.line_auc, = ax_auc.plot([], [], color=C_GREEN, linewidth=1.8,
                                      marker="o", markersize=4, zorder=3)

        self.vline_acc = ax_acc.axvline(x=6, color=C_GOLD, linewidth=1.8,
                                         linestyle=":", alpha=0.0)
        self.vline_auc = ax_auc.axvline(x=6, color=C_GOLD, linewidth=1.8,
                                         linestyle=":", alpha=0.0)
        self.ann_r6_acc = ax_acc.annotate(
            "★ R6\n96.52%", xy=(6,0.9652), xytext=(7.5, CHART_ACC_LO+0.003),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)
        self.ann_r6_auc = ax_auc.annotate(
            "★ R6\n0.9964", xy=(6,0.9964), xytext=(7.5, CHART_AUC_LO+0.0005),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)

        self.fl_canvas_widget = FigureCanvasTkAgg(self.fl_fig, master=parent)
        self.fl_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fl_canvas_widget.draw()

    # ── FL orchestration ──────────────────────────────────────
    def _fl_next_round(self):
        if self.fl_animating: return
        if self.fl_auto_paused:
            self.fl_auto_paused = False
            self.lbl_auto_status.config(
                text="▶ Auto running…" if self.fl_auto_mode.get() else "")
        next_idx = self.fl_round_index + 1
        if next_idx >= len(ROUND_DATA):
            self.net_canvas.set_status(
                "All 10 rounds complete. Best-weight tracking deployed Round 6 model.",
                C_GOLD)
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto()
            return
        self.fl_animating = True
        self.btn_next.config(state=tk.DISABLED)
        self.auto_chk.config(state=tk.DISABLED)
        self._fl_run_round(next_idx)

    def _fl_run_round(self, idx):
        """Pure callback chain — no blocking thread.
        Each step animates, then waits for either auto-advance or manual Next Step."""
        rd       = ROUND_DATA[idx]
        selected = rd[6]
        rnd      = rd[0]

        self._fl_set_header(idx)

        # Enable Next Step button while round is in progress
        self.btn_next_step.config(state=tk.NORMAL, bg=C_ACCENT, fg="white")

        def _gate(next_cb):
            """After an animation finishes, either auto-advance or wait for button."""
            if self.fl_auto_mode.get():
                self.root.after(int(T_AUTO_GAP * 600), next_cb)
            else:
                # Park the callback — Next Step button will fire it
                self.fl_step_pending_cb = next_cb

        def step_select():
            self.step_card.show("select")
            self.net_canvas.animate_select(selected, rnd, lambda: _gate(step_train))

        def step_train():
            self.step_card.show("train")
            self.net_canvas.animate_train(selected, lambda: _gate(step_send))

        def step_send():
            self.step_card.show("send")
            self.net_canvas.animate_send(selected, lambda: _gate(step_validate))

        def step_validate():
            self.step_card.show("validate")
            self.net_canvas.animate_validate(selected, lambda: _gate(step_aggregate))

        def step_aggregate():
            self.step_card.show("aggregate")
            self.net_canvas.animate_aggregate(selected, lambda: _gate(step_rep))

        def step_rep():
            self.step_card.show("rep")
            self.net_canvas.animate_reputation(selected, rnd, step_finish)

        def step_finish():
            # Disable Next Step — round done
            self.btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
            self.fl_step_pending_cb = None
            self._fl_update_metrics(idx)
            self._fl_update_chart(idx)
            self.ledger.advance_to(rnd)
            self.step_card.clear()
            self.net_canvas.clear_breadcrumb()
            if rnd == 6:
                self._fl_show_best_checkpoint()
                self._fl_pause_auto_at_r6()
            else:
                self._fl_finish_round(idx)

        self.root.after(0, step_select)

    def _fl_advance_step(self):
        """Called by the Next Step button to fire the pending inter-step callback."""
        cb = self.fl_step_pending_cb
        if cb is not None:
            self.fl_step_pending_cb = None
            cb()

    def _fl_set_header(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        if rnd == 0:
            self.lbl_round.config(text="Pre-Training Baseline")
            self.lbl_phase.config(
                text="Global model initialised with pre-trained EfficientNetB4 weights. "
                     "No federated rounds applied yet.")
        else:
            self.lbl_round.config(text=f"Round {rnd}  /  10")
            self.lbl_phase.config(
                text=f"Selected: C{rd[6][0]}  ·  C{rd[6][1]}  ·  C{rd[6][2]}     "
                     f"Updates accepted: 3 / 3")

    def _fl_update_metrics(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        for key, val in [("acc", f"{rd[1]*100:.2f}%"),
                          ("f1",  f"{rd[2]*100:.2f}%"),
                          ("auc", f"{rd[3]:.4f}")]:
            self.metric_vars[key].set(val)
            cell, lbl = self.metric_frames[key]
            color = C_GOLD if rnd == 6 else C_GREEN
            bg    = "#1A1500" if rnd == 6 else C_CARD
            lbl.config(fg=color); cell.config(bg=bg)

        # Flash metrics gold three times at Round 6
        if rnd == 6:
            self._flash_metrics(3)

    def _flash_metrics(self, remaining):
        """Toggle metric labels bright-white/gold to draw the eye."""
        if remaining <= 0:
            for key in ("acc","f1","auc"):
                _, lbl = self.metric_frames[key]
                lbl.config(fg=C_GOLD)
            return
        flash_on = (remaining % 2 == 1)
        for key in ("acc","f1","auc"):
            cell, lbl = self.metric_frames[key]
            lbl.config(fg="white" if flash_on else C_GOLD)
            cell.config(bg="#3A2800" if flash_on else "#1A1500")
        self.root.after(220, lambda: self._flash_metrics(remaining - 1))

    def _fl_update_chart(self, idx):
        xs   = [ROUND_DATA[i][0] for i in range(idx+1)]
        accs = [ROUND_DATA[i][1] for i in range(idx+1)]
        f1s  = [ROUND_DATA[i][2] for i in range(idx+1)]
        aucs = [ROUND_DATA[i][3] for i in range(idx+1)]
        self.line_acc.set_data(xs, accs)
        self.line_f1.set_data(xs, f1s)
        self.line_auc.set_data(xs, aucs)
        if ROUND_DATA[idx][0] >= 6:
            self.vline_acc.set_alpha(0.9); self.vline_auc.set_alpha(0.9)
            self.ann_r6_acc.set_alpha(1.0); self.ann_r6_auc.set_alpha(1.0)
        self.fl_canvas_widget.draw_idle()

    def _fl_show_best_checkpoint(self):
        self.best_banner.pack(fill=tk.X, pady=(0,5),
                              after=self.metric_frames["auc"][0].master)
        self.net_canvas.show_best_checkpoint()

    def _fl_finish_round(self, idx):
        self.net_canvas.round_done_reset()
        self.fl_round_index = idx
        self.fl_animating   = False
        self.auto_chk.config(state=tk.NORMAL)

        if idx < len(ROUND_DATA) - 1:
            self.btn_next.config(state=tk.NORMAL)
        else:
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto(); return

        if self.fl_auto_mode.get() and not self.fl_auto_paused:
            self.fl_auto_after = self.root.after(
                int(T_AUTO_GAP*1000), self._fl_next_round)

    def _fl_pause_auto_at_r6(self):
        self.net_canvas.round_done_reset()
        self.fl_round_index = 6
        self.fl_animating   = False
        self.fl_auto_paused = True
        self.lbl_auto_status.config(text="⏸ Paused at Round 6")
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)

    def _fl_on_auto_toggle(self):
        if self.fl_auto_mode.get():
            self.lbl_auto_status.config(text="▶ Auto running…")
            if self.fl_auto_paused:
                self.fl_auto_paused = False
            if not self.fl_animating and self.fl_round_index < len(ROUND_DATA)-1:
                self.fl_auto_after = self.root.after(
                    int(T_AUTO_GAP*1000), self._fl_next_round)
        else:
            self.lbl_auto_status.config(text="")
            if self.fl_auto_after:
                try: self.root.after_cancel(self.fl_auto_after)
                except: pass
                self.fl_auto_after = None

    def _fl_disable_auto(self):
        self.fl_auto_mode.set(False)
        self.lbl_auto_status.config(text="")
        if self.fl_auto_after:
            try: self.root.after_cancel(self.fl_auto_after)
            except: pass
            self.fl_auto_after = None

    def _fl_reset(self):
        self._fl_disable_auto()
        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_paused = False
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)
        self.btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
        self.fl_step_pending_cb = None
        self.net_canvas.reset()
        self.ledger.reset()
        self.step_card.clear()

        self.line_acc.set_data([],[]); self.line_f1.set_data([],[])
        self.line_auc.set_data([],[])
        self.vline_acc.set_alpha(0.0); self.vline_auc.set_alpha(0.0)
        self.ann_r6_acc.set_alpha(0.0); self.ann_r6_auc.set_alpha(0.0)
        self.fl_canvas_widget.draw_idle()

        for key in ("acc","f1","auc"):
            self.metric_vars[key].set("—")
            cell, lbl = self.metric_frames[key]
            lbl.config(fg=C_GREEN); cell.config(bg=C_CARD)

        try: self.best_banner.pack_forget()
        except: pass

        self.lbl_round.config(text="Pre-Training Baseline")
        self.lbl_phase.config(
            text="Global model initialised. No FL rounds applied yet.")
        self._fl_update_display(0)

    def _fl_update_display(self, idx):
        rd = ROUND_DATA[idx]
        self.metric_vars["acc"].set(f"{rd[1]*100:.2f}%")
        self.metric_vars["f1"].set(f"{rd[2]*100:.2f}%")
        self.metric_vars["auc"].set(f"{rd[3]:.4f}")


    # ══════════════════════════════════════════════════════════
    # TAB 2 — Plain FedAvg (baseline)
    # ══════════════════════════════════════════════════════════
    def _build_plain_fl_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Plain FedAvg (Baseline)  ")

        self.pf_round_index = 0
        self.pf_animating = False
        self.pf_auto_mode = tk.BooleanVar(value=False)
        self.pf_auto_after = None
        self.pf_step_pending_cb = None

        left = tk.Frame(frame, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,4), pady=6)

        centre = tk.Frame(frame, bg=C_BG, width=300)
        centre.pack(side=tk.LEFT, fill=tk.Y, padx=(0,4), pady=6)
        centre.pack_propagate(False)

        right = tk.Frame(frame, bg=C_BG, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,8), pady=6)
        right.pack_propagate(False)

        canvas_host = tk.Frame(left, bg=CV_BG)
        canvas_host.pack(fill=tk.BOTH, expand=True)
        self.pf_canvas = SimpleNetworkCanvas(canvas_host, width=940, height=560)

        # No reputation ledger for plain FedAvg — replaced with a note strip
        note = tk.Frame(left, bg=C_PANEL, pady=6, padx=8)
        note.pack(fill=tk.X, pady=(6,0))
        tk.Label(note, text="ℹ  Plain FedAvg keeps no reputation ledger — client "
                             "selection is uniform-random every round, with no "
                             "memory of past performance.",
                 bg=C_PANEL, fg=C_TEXT2, font=("Helvetica", 9), anchor="w",
                 wraplength=900, justify="left").pack(fill=tk.X)

        tk.Label(centre, text="WHAT'S HAPPENING", bg=C_BG, fg=C_TEXT2,
                 font=("Helvetica",8,"bold")).pack(fill=tk.X, padx=6, pady=(0,4))
        card_host = tk.Frame(centre, bg=C_CARD2)
        card_host.pack(fill=tk.BOTH, expand=True, padx=0)
        self.pf_step_card = SimpleStepCard(card_host)
        self.pf_step_card.clear()

        ctrl_outer = tk.Frame(right, bg=C_PANEL, pady=6, padx=8)
        ctrl_outer.pack(fill=tk.X, pady=(0,5))
        ctrl_row1 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row1.pack(fill=tk.X, pady=(0,3))
        self.pf_btn_next = tk.Button(ctrl_row1, text="▶  Next Round",
                                      command=self._pf_next_round,
                                      bg=C_ACCENT, fg="white",
                                      font=("Helvetica",10,"bold"),
                                      relief=tk.FLAT, padx=10, pady=5)
        self.pf_btn_next.pack(side=tk.LEFT, padx=(0,5))
        tk.Button(ctrl_row1, text="⟳  Reset", command=self._pf_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)
        self.pf_btn_next_step = tk.Button(ctrl_row1, text="Step ▶",
                                           command=self._pf_advance_step,
                                           bg="#3A4060", fg=C_TEXT2,
                                           font=("Helvetica",9),
                                           relief=tk.FLAT, padx=8, pady=5,
                                           state=tk.DISABLED)
        self.pf_btn_next_step.pack(side=tk.LEFT, padx=(8,0))

        ctrl_row2 = tk.Frame(ctrl_outer, bg=C_PANEL)
        ctrl_row2.pack(fill=tk.X)
        self.pf_auto_chk = tk.Checkbutton(
            ctrl_row2, text="Auto-run  (stops after Round 10)",
            variable=self.pf_auto_mode, command=self._pf_on_auto_toggle,
            bg=C_PANEL, fg=C_TEXT, selectcolor=C_CARD, font=("Helvetica",9),
            activebackground=C_PANEL, activeforeground=C_TEXT)
        self.pf_auto_chk.pack(side=tk.LEFT)
        self.pf_lbl_auto_status = tk.Label(ctrl_row2, text="", bg=C_PANEL,
                                            fg="#B0B8D0", font=("Helvetica",8,"bold"))
        self.pf_lbl_auto_status.pack(side=tk.LEFT, padx=(8,0))

        hcard = tk.Frame(right, bg=C_CARD, pady=8, padx=12)
        hcard.pack(fill=tk.X, pady=(0,5))
        self.pf_lbl_round = tk.Label(hcard, text="Pre-Training Baseline",
                                      bg=C_CARD, fg=C_TEXT,
                                      font=("Helvetica",12,"bold"), anchor="w")
        self.pf_lbl_round.pack(fill=tk.X)
        self.pf_lbl_phase = tk.Label(hcard,
                                      text="Global model initialised. No FL rounds applied yet.",
                                      bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8),
                                      anchor="w", wraplength=300, justify="left")
        self.pf_lbl_phase.pack(fill=tk.X, pady=(3,0))

        mf = tk.Frame(right, bg=C_CARD)
        mf.pack(fill=tk.X, pady=(0,5))
        self.pf_metric_vars = {}
        self.pf_metric_frames = {}
        for col, (key, lbl) in enumerate([("acc","Accuracy"),("f1","F1 Score"),("auc","ROC-AUC")]):
            cell = tk.Frame(mf, bg=C_CARD, padx=8, pady=7)
            cell.grid(row=0, column=col, sticky="nsew")
            mf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg=C_CARD, fg=C_TEXT2,
                     font=("Helvetica",7,"bold")).pack()
            v = tk.StringVar(value="—")
            self.pf_metric_vars[key] = v
            ml = tk.Label(cell, textvariable=v, bg=C_CARD, fg="#B0B8D0",
                          font=("Helvetica",16,"bold"))
            ml.pack()
            self.pf_metric_frames[key] = (cell, ml)

        self.pf_best_banner = tk.Frame(right, bg="#5A5E70")
        tk.Label(self.pf_best_banner, text="★  BEST CHECKPOINT  —  ROUND 8  ★",
                 bg="#5A5E70", fg="white",
                 font=("Helvetica",9,"bold")).pack(pady=(5,1))
        bmf = tk.Frame(self.pf_best_banner, bg="#5A5E70")
        bmf.pack(fill=tk.X, padx=6, pady=(0,5))
        for col, (lbl, val) in enumerate([("Accuracy","94.42%"),
                                           ("F1 Score","94.58%"),
                                           ("ROC-AUC","0.9865")]):
            cell = tk.Frame(bmf, bg="#42465A", padx=8, pady=3)
            cell.grid(row=0, column=col, padx=2, sticky="nsew")
            bmf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg="#42465A", fg="#D8DCE8",
                     font=("Helvetica",7,"bold")).pack()
            tk.Label(cell, text=val, bg="#42465A", fg="white",
                     font=("Helvetica",12,"bold")).pack()

        self._build_pf_chart(right)
        self._pf_update_display(0)

    def _build_pf_chart(self, parent):
        fig_bg = C_BG; ax_bg = "#22243A"
        self.pf_fig, self.pf_axes = plt.subplots(
            2, 1, figsize=(3.4, 4.2), facecolor=fig_bg,
            gridspec_kw={"hspace":0.60})
        self.pf_fig.patch.set_facecolor(fig_bg)
        for ax in self.pf_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = self.pf_axes
        ax_acc.set_title("Accuracy & F1 per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_acc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_acc.set_ylabel("Score", color=C_TEXT2, fontsize=7)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(CHART_PLAIN_ACC_LO, CHART_PLAIN_ACC_HI)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_acc.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT,
                          fontsize=8, pad=4, fontweight="bold")
        ax_auc.set_xlabel("Round", color=C_TEXT2, fontsize=7)
        ax_auc.set_ylabel("AUC", color=C_TEXT2, fontsize=7)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(CHART_PLAIN_AUC_LO, CHART_PLAIN_AUC_HI)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color=C_TEXT2, fontsize=6)
        ax_auc.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))

        self.pf_line_acc, = ax_acc.plot([], [], color="#8FA8D0", linewidth=1.8,
                                         marker="o", markersize=4, label="Accuracy", zorder=3)
        self.pf_line_f1, = ax_acc.plot([], [], color="#C9A0DC", linewidth=1.2,
                                        linestyle="--", marker="s", markersize=3,
                                        label="F1", zorder=3)
        ax_acc.legend(fontsize=6, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)
        self.pf_line_auc, = ax_auc.plot([], [], color="#8FD0B0", linewidth=1.8,
                                         marker="o", markersize=4, zorder=3)
        self.pf_vline_acc = ax_acc.axvline(x=8, color="#B0B8D0", linewidth=1.8,
                                            linestyle=":", alpha=0.0)
        self.pf_vline_auc = ax_auc.axvline(x=8, color="#B0B8D0", linewidth=1.8,
                                            linestyle=":", alpha=0.0)
        self.pf_ann_r8_acc = ax_acc.annotate(
            "★ R8\n94.42%", xy=(8,0.9442), xytext=(6.0, CHART_PLAIN_ACC_LO+0.003),
            color="#B0B8D0", fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B0B8D0", lw=1.0), alpha=0.0)
        self.pf_ann_r8_auc = ax_auc.annotate(
            "★ R8\n0.9865", xy=(8,0.9865), xytext=(6.0, CHART_PLAIN_AUC_LO+0.0005),
            color="#B0B8D0", fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B0B8D0", lw=1.0), alpha=0.0)

        self.pf_canvas_widget = FigureCanvasTkAgg(self.pf_fig, master=parent)
        self.pf_canvas_widget.get_tk_widget().pack(fill=tk.X, pady=(5,0))
        self.pf_canvas_widget.draw()

    def _pf_next_round(self):
        if self.pf_animating: return
        next_idx = self.pf_round_index + 1
        if next_idx >= len(PLAIN_ROUND_DATA_FILLED):
            self.pf_canvas.set_status(
                "All 10 rounds complete. Round 8 kept as best (highest test accuracy).",
                "#B0B8D0")
            self.pf_btn_next.config(state=tk.DISABLED)
            self._pf_disable_auto()
            return
        self.pf_animating = True
        self.pf_btn_next.config(state=tk.DISABLED)
        self.pf_auto_chk.config(state=tk.DISABLED)
        self._pf_run_round(next_idx)

    def _pf_run_round(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd, selected, local_accs = rd[0], rd[5], rd[6]
        self._pf_set_header(idx)
        self.pf_btn_next_step.config(state=tk.NORMAL, bg=C_ACCENT, fg="white")

        def _gate(next_cb):
            if self.pf_auto_mode.get():
                self.root.after(int(T_AUTO_GAP * 600), next_cb)
            else:
                self.pf_step_pending_cb = next_cb

        def step_select():
            self.pf_step_card.show("select")
            self.pf_canvas.animate_select(selected, rnd, lambda: _gate(step_train))

        def step_train():
            self.pf_step_card.show("train")
            self.pf_canvas.animate_train(selected, local_accs, lambda: _gate(step_send))

        def step_send():
            self.pf_step_card.show("send")
            self.pf_canvas.animate_send(selected, lambda: _gate(step_average))

        def step_average():
            self.pf_step_card.show("average")
            self.pf_canvas.animate_average(selected, step_finish)

        def step_finish():
            self.pf_btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
            self.pf_step_pending_cb = None
            self._pf_update_metrics(idx)
            self._pf_update_chart(idx)
            self.pf_step_card.clear()
            self.pf_canvas.clear_breadcrumb()
            if rnd == PLAIN_BEST_ROUND:
                self._pf_show_best_checkpoint()
            self._pf_finish_round(idx)

        self.root.after(0, step_select)

    def _pf_advance_step(self):
        cb = self.pf_step_pending_cb
        if cb is not None:
            self.pf_step_pending_cb = None
            cb()

    def _pf_set_header(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd, selected = rd[0], rd[5]
        if rnd == 0:
            self.pf_lbl_round.config(text="Pre-Training Baseline")
            self.pf_lbl_phase.config(
                text="Global model initialised with pre-trained weights. "
                     "No federated rounds applied yet.")
        else:
            self.pf_lbl_round.config(text=f"Round {rnd}  /  10")
            self.pf_lbl_phase.config(
                text=f"Randomly selected: C{selected[0]}  ·  C{selected[1]}  ·  C{selected[2]}     "
                     f"All updates used (no validation gate)")

    def _pf_update_metrics(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        rnd = rd[0]
        for key, val in [("acc", f"{rd[1]*100:.2f}%"), ("f1", f"{rd[2]*100:.2f}%"),
                          ("auc", f"{rd[3]:.4f}")]:
            self.pf_metric_vars[key].set(val)
            cell, lbl = self.pf_metric_frames[key]
            color = "#B0B8D0" if rnd == PLAIN_BEST_ROUND else "#8FA8D0"
            bg = "#33364A" if rnd == PLAIN_BEST_ROUND else C_CARD
            lbl.config(fg=color); cell.config(bg=bg)

    def _pf_update_chart(self, idx):
        xs = [PLAIN_ROUND_DATA_FILLED[i][0] for i in range(idx+1)]
        accs = [PLAIN_ROUND_DATA_FILLED[i][1] for i in range(idx+1)]
        f1s = [PLAIN_ROUND_DATA_FILLED[i][2] for i in range(idx+1)]
        aucs = [PLAIN_ROUND_DATA_FILLED[i][3] for i in range(idx+1)]
        self.pf_line_acc.set_data(xs, accs)
        self.pf_line_f1.set_data(xs, f1s)
        self.pf_line_auc.set_data(xs, aucs)
        if PLAIN_ROUND_DATA_FILLED[idx][0] >= PLAIN_BEST_ROUND:
            self.pf_vline_acc.set_alpha(0.9); self.pf_vline_auc.set_alpha(0.9)
            self.pf_ann_r8_acc.set_alpha(1.0); self.pf_ann_r8_auc.set_alpha(1.0)
        self.pf_canvas_widget.draw_idle()

    def _pf_show_best_checkpoint(self):
        self.pf_best_banner.pack(fill=tk.X, pady=(0,5),
                                  after=self.pf_metric_frames["auc"][0].master)
        self.pf_canvas.show_best_checkpoint()

    def _pf_finish_round(self, idx):
        self.pf_canvas.round_done_reset()
        self.pf_round_index = idx
        self.pf_animating = False
        self.pf_auto_chk.config(state=tk.NORMAL)
        if idx < len(PLAIN_ROUND_DATA_FILLED) - 1:
            self.pf_btn_next.config(state=tk.NORMAL)
        else:
            self.pf_btn_next.config(state=tk.DISABLED)
            self._pf_disable_auto(); return
        if self.pf_auto_mode.get():
            self.pf_auto_after = self.root.after(int(T_AUTO_GAP*1000), self._pf_next_round)

    def _pf_on_auto_toggle(self):
        if self.pf_auto_mode.get():
            self.pf_lbl_auto_status.config(text="▶ Auto running…")
            if not self.pf_animating and self.pf_round_index < len(PLAIN_ROUND_DATA_FILLED)-1:
                self.pf_auto_after = self.root.after(int(T_AUTO_GAP*1000), self._pf_next_round)
        else:
            self.pf_lbl_auto_status.config(text="")
            if self.pf_auto_after:
                try: self.root.after_cancel(self.pf_auto_after)
                except: pass
                self.pf_auto_after = None

    def _pf_disable_auto(self):
        self.pf_auto_mode.set(False)
        self.pf_lbl_auto_status.config(text="")
        if self.pf_auto_after:
            try: self.root.after_cancel(self.pf_auto_after)
            except: pass
            self.pf_auto_after = None

    def _pf_reset(self):
        self._pf_disable_auto()
        self.pf_round_index = 0
        self.pf_animating = False
        self.pf_btn_next.config(state=tk.NORMAL)
        self.pf_auto_chk.config(state=tk.NORMAL)
        self.pf_btn_next_step.config(state=tk.DISABLED, bg="#3A4060", fg=C_TEXT2)
        self.pf_step_pending_cb = None
        self.pf_canvas.reset()
        self.pf_step_card.clear()
        self.pf_line_acc.set_data([],[]); self.pf_line_f1.set_data([],[])
        self.pf_line_auc.set_data([],[])
        self.pf_vline_acc.set_alpha(0.0); self.pf_vline_auc.set_alpha(0.0)
        self.pf_ann_r8_acc.set_alpha(0.0); self.pf_ann_r8_auc.set_alpha(0.0)
        self.pf_canvas_widget.draw_idle()
        for key in ("acc","f1","auc"):
            self.pf_metric_vars[key].set("—")
            cell, lbl = self.pf_metric_frames[key]
            lbl.config(fg="#8FA8D0"); cell.config(bg=C_CARD)
        try: self.pf_best_banner.pack_forget()
        except: pass
        self.pf_lbl_round.config(text="Pre-Training Baseline")
        self.pf_lbl_phase.config(text="Global model initialised. No FL rounds applied yet.")
        self._pf_update_display(0)

    def _pf_update_display(self, idx):
        rd = PLAIN_ROUND_DATA_FILLED[idx]
        self.pf_metric_vars["acc"].set(f"{rd[1]*100:.2f}%")
        self.pf_metric_vars["f1"].set(f"{rd[2]*100:.2f}%")
        self.pf_metric_vars["auc"].set(f"{rd[3]:.4f}")


    # ══════════════════════════════════════════════════════════
    # TAB 3 — Comparison (Enhanced vs Plain)
    # ══════════════════════════════════════════════════════════
    def _build_comparison_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Comparison  ")

        top = tk.Frame(frame, bg=C_PANEL, pady=9)
        top.pack(fill=tk.X)
        tk.Label(top, text="Enhanced (Reputation-Weighted) vs Plain FedAvg (Baseline)",
                 bg=C_PANEL, fg=C_TEXT, font=("Helvetica",12,"bold")).pack()

        body = tk.Frame(frame, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ── Left: pipeline step-count comparison ──────────────
        left = tk.Frame(body, bg=C_CARD2, padx=14, pady=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,6))
        tk.Label(left, text="PIPELINE STEPS", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(anchor="w", pady=(0,8))

        enh_row = tk.Frame(left, bg=C_CARD2)
        enh_row.pack(fill=tk.X, pady=(0,4))
        tk.Label(enh_row, text="Enhanced (6 steps):", bg=C_CARD2, fg=C_TEXT,
                 font=("Helvetica",9,"bold"), width=18, anchor="w").pack(side=tk.LEFT)
        for key, label in STEPS_DEF:
            tk.Label(enh_row, text=label.strip(), bg=STEP_COLORS.get(key,"#888"),
                     fg="#111", font=("Helvetica",8,"bold"), padx=6, pady=2
                     ).pack(side=tk.LEFT, padx=2)

        plain_row = tk.Frame(left, bg=C_CARD2)
        plain_row.pack(fill=tk.X, pady=(4,10))
        tk.Label(plain_row, text="Plain (4 steps):", bg=C_CARD2, fg=C_TEXT,
                 font=("Helvetica",9,"bold"), width=18, anchor="w").pack(side=tk.LEFT)
        for key, label in PLAIN_STEPS_DEF:
            tk.Label(plain_row, text=label.strip(), bg=PLAIN_STEP_COLORS.get(key,"#888"),
                     fg="#111", font=("Helvetica",8,"bold"), padx=6, pady=2
                     ).pack(side=tk.LEFT, padx=2)

        tk.Label(left, text="Enhanced adds Validate (safety checks) and Reputation "
                             "(ledger + decay) — steps that plain FedAvg skips entirely. "
                             "Enhanced also selects by score; plain selects at random, "
                             "and enhanced weights contributions by quality where plain "
                             "averages every update equally.",
                 bg=C_CARD2, fg=C_TEXT2, font=("Helvetica",9), wraplength=420,
                 justify="left", anchor="w").pack(fill=tk.X, pady=(4,10))

        # Feature comparison table
        table = tk.Frame(left, bg=C_CARD2)
        table.pack(fill=tk.X, pady=(4,0))
        rows = [
            ("Selection", "Score-based (Vᵢ·Hᵢ·Rᵢ)", "Uniform random"),
            ("Validation gate", "L2 norm + gain test", "None"),
            ("Aggregation", "Weighted by score", "Equal mean (1/3 each)"),
            ("Reputation ledger", "Yes, 0.99 decay/round", "None"),
            ("Best round", "Round 6", "Round 8"),
            ("Best Accuracy", "96.52%", "94.42%"),
            ("Best F1", "96.52%", "94.58%"),
            ("Best ROC-AUC", "0.9964", "0.9865"),
        ]
        hdr = tk.Frame(table, bg=C_CARD)
        hdr.pack(fill=tk.X)
        for i, h in enumerate(["Aspect","Enhanced","Plain"]):
            tk.Label(hdr, text=h, bg=C_CARD, fg=C_TEXT2, font=("Helvetica",8,"bold"),
                     width=[16,20,20][i], anchor="w", padx=4, pady=3
                     ).grid(row=0, column=i, sticky="w")
        for r, (a, e, p) in enumerate(rows):
            rowbg = C_CARD2 if r % 2 == 0 else "#33364E"
            rf = tk.Frame(table, bg=rowbg)
            rf.pack(fill=tk.X)
            tk.Label(rf, text=a, bg=rowbg, fg=C_TEXT, font=("Helvetica",8),
                     width=16, anchor="w", padx=4, pady=2).grid(row=0, column=0, sticky="w")
            tk.Label(rf, text=e, bg=rowbg, fg=C_GOLD, font=("Helvetica",8,"bold"),
                     width=20, anchor="w", padx=4, pady=2).grid(row=0, column=1, sticky="w")
            tk.Label(rf, text=p, bg=rowbg, fg="#B0B8D0", font=("Helvetica",8,"bold"),
                     width=20, anchor="w", padx=4, pady=2).grid(row=0, column=2, sticky="w")

        # ── Right: overlaid metric charts ──────────────────────
        right = tk.Frame(body, bg=C_CARD2, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6,0))
        tk.Label(right, text="METRICS OVERLAY (ALL 10 ROUNDS)", bg=C_CARD2, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(anchor="w", pady=(0,6))
        self._build_cmp_chart(right)


    def _build_cmp_chart(self, parent):
        fig_bg = C_BG; ax_bg = "#22243A"
        fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.6), facecolor=fig_bg,
                                  gridspec_kw={"hspace":0.55})
        fig.patch.set_facecolor(fig_bg)
        for ax in axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=C_TEXT2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555577")

        ax_acc, ax_auc = axes
        enh_x = [rd[0] for rd in ROUND_DATA]
        enh_acc = [rd[1] for rd in ROUND_DATA]
        enh_auc = [rd[3] for rd in ROUND_DATA]
        plain_x = [rd[0] for rd in PLAIN_ROUND_DATA_FILLED]
        plain_acc = [rd[1] for rd in PLAIN_ROUND_DATA_FILLED]
        plain_auc = [rd[3] for rd in PLAIN_ROUND_DATA_FILLED]

        ax_acc.set_title("Accuracy per Round", color=C_TEXT, fontsize=9,
                          pad=5, fontweight="bold")
        ax_acc.plot(enh_x, enh_acc, color=C_GOLD, linewidth=2.0, marker="o",
                    markersize=4, label="Enhanced", zorder=3)
        ax_acc.plot(plain_x, plain_acc, color="#8FA8D0", linewidth=2.0,
                    linestyle="--", marker="s", markersize=4, label="Plain", zorder=3)
        ax_acc.axvline(x=6, color=C_GOLD, linewidth=1.2, linestyle=":", alpha=0.6)
        ax_acc.axvline(x=8, color="#8FA8D0", linewidth=1.2, linestyle=":", alpha=0.6)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)], color=C_TEXT2, fontsize=6)
        ax_acc.set_ylabel("Accuracy", color=C_TEXT2, fontsize=7)
        ax_acc.legend(fontsize=7, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        ax_auc.set_title("ROC-AUC per Round", color=C_TEXT, fontsize=9,
                          pad=5, fontweight="bold")
        ax_auc.plot(enh_x, enh_auc, color=C_GOLD, linewidth=2.0, marker="o",
                    markersize=4, label="Enhanced", zorder=3)
        ax_auc.plot(plain_x, plain_auc, color="#8FD0B0", linewidth=2.0,
                    linestyle="--", marker="s", markersize=4, label="Plain", zorder=3)
        ax_auc.axvline(x=6, color=C_GOLD, linewidth=1.2, linestyle=":", alpha=0.6)
        ax_auc.axvline(x=8, color="#8FD0B0", linewidth=1.2, linestyle=":", alpha=0.6)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)], color=C_TEXT2, fontsize=6)
        ax_auc.set_ylabel("ROC-AUC", color=C_TEXT2, fontsize=7)
        ax_auc.legend(fontsize=7, facecolor="#2A2C44", labelcolor=C_TEXT2,
                      loc="lower right", framealpha=0.9)

        canvas_widget = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_widget.draw()


    def _build_sync_playback_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  Synced Playback  ")

        top = tk.Frame(frame, bg=C_PANEL, pady=9)
        top.pack(fill=tk.X)
        tk.Label(top, text="Side-by-Side Process Playback — Same Round, Both Pipelines",
                 bg=C_PANEL, fg=C_TEXT, font=("Helvetica",12,"bold")).pack()

        # ── Synced Side-by-Side Process Playback ───────────────
        sync_outer = tk.Frame(frame, bg=C_PANEL, pady=8, padx=10)
        sync_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        sync_hdr = tk.Frame(sync_outer, bg=C_PANEL)
        sync_hdr.pack(fill=tk.X, pady=(0,6))
        tk.Label(sync_hdr, text="SYNCED PROCESS PLAYBACK", bg=C_PANEL, fg=C_TEXT2,
                 font=("Helvetica",9,"bold")).pack(side=tk.LEFT)
        self.cmp_lbl_round = tk.Label(sync_hdr, text="Pre-Training", bg=C_PANEL,
                                       fg=C_TEXT, font=("Helvetica",9,"bold"))
        self.cmp_lbl_round.pack(side=tk.LEFT, padx=(14,0))

        sync_ctrl = tk.Frame(sync_outer, bg=C_PANEL)
        sync_ctrl.pack(fill=tk.X, pady=(0,6))
        self.cmp_btn_play = tk.Button(sync_ctrl, text="▶  Play Round (Both)",
                                       command=self._cmp_play_round,
                                       bg=C_ACCENT, fg="white",
                                       font=("Helvetica",10,"bold"),
                                       relief=tk.FLAT, padx=10, pady=5)
        self.cmp_btn_play.pack(side=tk.LEFT, padx=(0,5))
        tk.Button(sync_ctrl, text="⟳  Reset", command=self._cmp_reset,
                  bg=C_CARD, fg=C_TEXT, font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=5).pack(side=tk.LEFT)
        self.cmp_lbl_status = tk.Label(sync_ctrl, text="", bg=C_PANEL, fg=C_TEXT2,
                                        font=("Helvetica",8,"italic"))
        self.cmp_lbl_status.pack(side=tk.LEFT, padx=(12,0))

        dual = tk.Frame(sync_outer, bg=C_PANEL)
        dual.pack(fill=tk.BOTH, expand=True)

        left_host = tk.Frame(dual, bg=CV_BG)
        left_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,4))
        tk.Label(left_host, text="ENHANCED — 6 steps", bg="#14142A", fg=C_GOLD,
                 font=("Helvetica",9,"bold"), pady=3).pack(fill=tk.X)
        self.cmp_enh_canvas = NetworkCanvas(left_host, width=560, height=380)

        right_host = tk.Frame(dual, bg=CV_BG)
        right_host.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4,0))
        tk.Label(right_host, text="PLAIN FEDAVG — 4 steps", bg="#14142A", fg="#B0B8D0",
                 font=("Helvetica",9,"bold"), pady=3).pack(fill=tk.X)
        self.cmp_pf_canvas = SimpleNetworkCanvas(right_host, width=560, height=380)

        self.cmp_round_index = 0
        self.cmp_animating = False
        self.cmp_enh_done = False
        self.cmp_pf_done = False



    def _cmp_play_round(self):
        if self.cmp_animating:
            return
        next_idx = self.cmp_round_index + 1
        max_idx = min(len(ROUND_DATA), len(PLAIN_ROUND_DATA_FILLED)) - 1
        if next_idx > max_idx:
            self.cmp_lbl_status.config(text="All synced rounds complete.")
            self.cmp_btn_play.config(state=tk.DISABLED)
            return
        self.cmp_animating = True
        self.cmp_enh_done = False
        self.cmp_pf_done = False
        self.cmp_btn_play.config(state=tk.DISABLED)
        self.cmp_lbl_round.config(text=f"Round {next_idx} / 10  (same round, both sides)")
        self.cmp_lbl_status.config(text="Playing both pipelines for the same round…")

        enh_rd = ROUND_DATA[next_idx]
        enh_selected = enh_rd[6]
        enh_rnd = enh_rd[0]

        pf_rd = PLAIN_ROUND_DATA_FILLED[next_idx]
        pf_selected = pf_rd[5]
        pf_local_accs = pf_rd[6]
        pf_rnd = pf_rd[0]

        def enh_finished():
            self.cmp_enh_canvas.round_done_reset()
            self.cmp_enh_canvas.clear_breadcrumb()
            self.cmp_enh_done = True
            self._cmp_check_both_done(next_idx)

        def pf_finished():
            self.cmp_pf_canvas.round_done_reset()
            self.cmp_pf_canvas.clear_breadcrumb()
            self.cmp_pf_done = True
            self._cmp_check_both_done(next_idx)

        def enh_select():
            self.cmp_enh_canvas.animate_select(enh_selected, enh_rnd, enh_train)
        def enh_train():
            self.cmp_enh_canvas.animate_train(enh_selected, enh_send)
        def enh_send():
            self.cmp_enh_canvas.animate_send(enh_selected, enh_validate)
        def enh_validate():
            self.cmp_enh_canvas.animate_validate(enh_selected, enh_aggregate)
        def enh_aggregate():
            self.cmp_enh_canvas.animate_aggregate(enh_selected, enh_rep)
        def enh_rep():
            self.cmp_enh_canvas.animate_reputation(enh_selected, enh_rnd, enh_finished)

        def pf_select():
            self.cmp_pf_canvas.animate_select(pf_selected, pf_rnd, pf_train)
        def pf_train():
            self.cmp_pf_canvas.animate_train(pf_selected, pf_local_accs, pf_send)
        def pf_send():
            self.cmp_pf_canvas.animate_send(pf_selected, pf_average)
        def pf_average():
            self.cmp_pf_canvas.animate_average(pf_selected, pf_finished)

        # Start both pipelines at the same moment — Plain will naturally
        # finish first since it has fewer steps, visually showing the gap.
        self.root.after(0, enh_select)
        self.root.after(0, pf_select)

    def _cmp_check_both_done(self, idx):
        if self.cmp_enh_done and self.cmp_pf_done:
            self.cmp_round_index = idx
            self.cmp_animating = False
            self.cmp_btn_play.config(state=tk.NORMAL)
            self.cmp_lbl_status.config(
                text="Both sides finished this round. Notice Plain reached "
                     "'done' after only 4 steps.")

    def _cmp_reset(self):
        self.cmp_round_index = 0
        self.cmp_animating = False
        self.cmp_enh_done = False
        self.cmp_pf_done = False
        self.cmp_enh_canvas.reset()
        self.cmp_pf_canvas.reset()
        self.cmp_btn_play.config(state=tk.NORMAL)
        self.cmp_lbl_round.config(text="Pre-Training")
        self.cmp_lbl_status.config(text="")


    # ── Detection helpers ─────────────────────────────────────
    @staticmethod
    def _sep(p):
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

    def _pfdet_show_placeholder(self):
        img = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), dtype=np.uint8)
        img[:] = (30, 32, 50)
        cv2.putText(img,"No video loaded",
                    (DISPLAY_MAX_W//2-140,DISPLAY_MAX_H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(100,110,160),2)
        cv2.putText(img,"Click  Open Video  to begin",
                    (DISPLAY_MAX_W//2-200,DISPLAY_MAX_H//2+36),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(70,80,120),1)
        self._pfdet_render(img)

    def _pfdet_open_file(self):
        p = filedialog.askopenfilename(title="Select a video file",
                                        filetypes=VIDEO_EXTS)
        if p: self._pfdet_load_video(p)

    def _pfdet_browse_model(self):
        """Let the user pick any .tflite model file from disk and register it
        in the dropdown, without having to edit any code."""
        p = filedialog.askopenfilename(
            title="Select a TFLite model to load",
            filetypes=[("TFLite model","*.tflite"),("All files","*.*")],
            initialdir=str(Path(self.pf_model_path).parent))
        if not p:
            return
        name = Path(p).name
        # Avoid clobbering an existing entry with the same filename but a
        # different path — disambiguate by appending a counter if needed.
        base_name = name
        i = 2
        while name in self.pf_known_models and self.pf_known_models[name] != p:
            name = f"{base_name} ({i})"
            i += 1
        self.pf_known_models[name] = p
        self.pfdet_model_combo.config(values=list(self.pf_known_models.keys()))
        self.pfdet_model_var.set(name)
        self._pfdet_load_model(name)

    def _pfdet_on_model_selected(self, event=None):
        name = self.pfdet_model_var.get()
        self._pfdet_load_model(name)

    def _pfdet_load_model(self, name):
        """Swap the active TFLite model at runtime. The interpreter build
        (allocate_tensors etc.) can take several seconds for larger models,
        so it is done on a background thread to avoid freezing the UI.
        Already-loaded interpreters are cached so re-selecting a model you
        already loaded this session is instant."""
        path = self.pf_known_models.get(name)
        if not path:
            return
        if name == self.pf_active_model_name:
            return
        if self.pfdet_model_loading:
            # A load is already in flight — ignore extra clicks/selections
            # rather than queuing overlapping background loads.
            self.pfdet_model_var.set(self.pf_active_model_name)
            return

        cached = self.pf_model_cache.get(path)
        if cached is not None:
            self._pfdet_apply_loaded_model(name, path, cached)
            return

        self.pfdet_model_loading = True
        self.pfdet_model_combo.config(state="disabled")
        self.pfdet_model_status.config(text="⏳ Loading…", fg=C_TEXT2)

        def worker():
            try:
                result = load_tflite_model(path)
                error = None
            except Exception as e:
                result = None
                error = e
            self.root.after(0, lambda: self._pfdet_on_model_loaded(name, path, result, error))

        threading.Thread(target=worker, daemon=True).start()

    def _pfdet_on_model_loaded(self, name, path, result, error):
        self.pfdet_model_loading = False
        self.pfdet_model_combo.config(state="readonly")
        if error is not None:
            messagebox.showerror("Model Load Failed",
                                  f"Could not load model:\n{path}\n\n{error}")
            self.pfdet_model_var.set(self.pf_active_model_name)
            self.pfdet_model_status.config(text="")
            return
        self.pf_model_cache[path] = result
        self._pfdet_apply_loaded_model(name, path, result)

    def _pfdet_apply_loaded_model(self, name, path, loaded):
        new_interp, new_inp, new_out = loaded
        with self.pf_model_lock:
            self.pf_interpreter = new_interp
            self.pf_inp = new_inp
            self.pf_out = new_out
            self.pf_inp_dtype = self.pf_inp["dtype"]
            self.pf_inp_scale, self.pf_inp_zp = self.pf_inp.get("quantization",(1.0,0))
            self.pf_out_scale, self.pf_out_zp = self.pf_out.get("quantization",(1.0,0))
            self.pf_model_path = path
            self.pf_active_model_name = name

        # Reset the smoothing history so old-model predictions don't bleed
        # into the new model's rolling average, and clear any prior error
        # flag so a fresh model gets a fresh chance to report problems.
        self.pfdet_history.clear()
        self.pfdet_infer_error_shown = False
        self.pfdet_model_status.config(text="✓ Switched", fg=C_GREEN)
        self.root.after(1500, lambda: self.pfdet_model_status.config(text=""))

    def _pfdet_load_video(self, path):
        self._pfdet_cancel_loop(); self._pfdet_stop()
        if self.pfdet_cap: self.pfdet_cap.release()
        self.pfdet_cap = cv2.VideoCapture(path)
        if not self.pfdet_cap.isOpened():
            messagebox.showerror("Error",f"Cannot open video:\n{path}"); return
        self.pfdet_video_path=path
        self.pfdet_total_frames=int(self.pfdet_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pfdet_video_fps=self.pfdet_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pfdet_frame_idx=0; self.pfdet_history.clear()
        self.pfdet_raw_prob=0.5; self.pfdet_smoothed=0.5
        self.pfdet_label="NO FACE"; self.pfdet_cv_color=COLOR_NO_FACE
        self.pfdet_hex_color=C_NEUTRAL; self.pfdet_confidence=0.0
        self.pfdet_inf_ms=0.0; self.pfdet_bbox=None
        self.pfdet_verdict_counts={"REAL":0,"FAKE":0}
        self.pfdet_frame_log=[]; self.pfdet_last_det=None
        self.pfdet_fps_display=0.0; self.pfdet_t_prev=time.time()
        vw=int(self.pfdet_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh=int(self.pfdet_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sc=min(DISPLAY_MAX_W/vw,DISPLAY_MAX_H/vh,1.0)
        self.pfdet_display_w=int(vw*sc); self.pfdet_display_h=int(vh*sc)
        self.pfdet_vid_w=vw; self.pfdet_vid_h=vh
        self.pfdet_progress_bar.config(to=self.pfdet_total_frames)
        fname=Path(path).name
        self.pfdet_lbl_status.config(
            text=f"{fname}  |  {vw}×{vh}  |  {self.pfdet_video_fps:.1f} fps  |  {self.pfdet_total_frames} frames")
        self.root.title(f"Enhanced FL Demo — {fname}")
        self._pfdet_update_overall_verdict()
        ret,first=self.pfdet_cap.read()
        if ret: self._pfdet_render_and_show(first)
        self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.pfdet_after_id=self.root.after(33,self._pfdet_update_loop)

    def _pfdet_start_inference_thread(self):
        self.pfdet_infer_result={
            "raw_prob":0.5,"smoothed":0.5,"label":"NO FACE",
            "cv_color":COLOR_NO_FACE,"hex_color":C_NEUTRAL,
            "confidence":0.0,"inf_ms":0.0,"bbox":None}
        def worker():
            while not self.pfdet_stop_event.is_set():
                crop=bb=None
                with self.pfdet_infer_queue_lock:
                    if self.pfdet_infer_queue:
                        crop,bb=self.pfdet_infer_queue.pop()
                        self.pfdet_infer_queue.clear()
                if crop is None: time.sleep(0.005); continue
                # Snapshot the active model under lock so a live model swap
                # (triggered from the UI thread) can never be read half-updated.
                with self.pf_model_lock:
                    interp, inp, out = self.pf_interpreter, self.pf_inp, self.pf_out
                    dtype, scale, zp = self.pf_inp_dtype, self.pf_inp_scale, self.pf_inp_zp
                    o_scale, o_zp = self.pf_out_scale, self.pf_out_zp
                try:
                    t0=time.perf_counter()
                    arr=preprocess_face(crop,dtype,scale,zp)
                    interp.set_tensor(inp["index"],arr)
                    interp.invoke()
                    ro=interp.get_tensor(out["index"])
                    rp=(float((ro[0][0]-o_zp)*o_scale)
                        if dtype in (np.uint8, np.int8) else float(ro[0][0]))
                    ms=(time.perf_counter()-t0)*1000
                    self.pfdet_history.append(rp)
                    sm=float(np.mean(self.pfdet_history))
                    lb,cc,hc,cf=classify(sm)
                    with self.pfdet_infer_lock:
                        self.pfdet_infer_result.update({
                            "raw_prob":rp,"smoothed":sm,"label":lb,
                            "cv_color":cc,"hex_color":hc,"confidence":cf,
                            "inf_ms":ms,"bbox":bb})
                except Exception as e:
                    # Never let a bad frame/model mismatch silently kill this
                    # thread — surface the error once and keep the loop alive
                    # so a subsequent model swap or frame can recover.
                    with self.pfdet_infer_lock:
                        self.pfdet_infer_result["label"] = "INFER ERROR"
                    if not self.pfdet_infer_error_shown:
                        self.pfdet_infer_error_shown = True
                        err_text = str(e)
                        self.root.after(0, lambda: messagebox.showerror(
                            "Inference Error",
                            f"The active model raised an error during inference:\n\n{err_text}"))
        threading.Thread(target=worker,daemon=True).start()

    def _pfdet_cancel_loop(self):
        if self.pfdet_after_id:
            try: self.root.after_cancel(self.pfdet_after_id)
            except: pass
            self.pfdet_after_id=None

    def _pfdet_update_loop(self):
        if self.pfdet_playing and self.pfdet_cap and self.pfdet_cap.isOpened():
            if self.pfdet_seek_pending is not None:
                self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,self.pfdet_seek_pending)
                self.pfdet_history.clear(); self.pfdet_seek_pending=None
            ret,frame=self.pfdet_cap.read()
            if not ret:
                self.pfdet_playing=False; self.pfdet_btn_play.config(text="▶")
                self.pfdet_lbl_status.config(text="Playback complete.")
                self._pfdet_update_overall_verdict()
            else:
                self.pfdet_frame_idx=int(self.pfdet_cap.get(cv2.CAP_PROP_POS_FRAMES))
                tn=time.time()
                self.pfdet_fps_display=(0.9*self.pfdet_fps_display+
                                      0.1/(max(tn-self.pfdet_t_prev,1e-6)))
                self.pfdet_t_prev=tn
                if self.pfdet_frame_idx%FRAME_SKIP==0:
                    sm=cv2.resize(frame,(DETECT_W,DETECT_H),
                                  interpolation=cv2.INTER_LINEAR)
                    results=self.pf_face_detector.process(
                        cv2.cvtColor(sm,cv2.COLOR_BGR2RGB))
                    self.pfdet_last_det=results
                else:
                    results=self.pfdet_last_det
                dsx=self.pfdet_vid_w/DETECT_W; dsy=self.pfdet_vid_h/DETECT_H
                if results and results.detections:
                    lg=max(results.detections,
                           key=lambda d:(d.location_data.relative_bounding_box.width*
                                         d.location_data.relative_bounding_box.height))
                    rb=lg.location_data.relative_bounding_box
                    x1=int(rb.xmin*DETECT_W*dsx); y1=int(rb.ymin*DETECT_H*dsy)
                    bw=int(rb.width*DETECT_W*dsx); bh=int(rb.height*DETECT_H*dsy)
                    px=int(bw*FACE_PADDING); py=int(bh*FACE_PADDING)
                    x1=max(0,x1-px); y1=max(0,y1-py)
                    x2=min(self.pfdet_vid_w,x1+bw+2*px)
                    y2=min(self.pfdet_vid_h,y1+bh+2*py)
                    if x2>x1 and y2>y1:
                        with self.pfdet_infer_queue_lock:
                            self.pfdet_infer_queue.clear()
                            self.pfdet_infer_queue.append(
                                (frame[y1:y2,x1:x2].copy(),(x1,y1,x2,y2)))
                with self.pfdet_infer_lock: res=dict(self.pfdet_infer_result)
                nf=not(results and results.detections)
                lb="NO FACE" if nf else res["label"]
                cc=COLOR_NO_FACE if nf else res["cv_color"]
                hc=C_NEUTRAL if nf else res["hex_color"]
                cf=0.0 if nf else res["confidence"]
                rp=res["raw_prob"]; sm=res["smoothed"]; ms=res["inf_ms"]
                bb=None if nf else res["bbox"]
                if not nf and lb in self.pfdet_verdict_counts:
                    self.pfdet_verdict_counts[lb]+=1
                if len(self.pfdet_frame_log)<50_000:
                    self.pfdet_frame_log.append({
                        "frame":self.pfdet_frame_idx,"face_found":not nf,
                        "raw_prob":round(rp,6),"smoothed":round(sm,6),
                        "label":lb,"confidence":round(cf,2),"inf_ms":round(ms,2)})
                self._pfdet_render_and_show(frame,bb,lb,cc,cf,rp,sm,ms)
                self._pfdet_update_stats(lb,hc,cf,rp,sm,ms)
                self.pfdet_progress_var.set(self.pfdet_frame_idx)
                el=self.pfdet_frame_idx/max(self.pfdet_video_fps,1)
                ts=self.pfdet_total_frames/max(self.pfdet_video_fps,1)
                self.pfdet_lbl_time.config(
                    text=f"{self._fmt_time(el)} / {self._fmt_time(ts)}")
                dl=max(1,int((1000/self.pfdet_video_fps)/self.pfdet_speed))
                self.pfdet_after_id=self.root.after(dl,self._pfdet_update_loop)
                return
        if self.pfdet_loop_active:
            self.pfdet_after_id=self.root.after(33,self._pfdet_update_loop)

    def _pfdet_render(self,frame_bgr):
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        pil=Image.fromarray(rgb)
        imgtk=ImageTk.PhotoImage(image=pil)
        self.pfdet_canvas.imgtk=imgtk
        self.pfdet_canvas.config(image=imgtk)

    def _pfdet_render_and_show(self,frame_bgr,bbox=None,label="",
                              cv_color=COLOR_NO_FACE,confidence=0.0,
                              raw_prob=0.5,smoothed=0.5,inf_ms=0.0):
        frame=frame_bgr.copy() if bbox else frame_bgr
        if bbox:
            x1,y1,x2,y2=bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),cv_color,2)
            tag=f"{label}  {confidence:.1f}%"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+8,y1),cv_color,-1)
            cv2.putText(frame,tag,(x1+4,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0),2)
            bw=x2-x1; fl=int(bw*min(confidence,100)/100)
            cv2.rectangle(frame,(x1,y2+4),(x2,y2+16),(40,40,40),-1)
            cv2.rectangle(frame,(x1,y2+4),(x1+fl,y2+16),cv_color,-1)
        disp=cv2.resize(frame,(self.pfdet_display_w,self.pfdet_display_h),
                        interpolation=cv2.INTER_LINEAR)
        self._pfdet_render(disp)

    def _pfdet_update_stats(self,label,hc,cf,rp,sm,ms):
        self.pfdet_stat_vars["frame"].set(f"{self.pfdet_frame_idx} / {self.pfdet_total_frames}")
        self.pfdet_stat_vars["fps"].set(f"{self.pfdet_fps_display:.1f}")
        self.pfdet_stat_vars["inf_ms"].set(f"{ms:.1f} ms")
        self.pfdet_stat_vars["raw"].set(f"{rp:.4f}")
        self.pfdet_stat_vars["smooth"].set(f"{sm:.4f}")
        total=sum(self.pfdet_verdict_counts.values())
        for v,var in self.pfdet_tally_vars.items():
            n=self.pfdet_verdict_counts[v]
            p=n/total*100 if total>0 else 0
            var.set(f"{n}  ({p:.1f}%)")
        self._pfdet_update_overall_verdict()

    def _pfdet_update_overall_verdict(self):
        total=sum(self.pfdet_verdict_counts.values())
        if total==0:
            self.pfdet_lbl_overall.config(text="—",fg=C_NEUTRAL)
            self.pfdet_lbl_overall_sub.config(text="No inference yet."); return
        dom=max(self.pfdet_verdict_counts,key=self.pfdet_verdict_counts.get)
        pct=self.pfdet_verdict_counts[dom]/total*100
        hm={"REAL":C_REAL,"FAKE":C_FAKE}
        self.pfdet_lbl_overall.config(text=dom,fg=hm[dom])
        self.pfdet_lbl_overall_sub.config(
            text=f"{dom} in {pct:.1f}% of\n{total} inferred frames")

    def _pfdet_toggle_play(self):
        if self.pfdet_cap is None: self._pfdet_open_file(); return
        self.pfdet_playing=not self.pfdet_playing
        self.pfdet_btn_play.config(text="⏸" if self.pfdet_playing else "▶")
        if self.pfdet_playing: self.pfdet_t_prev=time.time()

    def _pfdet_stop(self):
        self.pfdet_playing=False; self.pfdet_btn_play.config(text="▶")
        if self.pfdet_cap:
            self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.pfdet_frame_idx=0; self.pfdet_progress_var.set(0)
            self.pfdet_lbl_time.config(text="0:00 / 0:00")

    def _pfdet_restart(self):
        if self.pfdet_cap:
            self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.pfdet_frame_idx=0; self.pfdet_history.clear()
            self.pfdet_verdict_counts={"REAL":0,"FAKE":0}
            self.pfdet_frame_log=[]; self.pfdet_progress_var.set(0)
            self._pfdet_update_overall_verdict()
            self.pfdet_playing=True; self.pfdet_btn_play.config(text="⏸")
            self.pfdet_t_prev=time.time()

    def _pfdet_on_seek(self,val):
        if self.pfdet_cap: self.pfdet_seek_pending=int(float(val))

    def _pfdet_on_speed_change(self,event=None):
        val=self.pfdet_speed_var.get().replace("×","")
        try: self.pfdet_speed=float(val)
        except: self.pfdet_speed=1.0

    def _pfdet_export_csv(self):
        if not self.pfdet_frame_log:
            messagebox.showinfo("Export","No inference data yet.\nPlay the video first."); return
        path=filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"),("All files","*.*")],
            initialfile="deepfake_results.csv")
        if not path: return
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=self.pfdet_frame_log[0].keys())
            w.writeheader(); w.writerows(self.pfdet_frame_log)
        messagebox.showinfo("Exported",f"Saved {len(self.pfdet_frame_log)} rows to:\n{path}")

    def _pfdet_save_frame(self):
        if self.pfdet_cap is None: return
        pos=int(self.pfdet_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,pos-1))
        ret,frame=self.pfdet_cap.read()
        self.pfdet_cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        if not ret: return
        path=filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files","*.png"),("All files","*.*")],
            initialfile=f"frame_{self.pfdet_frame_idx:05d}.png")
        if path:
            cv2.imwrite(path,frame)
            messagebox.showinfo("Saved",f"Frame saved to:\n{path}")

    @staticmethod

    def _fmt_time(s):
        return f"{int(s)//60}:{int(s)%60:02d}"

    def on_close(self):
        self.det_loop_active=False; self._det_cancel_loop()
        self.det_stop_event.set(); self.det_playing=False
        if self.det_cap: self.det_cap.release()
        self.face_detector.close()

        self.pfdet_loop_active=False; self._pfdet_cancel_loop()
        self.pfdet_stop_event.set(); self.pfdet_playing=False
        if self.pfdet_cap: self.pfdet_cap.release()
        self.pf_face_detector.close()

        plt.close("all"); self.root.destroy()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Defense Demo v8.2")
    parser.add_argument("--model", default=None,
                         help="Enhanced (reputation-weighted) TFLite model path")
    parser.add_argument("--plain-model", default=None,
                         help="Plain FedAvg baseline TFLite model path")
    args = parser.parse_args()
    model_path = resolve_model_path(args.model)
    plain_model_path = resolve_plain_model_path(args.plain_model)
    print(f"[INFO] Using enhanced model: {model_path}")
    print(f"[INFO] Using plain FedAvg model: {plain_model_path}")

    root = tk.Tk()
    app  = DefenseDemo(root, model_path, plain_model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.update_idletasks()
    sw,sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww,wh = 1500, 880
    root.geometry(f"{ww}x{wh}+{max(0,(sw-ww)//2)}+{max(0,(sh-wh)//2)}")
    root.mainloop()


if __name__ == "__main__":
    main()
