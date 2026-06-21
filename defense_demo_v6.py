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

def preprocess_face(face_bgr, dtype, scale, zp):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = tf.cast(face_rgb, tf.float32)
    img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    if dtype == np.uint8:
        img = preprocess_input(img)
        if scale: img = img / scale + zp
        img = tf.clip_by_value(img, 0, 255)
        return np.expand_dims(img.numpy().astype(np.uint8), 0)
    img = preprocess_input(img)
    return np.expand_dims(img.numpy().astype(np.float32), 0)

def classify(s):
    if s >= REAL_THRESHOLD:
        return "REAL",      COLOR_REAL,      C_REAL, s * 100
    if s < FAKE_THRESHOLD:
        return "FAKE",      COLOR_FAKE,      C_FAKE, (1-s) * 100
    conf = (abs(s - 0.575) / 0.375) * 100
    return "UNCERTAIN", COLOR_UNCERTAIN, C_UNC, min(conf, 100)


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
class DefenseDemo:
    def __init__(self, root, model_path):
        self.root       = root
        self.model_path = model_path
        self.root.title("Enhanced FL Cycle — Defense Demo v6")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)
        self.root.minsize(1400, 820)

        self.interpreter, self.inp, self.out = load_tflite_model(model_path)
        self.inp_dtype = self.inp["dtype"]
        self.inp_scale, self.inp_zp = self.inp.get("quantization",(1.0,0))
        self.out_scale, self.out_zp = self.out.get("quantization",(1.0,0))

        mp_fd = mp.solutions.face_detection
        self.face_detector = mp_fd.FaceDetection(
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
        self._build_fl_tab()

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
        tk.Label(badge, text=Path(self.model_path).name, bg=C_CARD2,
                 fg=C_GREEN, font=("Helvetica",8,"bold")).pack(side=tk.LEFT, padx=(4,0))

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
        for verdict, hc in [("REAL",C_REAL),("FAKE",C_FAKE),("UNCERTAIN",C_UNC)]:
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

        self.root.bind("<space>",  lambda e: self._det_toggle_play()
                       if self.nb.index(self.nb.select())==0 else None)
        self.root.bind("<Escape>", lambda e: self._det_stop())
        self.root.bind("<r>",      lambda e: self._det_restart())
        self.root.bind("<s>",      lambda e: self._det_save_frame())

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
        self.det_verdict_counts={"REAL":0,"FAKE":0,"UNCERTAIN":0}
        self.det_frame_log=[]; self.det_last_det=None
        self.det_fps_display=0.0; self.det_t_prev=time.time()
        self.det_vid_w=DISPLAY_MAX_W; self.det_vid_h=DISPLAY_MAX_H
        self.det_display_w=DISPLAY_MAX_W; self.det_display_h=DISPLAY_MAX_H
        self.det_infer_lock=threading.Lock()
        self.det_infer_queue=[]; self.det_infer_queue_lock=threading.Lock()
        self.det_stop_event=threading.Event()

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
        self.det_verdict_counts={"REAL":0,"FAKE":0,"UNCERTAIN":0}
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
                t0=time.perf_counter()
                arr=preprocess_face(crop,self.inp_dtype,self.inp_scale,self.inp_zp)
                self.interpreter.set_tensor(self.inp["index"],arr)
                self.interpreter.invoke()
                ro=self.interpreter.get_tensor(self.out["index"])
                rp=(float((ro[0][0]-self.out_zp)*self.out_scale)
                    if self.inp_dtype==np.uint8 else float(ro[0][0]))
                ms=(time.perf_counter()-t0)*1000
                self.det_history.append(rp)
                sm=float(np.mean(self.det_history))
                lb,cc,hc,cf=classify(sm)
                with self.det_infer_lock:
                    self.det_infer_result.update({
                        "raw_prob":rp,"smoothed":sm,"label":lb,
                        "cv_color":cc,"hex_color":hc,"confidence":cf,
                        "inf_ms":ms,"bbox":bb})
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
        hm={"REAL":C_REAL,"FAKE":C_FAKE,"UNCERTAIN":C_UNC}
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
            self.det_verdict_counts={"REAL":0,"FAKE":0,"UNCERTAIN":0}
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

    @staticmethod
    def _fmt_time(s):
        return f"{int(s)//60}:{int(s)%60:02d}"

    def on_close(self):
        self.det_loop_active=False; self._det_cancel_loop()
        self.det_stop_event.set(); self.det_playing=False
        if self.det_cap: self.det_cap.release()
        self.face_detector.close()
        plt.close("all"); self.root.destroy()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Defense Demo v5")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    model_path = resolve_model_path(args.model)
    print(f"[INFO] Using model: {model_path}")

    root = tk.Tk()
    app  = DefenseDemo(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.update_idletasks()
    sw,sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww,wh = 1500, 880
    root.geometry(f"{ww}x{wh}+{max(0,(sw-ww)//2)}+{max(0,(sh-wh)//2)}")
    root.mainloop()


if __name__ == "__main__":
    main()
