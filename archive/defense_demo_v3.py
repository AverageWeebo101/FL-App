"""
Defense Demo — Enhanced Federated Learning Cycle  (v3)
=======================================================
Two-tab Tkinter application for thesis defense presentation.

Tab 1 — Deepfake Detector
    Runs the trained TFLite model on a video file (main component).

Tab 2 — FL Simulation Replay
    Animated network diagram canvas showing the FL cycle step-by-step.
    Manual or Auto mode (toggle anytime). Auto pauses at Round 6.
    Chart sidebar shows per-round Accuracy/F1 and AUC (no FL-TENB4 baseline).

Model resolution:
  1. effnet_global_fl_final_quantised.tflite next to this script
  2. Browse dialog fallback

Usage:
    python defense_demo_v3.py
    python defense_demo_v3.py --model path/to/model.tflite
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

# ── Import fix for Pylance ────────────────────────────────────
try:
    from keras.applications.efficientnet import preprocess_input
except ImportError:
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# ── TFLite runtime fallback chain ────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

# ══════════════════════════════════════════════════════════════
# TRAINING DATA  (May 5 2026 run)
# ══════════════════════════════════════════════════════════════
ROUND_DATA = [
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

# ── Palette ───────────────────────────────────────────────────
C_BG        = "#1A1A2E"
C_PANEL     = "#16213E"
C_CARD      = "#0F3460"
C_TEXT      = "#E0E0E0"
C_ACCENT    = "#3A6EA5"
C_GREEN     = "#44CC66"
C_GOLD      = "#F0C040"
C_REAL      = "#44FF44"
C_FAKE      = "#FF4444"
C_UNC       = "#FFD700"
C_NEUTRAL   = "#AAAAAA"
C_IDLE_NODE = "#2A2A4A"
C_SEL_NODE  = "#00B4D8"
C_SERVER    = "#3A6EA5"

# Canvas colours
CV_BG       = "#0D0D1F"
CV_NODE_C   = "#2A3060"   # client node fill (idle)
CV_NODE_S   = "#1A4080"   # server node fill
CV_SEL      = "#00B4D8"   # selected client
CV_TRAIN    = "#4EA8F0"   # training
CV_ARROW_UP = "#F07850"   # update arrow (client→server)
CV_ARROW_DN = "#44CC66"   # broadcast arrow (server→client)
CV_GOLD     = "#F0C040"
CV_TEXT     = "#CCCCEE"

# Detection colours
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

# Animation timing (seconds)
T_SELECT    = 1.1   # selection highlight + lines draw
T_TRAIN     = 1.4   # training bar fill per client
T_SEND      = 1.0   # arrow travel client→server
T_VALIDATE  = 0.9   # checkmark pop
T_AGGREGATE = 1.0   # server pulse
T_REP       = 0.8   # reputation badges
T_AUTO_GAP  = 0.5   # pause between rounds in auto mode


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
        if p.exists():
            return str(p)
    auto = here / default
    if auto.exists():
        return str(auto)
    root_tmp = tk.Tk(); root_tmp.withdraw()
    path = filedialog.askopenfilename(
        title="Locate TFLite model",
        filetypes=[("TFLite model","*.tflite"),("All files","*.*")],
        initialdir=str(here))
    root_tmp.destroy()
    if not path:
        sys.exit(1)
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
    return "UNCERTAIN", COLOR_UNCERTAIN, C_UNC,  min(conf, 100)


# ══════════════════════════════════════════════════════════════
# Network Diagram Canvas
# ══════════════════════════════════════════════════════════════
class NetworkCanvas:
    """
    Draws a live FL network diagram on a tk.Canvas.
    Layout:
        10 client nodes in two arcs (top 5, bottom 5) around a central server node.
    Animated per step:
      select    → selected clients glow + dashed selection lines drawn
      train     → animated progress bar inside each selected client
      send      → orange arrows travel from clients to server (model update)
      validate  → green ✓ or red ✗ pops on each arrow at server end
      aggregate → server pulses gold, "Global Model Updated" text
      rep       → small +/− badges flash on each client
    """

    R_CLIENT = 36      # client node radius
    R_SERVER = 52      # server node radius
    ANIM_FPS = 60

    def __init__(self, parent, width, height):
        self.w = width
        self.h = height
        self.canvas = tk.Canvas(parent, width=width, height=height,
                                bg=CV_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.cx = width  // 2      # server centre x
        self.cy = height // 2 + 10 # server centre y

        # Compute client positions: 5 top arc, 5 bottom arc
        self._client_pos = {}
        top_ids    = ["000","001","002","003","004"]
        bottom_ids = ["005","006","007","008","009"]
        rx, ry = int(width * 0.38), int(height * 0.37)

        for i, cid in enumerate(top_ids):
            angle = math.pi + (math.pi / (len(top_ids)+1)) * (i+1)
            x = self.cx + int(rx * math.cos(angle))
            y = self.cy + int(ry * math.sin(angle))
            self._client_pos[cid] = (x, y)

        for i, cid in enumerate(bottom_ids):
            angle = (math.pi / (len(bottom_ids)+1)) * (i+1)
            x = self.cx + int(rx * math.cos(angle))
            y = self.cy + int(ry * math.sin(angle))
            self._client_pos[cid] = (x, y)

        # Canvas item IDs
        self._node_ovals  = {}   # cid → oval id
        self._node_labels = {}   # cid → text id
        self._rep_labels  = {}   # cid → rep text id
        self._server_oval = None
        self._server_lbl  = None
        self._server_sub  = None
        self._anim_items  = []   # temporary animation items to clean up
        self._train_bars  = {}   # cid → (bg_rect, fill_rect)
        self._anim_after  = None
        self._root        = parent.winfo_toplevel()

        self._draw_static()

    # ── static layout ─────────────────────────────────────────
    def _draw_static(self):
        c = self.canvas
        # Background grid (subtle)
        for x in range(0, self.w, 60):
            c.create_line(x, 0, x, self.h, fill="#1A1A30", width=1)
        for y in range(0, self.h, 60):
            c.create_line(0, y, self.w, y, fill="#1A1A30", width=1)

        # Title
        c.create_text(self.cx, 18, text="Federated Learning Network",
                      fill=CV_TEXT, font=("Helvetica", 11, "bold"))

        # Server node
        sx, sy = self.cx, self.cy
        r = self.R_SERVER
        self._server_oval = c.create_oval(
            sx-r, sy-r, sx+r, sy+r,
            fill=CV_NODE_S, outline="#4A80C0", width=2)
        self._server_lbl = c.create_text(
            sx, sy-10, text="SERVER", fill=CV_TEXT,
            font=("Helvetica", 10, "bold"))
        self._server_sub = c.create_text(
            sx, sy+8, text="Global Model", fill="#8888BB",
            font=("Helvetica", 8))

        # Client nodes
        for cid, (x, y) in self._client_pos.items():
            r = self.R_CLIENT
            ov = c.create_oval(x-r, y-r, x+r, y+r,
                               fill=CV_NODE_C, outline="#4A4A7A", width=1)
            lbl = c.create_text(x, y-7, text=f"C{cid}",
                                fill=CV_TEXT, font=("Helvetica", 8, "bold"))
            rep = c.create_text(x, y+7,
                                text=f"R:{CLIENT_REP[cid]:.3f}",
                                fill="#7777AA", font=("Helvetica", 7))
            self._node_ovals[cid]  = ov
            self._node_labels[cid] = lbl
            self._rep_labels[cid]  = rep

        # Static idle connection lines (very faint)
        for cid, (x, y) in self._client_pos.items():
            c.create_line(x, y, self.cx, self.cy,
                          fill="#222240", width=1, dash=(3,6),
                          tags="idle_line")
        # Send all idle lines to back
        c.tag_lower("idle_line")

    # ── helpers ───────────────────────────────────────────────
    def _clear_anim(self):
        for item in self._anim_items:
            try: self.canvas.delete(item)
            except Exception: pass
        self._anim_items.clear()
        self._train_bars.clear()

    def _cancel_anim(self):
        if self._anim_after:
            try: self._root.after_cancel(self._anim_after)
            except Exception: pass
            self._anim_after = None

    def reset(self):
        self._cancel_anim()
        self._clear_anim()
        c = self.canvas
        for cid, ov in self._node_ovals.items():
            c.itemconfig(ov, fill=CV_NODE_C, outline="#4A4A7A", width=1)
            c.itemconfig(self._node_labels[cid], fill=CV_TEXT)
            c.itemconfig(self._rep_labels[cid], fill="#7777AA")
        c.itemconfig(self._server_oval, fill=CV_NODE_S, outline="#4A80C0", width=2)
        c.itemconfig(self._server_lbl, fill=CV_TEXT)
        c.itemconfig(self._server_sub, text="Global Model", fill="#8888BB")

    def _node_center(self, cid):
        return self._client_pos[cid]

    # ── Step 1: Client Selection ───────────────────────────────
    def animate_select(self, selected, on_done):
        """Highlight selected clients and draw score lines."""
        c = self.canvas
        self._clear_anim()

        # Dim all, highlight selected
        for cid, ov in self._node_ovals.items():
            if cid in selected:
                c.itemconfig(ov, fill=CV_SEL, outline=CV_GOLD, width=3)
                c.itemconfig(self._node_labels[cid], fill="#001A22")
            else:
                c.itemconfig(ov, fill="#181830", outline="#333355", width=1)
                c.itemconfig(self._node_labels[cid], fill="#444466")

        # Animate lines from selected clients to server
        lines  = []
        scores = [0.38, 0.35, 0.27]  # example contribution scores
        for i, cid in enumerate(selected):
            x, y = self._node_center(cid)
            lid = c.create_line(x, y, x, y,
                                fill=CV_GOLD, width=2, dash=(6, 3),
                                arrow=tk.LAST, arrowshape=(10,12,4))
            self._anim_items.append(lid)
            lines.append((lid, x, y, self.cx, self.cy))
            # Score badge
            mid_x = (x + self.cx) // 2
            mid_y = (y + self.cy) // 2
            bid = c.create_text(mid_x, mid_y,
                                text=f"score={scores[i]:.2f}",
                                fill=CV_GOLD, font=("Helvetica", 7, "bold"),
                                state=tk.HIDDEN)
            self._anim_items.append(bid)

        # Selection label on server
        slbl = c.create_text(self.cx, self.cy + self.R_SERVER + 16,
                              text=f"Selecting: C{selected[0]}, C{selected[1]}, C{selected[2]}",
                              fill=CV_GOLD, font=("Helvetica", 8, "bold"))
        self._anim_items.append(slbl)

        steps = 25
        def draw_lines(step=0):
            frac = step / steps
            for lid, x1, y1, x2, y2 in lines:
                nx = x1 + (x2-x1)*frac
                ny = y1 + (y2-y1)*frac
                c.coords(lid, x1, y1, nx, ny)
            if step < steps:
                self._anim_after = self._root.after(
                    int(T_SELECT*1000/steps), lambda s=step+1: draw_lines(s))
            else:
                # Show score badges
                for item in self._anim_items:
                    try:
                        if c.type(item) == "text" and "score=" in (c.itemcget(item,"text") or ""):
                            c.itemconfig(item, state=tk.NORMAL)
                    except Exception:
                        pass
                self._anim_after = self._root.after(300, on_done)
        draw_lines()

    # ── Step 2: Local Training ─────────────────────────────────
    def animate_train(self, selected, on_done):
        """Show training progress bars inside selected clients."""
        c = self.canvas
        # Change server label
        c.itemconfig(self._server_sub, text="Waiting for\nupdates…", fill="#8888BB")

        bars = {}
        for cid in selected:
            x, y = self._node_center(cid)
            r = self.R_CLIENT
            # Bar background
            bg = c.create_rectangle(x-r+4, y+14, x+r-4, y+22,
                                    fill="#111130", outline="#333355")
            # Bar fill (starts empty)
            fill = c.create_rectangle(x-r+4, y+14, x-r+4, y+22,
                                      fill=CV_TRAIN, outline="")
            # "Training…" text above bar
            tlbl = c.create_text(x, y+28,
                                 text="Training…", fill=CV_TRAIN,
                                 font=("Helvetica", 6))
            # Lock icon — data stays private
            lock = c.create_text(x, y-22,
                                 text="🔒 private data",
                                 fill="#888899", font=("Helvetica", 6))
            self._anim_items += [bg, fill, tlbl, lock]
            bars[cid] = (fill, x-r+4, y+14, x+r-4, y+22)
            self._train_bars[cid] = bars[cid]

        steps = 30
        def grow(step=0):
            frac = step / steps
            for cid, (fill, bx1, by1, bx2, by2) in bars.items():
                nx = bx1 + (bx2-bx1)*frac
                c.coords(fill, bx1, by1, nx, by2)
            if step < steps:
                self._anim_after = self._root.after(
                    int(T_TRAIN*1000/steps), lambda s=step+1: grow(s))
            else:
                # Mark done — bar turns green
                for cid, (fill, *_) in bars.items():
                    c.itemconfig(fill, fill=C_GREEN)
                self._anim_after = self._root.after(300, on_done)
        grow()

    # ── Step 3: Send Updates (client → server) ────────────────
    def animate_send(self, selected, on_done):
        """Animate 'Model Update' arrows flying from clients to server."""
        c = self.canvas
        c.itemconfig(self._server_sub, text="Receiving\nupdates…", fill=CV_ARROW_UP)

        arrows = []
        for cid in selected:
            x, y = self._node_center(cid)
            # Travelling dot
            dot = c.create_oval(x-6, y-6, x+6, y+6,
                                fill=CV_ARROW_UP, outline="")
            # Label that travels with it
            lbl = c.create_text(x, y-14, text="Model\nUpdate",
                                fill=CV_ARROW_UP, font=("Helvetica", 6, "bold"))
            self._anim_items += [dot, lbl]
            arrows.append((dot, lbl, x, y, self.cx, self.cy))

        steps = 28
        def travel(step=0):
            frac = step / steps
            ease = frac * frac * (3 - 2*frac)  # smooth-step
            for dot, lbl, x1, y1, x2, y2 in arrows:
                nx = x1 + (x2-x1)*ease
                ny = y1 + (y2-y1)*ease
                c.coords(dot, nx-6, ny-6, nx+6, ny+6)
                c.coords(lbl, nx, ny-14)
            if step < steps:
                self._anim_after = self._root.after(
                    int(T_SEND*1000/steps), lambda s=step+1: travel(s))
            else:
                # Remove travelling items
                for dot, lbl, *_ in arrows:
                    c.delete(dot); c.delete(lbl)
                on_done()
        travel()

    # ── Step 4: Validate ──────────────────────────────────────
    def animate_validate(self, selected, on_done):
        """Show ✓ badges arriving at server — all pass."""
        c = self.canvas
        c.itemconfig(self._server_sub, text="Validating\nupdates…", fill="#A78BFA")
        c.itemconfig(self._server_oval, outline="#A78BFA", width=3)

        checks = []
        offsets = [(-55,-30),(0,-55),(55,-30)]
        for i, cid in enumerate(selected):
            ox, oy = offsets[i]
            bx, by = self.cx+ox, self.cy+oy
            badge = c.create_oval(bx-14, by-14, bx+14, by+14,
                                  fill="#003A00", outline=C_GREEN, width=2)
            chk   = c.create_text(bx, by, text="✓",
                                  fill=C_GREEN, font=("Helvetica", 11, "bold"))
            clbl  = c.create_text(bx, by+22,
                                  text=f"C{cid} OK", fill=C_GREEN,
                                  font=("Helvetica", 6))
            self._anim_items += [badge, chk, clbl]
            checks.append((badge, chk, clbl))

        # Animate scale-in
        for badge, chk, clbl in checks:
            c.itemconfig(badge, state=tk.HIDDEN)
            c.itemconfig(chk,   state=tk.HIDDEN)
            c.itemconfig(clbl,  state=tk.HIDDEN)

        def show_checks(i=0):
            if i < len(checks):
                badge, chk, clbl = checks[i]
                c.itemconfig(badge, state=tk.NORMAL)
                c.itemconfig(chk,   state=tk.NORMAL)
                c.itemconfig(clbl,  state=tk.NORMAL)
                self._anim_after = self._root.after(
                    int(T_VALIDATE*1000/len(checks)), lambda: show_checks(i+1))
            else:
                c.itemconfig(self._server_sub,
                             text="All updates\naccepted ✓", fill=C_GREEN)
                self._anim_after = self._root.after(400, on_done)
        show_checks()

    # ── Step 5: Aggregate ─────────────────────────────────────
    def animate_aggregate(self, selected, on_done):
        """Server pulses gold — global model updated."""
        c = self.canvas
        c.itemconfig(self._server_sub, text="Aggregating\nweights…", fill=CV_GOLD)

        # Weighted arrows from each client converging
        weight_lbls = ["w=0.38", "w=0.35", "w=0.27"]
        for i, cid in enumerate(selected):
            x, y = self._node_center(cid)
            mid_x = (x + self.cx) // 2
            mid_y = (y + self.cy) // 2
            al = c.create_line(x, y, self.cx, self.cy,
                               fill=C_GREEN, width=2,
                               arrow=tk.LAST, arrowshape=(10,12,4))
            wl = c.create_text(mid_x, mid_y, text=weight_lbls[i],
                               fill=C_GREEN, font=("Helvetica", 7, "bold"))
            self._anim_items += [al, wl]

        # Pulse animation on server
        pulse_colors = [CV_GOLD, "#C09020", CV_GOLD, "#E0B030", CV_GOLD]
        def pulse(i=0):
            if i < len(pulse_colors):
                c.itemconfig(self._server_oval, fill=pulse_colors[i], width=3)
                self._anim_after = self._root.after(
                    int(T_AGGREGATE*1000/len(pulse_colors)),
                    lambda: pulse(i+1))
            else:
                c.itemconfig(self._server_oval, fill=CV_NODE_S,
                             outline=CV_GOLD, width=3)
                c.itemconfig(self._server_sub,
                             text="Global Model\nUpdated ✓", fill=CV_GOLD)
                # "Model Update Arrows" broadcast back (server→clients, green)
                for cid2 in ALL_CLIENTS:
                    x2, y2 = self._node_center(cid2)
                    bl = c.create_line(self.cx, self.cy, x2, y2,
                                      fill=CV_ARROW_DN, width=1, dash=(4,4))
                    self._anim_items.append(bl)
                self._anim_after = self._root.after(500, on_done)
        pulse()

    # ── Step 6: Reputation Update ─────────────────────────────
    def animate_reputation(self, selected, round_num, on_done):
        """Flash +/− badges on each client node."""
        c = self.canvas
        badges = []
        for cid in ALL_CLIENTS:
            x, y = self._node_center(cid)
            if cid in selected:
                txt   = "+reward"
                color = C_GREEN
            elif cid == "001":
                txt   = "(inactive)"
                color = "#555577"
            else:
                txt   = "×0.99 decay"
                color = "#888888"

            badge = c.create_text(x, y - self.R_CLIENT - 10,
                                  text=txt, fill=color,
                                  font=("Helvetica", 6, "bold"))
            self._anim_items.append(badge)
            badges.append(badge)

            # Update reputation text
            c.itemconfig(self._rep_labels[cid],
                         text=f"R:{CLIENT_REP[cid]:.3f}", fill="#9999CC")

        # Reset all node colours
        for cid2, ov in self._node_ovals.items():
            if cid2 in selected:
                c.itemconfig(ov, fill=CV_SEL, outline=C_GREEN, width=2)
            else:
                c.itemconfig(ov, fill=CV_NODE_C, outline="#4A4A7A", width=1)

        # Fade badges out after a moment
        def fade():
            for b in badges:
                try: c.delete(b)
                except Exception: pass
            on_done()

        self._anim_after = self._root.after(int(T_REP*1000), fade)

    # ── Round 6 special ───────────────────────────────────────
    def show_best_checkpoint(self):
        c = self.canvas
        c.itemconfig(self._server_oval, fill="#5A4000", outline=CV_GOLD, width=4)
        c.itemconfig(self._server_lbl, fill=CV_GOLD)
        c.itemconfig(self._server_sub,
                     text="★ Best\nCheckpoint!", fill=CV_GOLD)
        # Gold ring pulse
        glow = c.create_oval(
            self.cx - self.R_SERVER - 12,
            self.cy - self.R_SERVER - 12,
            self.cx + self.R_SERVER + 12,
            self.cy + self.R_SERVER + 12,
            outline=CV_GOLD, width=3, fill="")
        self._anim_items.append(glow)

    def round_done_reset(self):
        """Partial reset between rounds — keep nodes visible."""
        c = self.canvas
        # Remove only temporary arrows/badges, keep node highlights
        to_remove = []
        for item in self._anim_items:
            try:
                itype = c.type(item)
                if itype in ("line", "oval") and item != self._server_oval:
                    to_remove.append(item)
                elif itype == "text":
                    txt = c.itemcget(item, "text") or ""
                    if any(k in txt for k in ["score=","Training","private","Model\nUpdate",
                                              "Selecting","Aggregating","Receiving",
                                              "Validating","weight","reward","decay",
                                              "inactive","C0","All updates"]):
                        to_remove.append(item)
            except Exception:
                pass
        for item in to_remove:
            try: c.delete(item)
            except Exception: pass
        self._anim_items = [i for i in self._anim_items if i not in to_remove]
        self._train_bars.clear()


# ══════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════
class DefenseDemo:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root       = root
        self.model_path = model_path
        self.root.title("Enhanced FL Cycle — Defense Demo v3")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)
        self.root.minsize(1280, 760)

        self.interpreter, self.inp, self.out = load_tflite_model(model_path)
        self.inp_dtype = self.inp["dtype"]
        self.inp_scale, self.inp_zp = self.inp.get("quantization", (1.0, 0))
        self.out_scale, self.out_zp = self.out.get("quantization", (1.0, 0))

        mp_fd = mp.solutions.face_detection
        self.face_detector = mp_fd.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        self._build_ui()

    # ── Top-level UI ──────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#0D0D1A", pady=9)
        hdr.pack(fill=tk.X)
        tk.Label(hdr,
                 text="An Enhanced Federated Cycle for DeepFake Detection  ·  Defense Demo",
                 bg="#0D0D1A", fg=C_TEXT, font=("Helvetica", 13, "bold")).pack()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=C_PANEL, foreground=C_TEXT,
                        padding=[22,7],  font=("Helvetica", 11, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C_ACCENT)],
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

        # Toolbar
        tb = tk.Frame(frame, bg=C_PANEL, pady=7)
        tb.pack(fill=tk.X)
        tk.Button(tb, text="📂  Open Video", command=self._det_open_file,
                  bg=C_ACCENT, fg="white", font=("Helvetica",11,"bold"),
                  relief=tk.FLAT, padx=14, pady=5).pack(side=tk.LEFT, padx=(12,6))
        tk.Button(tb, text="💾  Export CSV", command=self._det_export_csv,
                  bg="#2A2A4A", fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="🖼  Save Frame", command=self._det_save_frame,
                  bg="#2A2A4A", fg=C_TEXT, font=("Helvetica",10),
                  relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=4)
        badge = tk.Frame(tb, bg="#0F3460", padx=10, pady=4)
        badge.pack(side=tk.RIGHT, padx=12)
        tk.Label(badge, text="Model:", bg="#0F3460", fg="#888888",
                 font=("Helvetica",8)).pack(side=tk.LEFT)
        tk.Label(badge, text=Path(self.model_path).name,
                 bg="#0F3460", fg=C_GREEN,
                 font=("Helvetica",8,"bold")).pack(side=tk.LEFT, padx=(4,0))

        # Main area
        main = tk.Frame(frame, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8,0))

        self.det_canvas = tk.Label(main, bg="#0A0A0A",
                                    width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.det_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Stats panel
        sp = tk.Frame(main, bg=C_PANEL, width=220)
        sp.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))
        sp.pack_propagate(False)

        tk.Label(sp, text="CURRENT FRAME", bg=C_PANEL, fg="#666688",
                 font=("Helvetica",8,"bold")).pack(pady=(14,2))
        self.det_lbl_verdict = tk.Label(sp, text="—", bg=C_PANEL, fg=C_NEUTRAL,
                                         font=("Helvetica",32,"bold"))
        self.det_lbl_verdict.pack(pady=(2,0))
        self.det_lbl_confidence = tk.Label(sp, text="Confidence: —",
                                            bg=C_PANEL, fg=C_NEUTRAL,
                                            font=("Helvetica",10))
        self.det_lbl_confidence.pack()
        self.det_conf_bar = ttk.Progressbar(sp, orient="horizontal",
                                             length=180, mode="determinate")
        self.det_conf_bar.pack(pady=(6,4))
        self._sep(sp)

        tk.Label(sp, text="INFERENCE STATS", bg=C_PANEL, fg="#666688",
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.det_stat_vars = {}
        for key, lbl in [("frame","Frame"),("fps","Display FPS"),
                          ("inf_ms","Inference ms"),("raw","Raw prob"),
                          ("smooth","Smoothed prob")]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=lbl+":", bg=C_PANEL, fg="#888888",
                     font=("Helvetica",9), anchor="w", width=14).pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            self.det_stat_vars[key] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9,"bold"), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        tk.Label(sp, text="FRAME TALLY", bg=C_PANEL, fg="#666688",
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.det_tally_vars = {}
        for verdict, hc in [("REAL",C_REAL),("FAKE",C_FAKE),("UNCERTAIN",C_UNC)]:
            row = tk.Frame(sp, bg=C_PANEL)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=verdict+":", bg=C_PANEL, fg=hc,
                     font=("Helvetica",9,"bold"), anchor="w", width=10).pack(side=tk.LEFT)
            v = tk.StringVar(value="0  (0.0%)")
            self.det_tally_vars[verdict] = v
            tk.Label(row, textvariable=v, bg=C_PANEL, fg=C_TEXT,
                     font=("Helvetica",9), anchor="e").pack(side=tk.RIGHT)
        self._sep(sp)

        tk.Label(sp, text="OVERALL VERDICT", bg=C_PANEL, fg="#666688",
                 font=("Helvetica",8,"bold")).pack(pady=(4,2))
        self.det_lbl_overall = tk.Label(sp, text="—", bg=C_PANEL, fg=C_NEUTRAL,
                                         font=("Helvetica",22,"bold"))
        self.det_lbl_overall.pack(pady=(2,0))
        self.det_lbl_overall_sub = tk.Label(sp, text="", bg=C_PANEL, fg="#888888",
                                             font=("Helvetica",9),
                                             wraplength=200, justify="center")
        self.det_lbl_overall_sub.pack(padx=8, pady=(0,8))

        # Progress
        pf = tk.Frame(frame, bg=C_BG, pady=4)
        pf.pack(fill=tk.X, padx=10)
        self.det_progress_var = tk.DoubleVar(value=0)
        self.det_progress_bar = ttk.Scale(pf, from_=0, to=100,
                                           orient="horizontal",
                                           variable=self.det_progress_var,
                                           command=self._det_on_seek)
        self.det_progress_bar.pack(fill=tk.X)
        self.det_lbl_time = tk.Label(pf, text="0:00 / 0:00",
                                      bg=C_BG, fg="#888888", font=("Helvetica",9))
        self.det_lbl_time.pack(anchor="e")

        # Controls
        cf = tk.Frame(frame, bg=C_PANEL, pady=8)
        cf.pack(fill=tk.X, side=tk.BOTTOM)
        bc = dict(bg="#2A2A4A", fg=C_TEXT, font=("Helvetica",12),
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
            bg=C_PANEL, fg="#888888", font=("Helvetica",9))
        self.det_lbl_status.pack(side=tk.LEFT, padx=16)

        self.root.bind("<space>",  lambda e: self._det_toggle_play()
                       if self.nb.index(self.nb.select()) == 0 else None)
        self.root.bind("<Escape>", lambda e: self._det_stop())
        self.root.bind("<r>",      lambda e: self._det_restart())
        self.root.bind("<s>",      lambda e: self._det_save_frame())

        # State
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
    # ══════════════════════════════════════════════════════════
    def _build_fl_tab(self):
        frame = tk.Frame(self.nb, bg=C_BG)
        self.nb.add(frame, text="  FL Simulation  ")

        # FL state
        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_mode   = tk.BooleanVar(value=False)
        self.fl_auto_paused = False   # set True when auto pauses at R6
        self.fl_auto_after  = None

        # ── Layout: left (diagram) | right (info + chart) ────
        left = tk.Frame(frame, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                  padx=(10,4), pady=8)

        right = tk.Frame(frame, bg=C_BG, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=8)
        right.pack_propagate(False)

        # ── Network diagram canvas ─────────────────────────────
        self.net_canvas = NetworkCanvas(left, width=720, height=480)

        # ── Right panel: round info + controls + chart ────────
        # Round header card
        hcard = tk.Frame(right, bg=C_CARD, pady=8, padx=12)
        hcard.pack(fill=tk.X, pady=(0,6))
        self.lbl_round = tk.Label(hcard, text="Pre-Training Baseline",
                                   bg=C_CARD, fg=C_TEXT,
                                   font=("Helvetica",13,"bold"), anchor="w")
        self.lbl_round.pack(fill=tk.X)
        self.lbl_phase = tk.Label(hcard,
                                   text="Global model initialised.\nNo FL rounds applied yet.",
                                   bg=C_CARD, fg="#9090AA", font=("Helvetica",8),
                                   anchor="w", wraplength=320, justify="left")
        self.lbl_phase.pack(fill=tk.X, pady=(4,0))

        # Metrics
        mf = tk.Frame(right, bg=C_CARD)
        mf.pack(fill=tk.X, pady=(0,6))
        self.metric_vars   = {}
        self.metric_frames = {}
        for col, (key, lbl) in enumerate([("acc","Accuracy"),
                                           ("f1","F1 Score"),
                                           ("auc","ROC-AUC")]):
            cell = tk.Frame(mf, bg=C_CARD, padx=8, pady=8)
            cell.grid(row=0, column=col, sticky="nsew")
            mf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg=C_CARD, fg="#9090AA",
                     font=("Helvetica",7,"bold")).pack()
            v = tk.StringVar(value="—")
            self.metric_vars[key] = v
            ml = tk.Label(cell, textvariable=v, bg=C_CARD, fg=C_GREEN,
                          font=("Helvetica",17,"bold"))
            ml.pack()
            self.metric_frames[key] = (cell, ml)

        # Round 6 best checkpoint banner (hidden until R6)
        self.best_banner = tk.Frame(right, bg=C_GOLD, pady=0)
        self.lbl_best_title = tk.Label(self.best_banner,
                                        text="★  BEST CHECKPOINT  —  ROUND 6  ★",
                                        bg=C_GOLD, fg="#1A1000",
                                        font=("Helvetica",10,"bold"))
        self.lbl_best_title.pack(pady=(6,2))
        bmf = tk.Frame(self.best_banner, bg=C_GOLD)
        bmf.pack(fill=tk.X, padx=8, pady=(0,6))
        for col,(lbl,val) in enumerate([("Accuracy","96.52%"),
                                         ("F1 Score","96.52%"),
                                         ("ROC-AUC","0.9964")]):
            cell = tk.Frame(bmf, bg="#C08000", padx=10, pady=4)
            cell.grid(row=0, column=col, padx=3, sticky="nsew")
            bmf.columnconfigure(col, weight=1)
            tk.Label(cell, text=lbl, bg="#C08000", fg="#3A2A00",
                     font=("Helvetica",7,"bold")).pack()
            tk.Label(cell, text=val, bg="#C08000", fg="#0A0500",
                     font=("Helvetica",13,"bold")).pack()

        # Step description (minimal — one line)
        self.lbl_step_desc = tk.Label(right,
                                       text="Click 'Next Round' to begin.",
                                       bg=C_BG, fg="#7777AA",
                                       font=("Helvetica",8,"italic"),
                                       wraplength=340, justify="left")
        self.lbl_step_desc.pack(fill=tk.X, pady=(0,4), padx=4)

        # Controls row
        ctrl = tk.Frame(right, bg=C_BG)
        ctrl.pack(fill=tk.X, pady=(0,6))

        self.btn_next = tk.Button(ctrl, text="▶  Next Round",
                                   command=self._fl_next_round,
                                   bg=C_ACCENT, fg="white",
                                   font=("Helvetica",10,"bold"),
                                   relief=tk.FLAT, padx=12, pady=6)
        self.btn_next.pack(side=tk.LEFT, padx=(0,6))

        tk.Button(ctrl, text="⟳  Reset",
                  command=self._fl_reset,
                  bg="#2A2A4A", fg=C_TEXT,
                  font=("Helvetica",9),
                  relief=tk.FLAT, padx=8, pady=6).pack(side=tk.LEFT)

        # Auto toggle
        auto_frame = tk.Frame(right, bg=C_BG)
        auto_frame.pack(fill=tk.X, pady=(0,6))

        self.auto_chk = tk.Checkbutton(
            auto_frame, text="Auto-run  (pauses at Round 6)",
            variable=self.fl_auto_mode,
            command=self._fl_on_auto_toggle,
            bg=C_BG, fg=C_TEXT,
            selectcolor="#2A2A4A",
            font=("Helvetica",9),
            activebackground=C_BG, activeforeground=C_TEXT)
        self.auto_chk.pack(side=tk.LEFT)

        self.lbl_auto_status = tk.Label(auto_frame, text="",
                                         bg=C_BG, fg=C_GOLD,
                                         font=("Helvetica",8,"bold"))
        self.lbl_auto_status.pack(side=tk.LEFT, padx=(8,0))

        # Chart
        self._build_fl_chart(right)

        # Draw initial state
        self._fl_update_display(0)

    def _build_fl_chart(self, parent):
        fig_bg = "#12122A"
        ax_bg  = "#0A0A1E"

        self.fl_fig, self.fl_axes = plt.subplots(
            2, 1, figsize=(3.6, 5.0),
            facecolor=fig_bg, gridspec_kw={"hspace":0.55})
        self.fl_fig.patch.set_facecolor(fig_bg)

        for ax in self.fl_axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors="#888888", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        ax_acc, ax_auc = self.fl_axes

        ax_acc.set_title("Accuracy & F1 per Round",
                          color=C_TEXT, fontsize=8, pad=4, fontweight="bold")
        ax_acc.set_xlabel("Round", color="#888888", fontsize=7)
        ax_acc.set_ylabel("Score", color="#888888", fontsize=7)
        ax_acc.set_xlim(-0.5, 10.5)
        ax_acc.set_ylim(0.92, 0.975)
        ax_acc.set_xticks(range(0,11))
        ax_acc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color="#888888", fontsize=6)

        ax_auc.set_title("ROC-AUC per Round",
                          color=C_TEXT, fontsize=8, pad=4, fontweight="bold")
        ax_auc.set_xlabel("Round", color="#888888", fontsize=7)
        ax_auc.set_ylabel("AUC", color="#888888", fontsize=7)
        ax_auc.set_xlim(-0.5, 10.5)
        ax_auc.set_ylim(0.985, 1.001)
        ax_auc.set_xticks(range(0,11))
        ax_auc.set_xticklabels(["Pre"]+[str(r) for r in range(1,11)],
                                color="#888888", fontsize=6)

        self.line_acc, = ax_acc.plot([], [], color="#4EA8F0",
                                      linewidth=1.8, marker="o",
                                      markersize=4, label="Accuracy", zorder=3)
        self.line_f1,  = ax_acc.plot([], [], color="#F07850",
                                      linewidth=1.2, linestyle="--",
                                      marker="s", markersize=3,
                                      label="F1", zorder=3)
        ax_acc.legend(fontsize=6, facecolor="#1E1E38",
                      labelcolor="#888888", loc="lower right", framealpha=0.8)

        self.line_auc, = ax_auc.plot([], [], color=C_GREEN,
                                      linewidth=1.8, marker="o",
                                      markersize=4, zorder=3)

        # Round 6 vertical markers
        self.vline_acc = ax_acc.axvline(x=6, color=C_GOLD,
                                         linewidth=1.8, linestyle=":", alpha=0.0)
        self.vline_auc = ax_auc.axvline(x=6, color=C_GOLD,
                                         linewidth=1.8, linestyle=":", alpha=0.0)
        self.ann_r6_acc = ax_acc.annotate(
            "★ R6\n96.52%", xy=(6, 0.9652), xytext=(7.5, 0.958),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)
        self.ann_r6_auc = ax_auc.annotate(
            "★ R6\n0.9964", xy=(6, 0.9964), xytext=(7.5, 0.995),
            color=C_GOLD, fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.0), alpha=0.0)

        self.fl_canvas = FigureCanvasTkAgg(self.fl_fig, master=parent)
        self.fl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fl_canvas.draw()

    # ── FL logic ──────────────────────────────────────────────
    def _fl_next_round(self):
        if self.fl_animating:
            return
        next_idx = self.fl_round_index + 1
        if next_idx >= len(ROUND_DATA):
            self.lbl_step_desc.config(
                text="All 10 rounds complete. Best-weight tracking deployed Round 6 model.")
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto()
            return
        self.fl_animating = True
        self.btn_next.config(state=tk.DISABLED)
        self.auto_chk.config(state=tk.DISABLED)
        threading.Thread(target=self._fl_run_round,
                         args=(next_idx,), daemon=True).start()

    def _fl_run_round(self, idx):
        """Runs in background thread — orchestrates step animations via root.after."""
        rd       = ROUND_DATA[idx]
        selected = rd[6]
        rnd      = rd[0]

        # Update round header immediately
        self.root.after(0, lambda: self._fl_set_header(idx))

        # Helper: block until a flag is set by the animation callback
        done_event = threading.Event()
        def on_done():
            done_event.set()

        def run_step(anim_fn, desc):
            done_event.clear()
            self.root.after(0, lambda: self.lbl_step_desc.config(text=desc))
            self.root.after(0, lambda: anim_fn(on_done))
            done_event.wait()

        # Step 1 — Select
        run_step(
            lambda cb: self.net_canvas.animate_select(selected, cb),
            f"Step 1 — Client Selection: C{selected[0]}, C{selected[1]}, C{selected[2]} chosen "
            f"by score = V_i × H_i × R_i  (2-round cooldown enforced)"
        )

        # Step 2 — Train
        run_step(
            lambda cb: self.net_canvas.animate_train(selected, cb),
            "Step 2 — Local Training: each selected client trains 5 epochs on its "
            "private local data. Data never leaves the device."
        )

        # Step 3 — Send (model update arrows)
        run_step(
            lambda cb: self.net_canvas.animate_send(selected, cb),
            "Step 3 — Only model weight updates are sent to the server — "
            "not the raw data. Privacy is preserved."
        )

        # Step 4 — Validate
        run_step(
            lambda cb: self.net_canvas.animate_validate(selected, cb),
            "Step 4 — Update Validation: server checks each update with "
            "L2 norm bound + gain test.  Result: 3 / 3 accepted ✓"
        )

        # Step 5 — Aggregate
        run_step(
            lambda cb: self.net_canvas.animate_aggregate(selected, cb),
            "Step 5 — Weighted Aggregation: updates merged proportional to "
            "each client's contribution score. Global model updated."
        )

        # Step 6 — Reputation
        run_step(
            lambda cb: self.net_canvas.animate_reputation(selected, rnd, cb),
            "Step 6 — Reputation Update: ledger updated for all 10 clients. "
            "Selected (valid) → +reward.  Others → ×0.99 decay."
        )

        # All steps done — update metrics & chart
        self.root.after(0, lambda: self._fl_update_metrics(idx))
        self.root.after(0, lambda: self._fl_update_chart(idx))

        # Special handling for Round 6
        if rnd == 6:
            self.root.after(0, self._fl_show_best_checkpoint)
            self.root.after(0, lambda: self.lbl_step_desc.config(
                text="★  Round 6 — Peak performance! Best checkpoint saved "
                     "(Acc 96.52%, F1 96.52%, AUC 0.9964). "
                     "Subsequent rounds will show convergence decline."))
            self.root.after(0, self._fl_pause_auto_at_r6)
        else:
            self.root.after(0, lambda: self._fl_finish_round(idx))

    def _fl_set_header(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        if rnd == 0:
            self.lbl_round.config(text="Pre-Training Baseline")
            self.lbl_phase.config(
                text="Global model initialised with pre-trained EfficientNetB4 weights.\n"
                     "No federated rounds applied yet.")
        else:
            self.lbl_round.config(text=f"Round {rnd}  /  10")
            self.lbl_phase.config(
                text=f"Selected: C{rd[6][0]}  ·  C{rd[6][1]}  ·  C{rd[6][2]}\n"
                     f"Updates accepted: 3 / 3")

    def _fl_update_metrics(self, idx):
        rd  = ROUND_DATA[idx]
        rnd = rd[0]
        for key, val_str in [("acc", f"{rd[1]*100:.2f}%"),
                              ("f1",  f"{rd[2]*100:.2f}%"),
                              ("auc", f"{rd[3]:.4f}")]:
            self.metric_vars[key].set(val_str)
            cell, lbl = self.metric_frames[key]
            color = C_GOLD if rnd == 6 else C_GREEN
            bg    = "#1A1500" if rnd == 6 else C_CARD
            lbl.config(fg=color)
            cell.config(bg=bg)

    def _fl_update_chart(self, idx):
        xs   = [ROUND_DATA[i][0] for i in range(idx+1)]
        accs = [ROUND_DATA[i][1] for i in range(idx+1)]
        f1s  = [ROUND_DATA[i][2] for i in range(idx+1)]
        aucs = [ROUND_DATA[i][3] for i in range(idx+1)]
        self.line_acc.set_data(xs, accs)
        self.line_f1.set_data(xs, f1s)
        self.line_auc.set_data(xs, aucs)
        if ROUND_DATA[idx][0] >= 6:
            self.vline_acc.set_alpha(0.9)
            self.vline_auc.set_alpha(0.9)
            self.ann_r6_acc.set_alpha(1.0)
            self.ann_r6_auc.set_alpha(1.0)
        self.fl_canvas.draw_idle()

    def _fl_show_best_checkpoint(self):
        self.best_banner.pack(fill=tk.X, pady=(0,6),
                              after=self.metric_frames["auc"][0].master)
        self.net_canvas.show_best_checkpoint()

    def _fl_finish_round(self, idx):
        self.net_canvas.round_done_reset()
        self.fl_round_index = idx
        self.fl_animating   = False
        self.auto_chk.config(state=tk.NORMAL)

        rnd = ROUND_DATA[idx][0]
        if idx < len(ROUND_DATA) - 1:
            self.btn_next.config(state=tk.NORMAL)
        else:
            self.btn_next.config(state=tk.DISABLED)
            self._fl_disable_auto()
            return

        # Auto-advance if enabled
        if self.fl_auto_mode.get() and not self.fl_auto_paused:
            self.fl_auto_after = self.root.after(
                int(T_AUTO_GAP * 1000), self._fl_next_round)

    def _fl_pause_auto_at_r6(self):
        """Called after Round 6 completes — pause auto, let panelists absorb."""
        self.net_canvas.round_done_reset()
        self.fl_round_index = 6   # index into ROUND_DATA where rnd==6
        self.fl_animating   = False
        self.fl_auto_paused = True
        self.lbl_auto_status.config(text="⏸ Paused at Round 6")
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)

    def _fl_on_auto_toggle(self):
        if self.fl_auto_mode.get():
            self.lbl_auto_status.config(text="▶ Auto running…")
            self.fl_auto_paused = False
            # Start auto-run if not currently animating and not finished
            if not self.fl_animating and self.fl_round_index < len(ROUND_DATA)-1:
                self.fl_auto_after = self.root.after(
                    int(T_AUTO_GAP * 1000), self._fl_next_round)
        else:
            self.lbl_auto_status.config(text="")
            if self.fl_auto_after:
                try: self.root.after_cancel(self.fl_auto_after)
                except Exception: pass
                self.fl_auto_after = None

    def _fl_disable_auto(self):
        self.fl_auto_mode.set(False)
        self.lbl_auto_status.config(text="")
        if self.fl_auto_after:
            try: self.root.after_cancel(self.fl_auto_after)
            except Exception: pass
            self.fl_auto_after = None

    def _fl_reset(self):
        self._fl_disable_auto()
        if self.fl_auto_after:
            try: self.root.after_cancel(self.fl_auto_after)
            except Exception: pass
        self.fl_round_index = 0
        self.fl_animating   = False
        self.fl_auto_paused = False
        self.btn_next.config(state=tk.NORMAL)
        self.auto_chk.config(state=tk.NORMAL)

        self.net_canvas.reset()

        self.line_acc.set_data([], [])
        self.line_f1.set_data([], [])
        self.line_auc.set_data([], [])
        self.vline_acc.set_alpha(0.0)
        self.vline_auc.set_alpha(0.0)
        self.ann_r6_acc.set_alpha(0.0)
        self.ann_r6_auc.set_alpha(0.0)
        self.fl_canvas.draw_idle()

        for key in ("acc","f1","auc"):
            self.metric_vars[key].set("—")
            cell, lbl = self.metric_frames[key]
            lbl.config(fg=C_GREEN); cell.config(bg=C_CARD)

        try: self.best_banner.pack_forget()
        except Exception: pass

        self.lbl_round.config(text="Pre-Training Baseline")
        self.lbl_phase.config(
            text="Global model initialised.\nNo FL rounds applied yet.")
        self.lbl_step_desc.config(text="Click 'Next Round' to begin.")
        self._fl_update_display(0)

    def _fl_update_display(self, idx):
        rd = ROUND_DATA[idx]
        self.metric_vars["acc"].set(f"{rd[1]*100:.2f}%")
        self.metric_vars["f1"].set(f"{rd[2]*100:.2f}%")
        self.metric_vars["auc"].set(f"{rd[3]:.4f}")

    # ── Detection helpers (unchanged from v2) ─────────────────
    @staticmethod
    def _sep(p):
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

    def _det_show_placeholder(self):
        img = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), dtype=np.uint8)
        img[:] = (20, 20, 35)
        cv2.putText(img, "No video loaded",
                    (DISPLAY_MAX_W//2-140, DISPLAY_MAX_H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80,80,130), 2)
        cv2.putText(img, "Click  Open Video  to begin",
                    (DISPLAY_MAX_W//2-200, DISPLAY_MAX_H//2+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55,55,90), 1)
        self._det_render(img)

    def _det_open_file(self):
        p = filedialog.askopenfilename(title="Select a video file",
                                        filetypes=VIDEO_EXTS)
        if p: self._det_load_video(p)

    def _det_load_video(self, path):
        self._det_cancel_loop()
        self._det_stop()
        if self.det_cap: self.det_cap.release()
        self.det_cap = cv2.VideoCapture(path)
        if not self.det_cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}"); return
        self.det_video_path   = path
        self.det_total_frames = int(self.det_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.det_video_fps    = self.det_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.det_frame_idx    = 0
        self.det_history.clear()
        self.det_raw_prob=0.5; self.det_smoothed=0.5
        self.det_label="NO FACE"; self.det_cv_color=COLOR_NO_FACE
        self.det_hex_color=C_NEUTRAL; self.det_confidence=0.0
        self.det_inf_ms=0.0; self.det_bbox=None
        self.det_verdict_counts={"REAL":0,"FAKE":0,"UNCERTAIN":0}
        self.det_frame_log=[]; self.det_last_det=None
        self.det_fps_display=0.0; self.det_t_prev=time.time()
        vid_w=int(self.det_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h=int(self.det_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale=min(DISPLAY_MAX_W/vid_w, DISPLAY_MAX_H/vid_h, 1.0)
        self.det_display_w=int(vid_w*scale); self.det_display_h=int(vid_h*scale)
        self.det_vid_w=vid_w; self.det_vid_h=vid_h
        self.det_progress_bar.config(to=self.det_total_frames)
        fname=Path(path).name
        self.det_lbl_status.config(
            text=f"{fname}  |  {vid_w}×{vid_h}  |  {self.det_video_fps:.1f} fps  |  {self.det_total_frames} frames")
        self.root.title(f"Enhanced FL Demo — {fname}")
        self._det_update_overall_verdict()
        ret,first=self.det_cap.read()
        if ret: self._det_render_and_show(first)
        self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.det_after_id=self.root.after(33, self._det_update_loop)

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
                if crop is None:
                    time.sleep(0.005); continue
                t0=time.perf_counter()
                inp_arr=preprocess_face(crop,self.inp_dtype,self.inp_scale,self.inp_zp)
                self.interpreter.set_tensor(self.inp["index"],inp_arr)
                self.interpreter.invoke()
                raw_out=self.interpreter.get_tensor(self.out["index"])
                raw_prob=(float((raw_out[0][0]-self.out_zp)*self.out_scale)
                          if self.inp_dtype==np.uint8 else float(raw_out[0][0]))
                inf_ms=(time.perf_counter()-t0)*1000
                self.det_history.append(raw_prob)
                smoothed=float(np.mean(self.det_history))
                label,cv_color,hex_color,confidence=classify(smoothed)
                with self.det_infer_lock:
                    self.det_infer_result.update({
                        "raw_prob":raw_prob,"smoothed":smoothed,
                        "label":label,"cv_color":cv_color,
                        "hex_color":hex_color,"confidence":confidence,
                        "inf_ms":inf_ms,"bbox":bb})
        threading.Thread(target=worker, daemon=True).start()

    def _det_cancel_loop(self):
        if self.det_after_id:
            try: self.root.after_cancel(self.det_after_id)
            except Exception: pass
            self.det_after_id=None

    def _det_update_loop(self):
        if self.det_playing and self.det_cap and self.det_cap.isOpened():
            if self.det_seek_pending is not None:
                self.det_cap.set(cv2.CAP_PROP_POS_FRAMES, self.det_seek_pending)
                self.det_history.clear(); self.det_seek_pending=None
            ret,frame_bgr=self.det_cap.read()
            if not ret:
                self.det_playing=False
                self.det_btn_play.config(text="▶")
                self.det_lbl_status.config(text="Playback complete.")
                self._det_update_overall_verdict()
            else:
                self.det_frame_idx=int(self.det_cap.get(cv2.CAP_PROP_POS_FRAMES))
                t_now=time.time()
                self.det_fps_display=(0.9*self.det_fps_display+
                                      0.1*(1/(max(t_now-self.det_t_prev,1e-6))))
                self.det_t_prev=t_now
                if self.det_frame_idx%FRAME_SKIP==0:
                    small=cv2.resize(frame_bgr,(DETECT_W,DETECT_H),
                                     interpolation=cv2.INTER_LINEAR)
                    results=self.face_detector.process(
                        cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
                    self.det_last_det=results
                else:
                    results=self.det_last_det
                dsx=self.det_vid_w/DETECT_W; dsy=self.det_vid_h/DETECT_H
                if results and results.detections:
                    largest=max(results.detections,
                                key=lambda d:(d.location_data.relative_bounding_box.width*
                                              d.location_data.relative_bounding_box.height))
                    rb=largest.location_data.relative_bounding_box
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
                                (frame_bgr[y1:y2,x1:x2].copy(),(x1,y1,x2,y2)))
                with self.det_infer_lock: res=dict(self.det_infer_result)
                no_face=not(results and results.detections)
                label="NO FACE" if no_face else res["label"]
                cv_color=COLOR_NO_FACE if no_face else res["cv_color"]
                hex_color=C_NEUTRAL if no_face else res["hex_color"]
                confidence=0.0 if no_face else res["confidence"]
                raw_prob=res["raw_prob"]; smoothed=res["smoothed"]
                inf_ms=res["inf_ms"]
                bbox=None if no_face else res["bbox"]
                if not no_face and label in self.det_verdict_counts:
                    self.det_verdict_counts[label]+=1
                if len(self.det_frame_log)<50_000:
                    self.det_frame_log.append({
                        "frame":self.det_frame_idx,"face_found":not no_face,
                        "raw_prob":round(raw_prob,6),"smoothed":round(smoothed,6),
                        "label":label,"confidence":round(confidence,2),
                        "inf_ms":round(inf_ms,2)})
                self._det_render_and_show(frame_bgr,bbox,label,cv_color,
                                          confidence,raw_prob,smoothed,inf_ms)
                self._det_update_stats(label,hex_color,confidence,
                                       raw_prob,smoothed,inf_ms)
                self.det_progress_var.set(self.det_frame_idx)
                elapsed=self.det_frame_idx/max(self.det_video_fps,1)
                total_s=self.det_total_frames/max(self.det_video_fps,1)
                self.det_lbl_time.config(
                    text=f"{self._fmt_time(elapsed)} / {self._fmt_time(total_s)}")
                delay=max(1,int((1000/self.det_video_fps)/self.det_speed))
                self.det_after_id=self.root.after(delay,self._det_update_loop)
                return
        if self.det_loop_active:
            self.det_after_id=self.root.after(33,self._det_update_loop)

    def _det_render(self, frame_bgr):
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
            bw=x2-x1; filled=int(bw*min(confidence,100)/100)
            cv2.rectangle(frame,(x1,y2+4),(x2,y2+16),(40,40,40),-1)
            cv2.rectangle(frame,(x1,y2+4),(x1+filled,y2+16),cv_color,-1)
        disp=cv2.resize(frame,(self.det_display_w,self.det_display_h),
                        interpolation=cv2.INTER_LINEAR)
        self._det_render(disp)

    def _det_update_stats(self,label,hex_color,confidence,raw_prob,smoothed,inf_ms):
        self.det_lbl_verdict.config(text=label,fg=hex_color)
        self.det_lbl_confidence.config(
            text=f"Confidence: {confidence:.1f}%",fg=hex_color)
        self.det_conf_bar["value"]=min(confidence,100)
        self.det_stat_vars["frame"].set(
            f"{self.det_frame_idx} / {self.det_total_frames}")
        self.det_stat_vars["fps"].set(f"{self.det_fps_display:.1f}")
        self.det_stat_vars["inf_ms"].set(f"{inf_ms:.1f} ms")
        self.det_stat_vars["raw"].set(f"{raw_prob:.4f}")
        self.det_stat_vars["smooth"].set(f"{smoothed:.4f}")
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
        if self.det_cap is None:
            self._det_open_file(); return
        self.det_playing=not self.det_playing
        self.det_btn_play.config(text="⏸" if self.det_playing else "▶")
        if self.det_playing: self.det_t_prev=time.time()

    def _det_stop(self):
        self.det_playing=False
        self.det_btn_play.config(text="▶")
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.det_frame_idx=0
            self.det_progress_var.set(0)
            self.det_lbl_time.config(text="0:00 / 0:00")

    def _det_restart(self):
        if self.det_cap:
            self.det_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            self.det_frame_idx=0
            self.det_history.clear()
            self.det_verdict_counts={"REAL":0,"FAKE":0,"UNCERTAIN":0}
            self.det_frame_log=[]
            self.det_progress_var.set(0)
            self._det_update_overall_verdict()
            self.det_playing=True
            self.det_btn_play.config(text="⏸")
            self.det_t_prev=time.time()

    def _det_on_seek(self,val):
        if self.det_cap: self.det_seek_pending=int(float(val))

    def _det_on_speed_change(self,event=None):
        val=self.det_speed_var.get().replace("×","")
        try: self.det_speed=float(val)
        except ValueError: self.det_speed=1.0

    def _det_export_csv(self):
        if not self.det_frame_log:
            messagebox.showinfo("Export","No inference data yet.\nPlay the video first.")
            return
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
        self.det_loop_active=False
        self._det_cancel_loop()
        self.det_stop_event.set()
        self.det_playing=False
        if self.det_cap: self.det_cap.release()
        self.face_detector.close()
        plt.close("all")
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Defense Demo v3")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"[INFO] Using model: {model_path}")

    root = tk.Tk()
    app  = DefenseDemo(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = 1440, 840
    root.geometry(f"{ww}x{wh}+{max(0,(sw-ww)//2)}+{max(0,(sh-wh)//2)}")
    root.mainloop()


if __name__ == "__main__":
    main()
