"""Person detection + follow controller for go2.py (no cloud, no API keys).

Detector: YOLOX-tiny (Apache-2.0, official Megvii ONNX release) run with onnxruntime on CPU.
Controller: keeps the chosen person centred (turn) and walks toward them until they fill a set fraction of the
frame (stop). Deliberately simple and SLOW; it has NO obstacle avoidance. The caller (go2.py) supplies the
safety net: confirm key, Space / any drive key / lost focus cancel it, and it auto-cancels when the person is lost.

Weights are not stored in the repo. Fetch once while online:  python go2.py --fetch-model
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
MODEL_PATH = os.path.expanduser(os.environ.get("GO2_FOLLOW_MODEL", "~/.cache/go2/yolox_tiny.onnx"))
INPUT_SIZE = 416          # yolox_tiny's fixed input
SCORE_MIN = 0.40          # objectness * person-class score
NMS_IOU = 0.45
MIN_BOX_H = 0.12          # ignore "people" smaller than 12% of the frame height (far away / false positives)


def model_present() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000


def fetch_model() -> str:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading {MODEL_URL}\n  -> {MODEL_PATH}", flush=True)
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    print(f"done ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)", flush=True)
    return MODEL_PATH


# ---- detector ---------------------------------------------------------------------------------------------
def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell offsets + strides that decode YOLOX's raw head output (same as the official demo_postprocess)."""
    grids, strides = [], []
    for s in (8, 16, 32):
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, n * n, 1), s))
    return np.concatenate(grids, 1).astype(np.float32), np.concatenate(strides, 1).astype(np.float32)


class PersonDetector:
    def __init__(self, path: str = MODEL_PATH):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.grid, self.stride = _grid(INPUT_SIZE)

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """rgb: HxWx3 uint8. Returns [(x1, y1, x2, y2, score)] for people, in frame pixel coordinates."""
        import cv2

        h, w = rgb.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nh, nw = int(h * r), int(w * r)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)   # letterbox, grey pad
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # YOLOX expects BGR, 0-255, no normalisation
        blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        out = self.sess.run(None, {self.input_name: blob})[0].copy()          # (1, N, 85)
        out[..., :2] = (out[..., :2] + self.grid) * self.stride
        out[..., 2:4] = np.exp(out[..., 2:4]) * self.stride
        pred = out[0]
        score = pred[:, 4] * pred[:, 5]                                        # objectness * class 0 (person)
        keep = score > SCORE_MIN
        if not keep.any():
            return []
        p, sc = pred[keep], score[keep]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x1, y1, x2, y2 = (cx - bw / 2) / r, (cy - bh / 2) / r, (cx + bw / 2) / r, (cy + bh / 2) / r
        x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)
        boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxes(boxes, sc.tolist(), SCORE_MIN, NMS_IOU)
        res = []
        for i in np.array(idx).reshape(-1):
            if (y2[i] - y1[i]) / h >= MIN_BOX_H:
                res.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(sc[i])))
        return res


# ---- follow controller ------------------------------------------------------------------------------------
def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FollowConfig:
    max_forward: float = 0.35     # m/s  hard cap on walking speed
    max_turn: float = 0.6         # rad/s hard cap on turning
    k_turn: float = 2.0           # rad/s per unit of horizontal error (error is -0.5..0.5 of frame width)
    turn_deadband: float = 0.05   # ignore errors smaller than 5% of the frame width
    turn_first: float = 0.28      # if the person is further off-centre than this, turn without walking
    k_forward: float = 1.0        # m/s per unit of (target_height - person_height), both as frame fractions
    target_height: float = 0.60   # stop walking when the person fills this fraction of the frame height
    lost_after: float = 1.2       # seconds without a matching person before giving up
    min_iou_or_dist: float = 0.10 # IoU with the last box needed to keep the lock (else centre-distance test)


@dataclass
class FollowResult:
    cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)   # (vx, vy, yaw)
    box: tuple | None = None                            # the tracked person's box, if any
    status: str = ""
    lost: bool = False                                  # True once the target has been missing too long


@dataclass
class Follower:
    cfg: FollowConfig = field(default_factory=FollowConfig)
    last_box: tuple | None = None
    last_seen: float = field(default_factory=time.time)

    def reset(self) -> None:
        self.last_box, self.last_seen = None, time.time()

    def _pick(self, dets, w):
        if not dets:
            return None
        if self.last_box is None:                            # first lock: the biggest person in view
            return max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        best = max(dets, key=lambda d: _iou(d, self.last_box))
        if _iou(best, self.last_box) >= self.cfg.min_iou_or_dist:
            return best
        lc = (self.last_box[0] + self.last_box[2]) / 2       # else: nearest centre, within a quarter frame
        near = min(dets, key=lambda d: abs((d[0] + d[2]) / 2 - lc))
        return near if abs((near[0] + near[2]) / 2 - lc) < 0.25 * w else None

    def step(self, dets, frame_shape, now: float | None = None) -> FollowResult:
        now = time.time() if now is None else now
        h, w = frame_shape[:2]
        c = self.cfg
        tgt = self._pick(dets, w)
        if tgt is None:
            if now - self.last_seen > c.lost_after:
                return FollowResult(status="lost the person", lost=True)
            return FollowResult(status="searching ...")      # brief dropout: stand still, keep the lock
        self.last_box, self.last_seen = tgt, now

        err = ((tgt[0] + tgt[2]) / 2 - w / 2) / w            # + means the person is right of centre
        hr = (tgt[3] - tgt[1]) / h
        yaw = 0.0 if abs(err) < c.turn_deadband else float(np.clip(-c.k_turn * err, -c.max_turn, c.max_turn))
        vx = 0.0
        if abs(err) <= c.turn_first and hr < c.target_height:
            vx = float(np.clip(c.k_forward * (c.target_height - hr), 0.0, c.max_forward))
        status = "close enough" if hr >= c.target_height else ("turning" if abs(err) > c.turn_first else "following")
        return FollowResult(cmd=(vx, 0.0, yaw), box=tgt, status=status)
