"""Object/person detection + follow controller for go2.py (no cloud, no API keys).

Detector: YOLOX-tiny (Apache-2.0, official Megvii ONNX release, trained on COCO's 80 classes) run with onnxruntime
on CPU. `detect_objects` returns everything it sees; `detect` keeps only people for the follow controller.
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
SCORE_MIN = 0.40          # objectness * class score
NMS_IOU = 0.45
MIN_BOX_H = 0.12          # follow mode ignores "people" smaller than 12% of the frame height (far away / false positives)
PERSON = 0

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


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


class ObjectDetector:
    def __init__(self, path: str = MODEL_PATH):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.grid, self.stride = _grid(INPUT_SIZE)

    def detect_objects(self, rgb: np.ndarray, min_score: float = SCORE_MIN) -> list[tuple]:
        """rgb: HxWx3 uint8. Returns [(x1, y1, x2, y2, score, class_id)] in frame pixel coordinates."""
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
        cls_scores = pred[:, 5:] * pred[:, 4:5]                                # (N, 80) objectness * class prob
        cls_id = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(pred)), cls_id]
        keep = score > min_score
        if not keep.any():
            return []
        p, sc, cid = pred[keep], score[keep], cls_id[keep]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x1, x2 = np.clip((cx - bw / 2) / r, 0, w), np.clip((cx + bw / 2) / r, 0, w)
        y1, y2 = np.clip((cy - bh / 2) / r, 0, h), np.clip((cy + bh / 2) / r, 0, h)
        boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxesBatched(boxes, sc.tolist(), cid.tolist(), min_score, NMS_IOU)   # per-class NMS
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(sc[i]), int(cid[i]))
                for i in np.array(idx).reshape(-1)]

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """People only, big enough to follow: [(x1, y1, x2, y2, score)]."""
        return people(self.detect_objects(rgb), rgb.shape[0])


PersonDetector = ObjectDetector  # older name


def people(dets: list[tuple], frame_h: int) -> list[tuple[float, float, float, float, float]]:
    """Filter a detect_objects() result down to followable people."""
    return [d[:5] for d in dets if d[5] == PERSON and (d[3] - d[1]) / frame_h >= MIN_BOX_H]


def summarize(dets: list[tuple]) -> str:
    """'2 person, cup, chair' - counts per class, most frequent first."""
    counts: dict[str, int] = {}
    for d in dets:
        name = COCO_CLASSES[d[5]]
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{n} {k}" if n > 1 else k for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))


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
