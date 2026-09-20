"""Object/person detection + follow controller for go2.py (no cloud, no API keys).

Detector: YOLOX-tiny (Apache-2.0, official Megvii ONNX release, trained on COCO's 80 classes) run with onnxruntime
on CPU. `detect_objects` returns everything it sees; `detect` keeps only people for the follow controller.
Controller: keeps the chosen person centred (turn) and walks toward them until they fill a set fraction of the
frame (stop). Deliberately simple and SLOW; it has NO obstacle avoidance. The caller (go2.py) supplies the
safety net: confirm key, Space / any drive key / lost focus cancel it, and it auto-cancels when the person is lost.

Weights are not stored in the repo. Fetch once while online:  python go2.py --fetch-model
"""
from __future__ import annotations

import math
import os
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

import obstacles

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


def pick_target(dets, last_box, w: int, min_iou: float):
    """Which detection is 'my person'. First lock: the biggest one in view. After that: the one overlapping the last
    box most, else the nearest centre within a quarter frame, else nobody (a brief dropout)."""
    if not dets:
        return None
    if last_box is None:
        return max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
    best = max(dets, key=lambda d: _iou(d, last_box))
    if _iou(best, last_box) >= min_iou:
        return best
    lc = (last_box[0] + last_box[2]) / 2
    near = min(dets, key=lambda d: abs((d[0] + d[2]) / 2 - lc))
    return near if abs((near[0] + near[2]) / 2 - lc) < 0.25 * w else None


# ---- who is "my person": clothing-colour lock ---------------------------------------------------------------
# Position alone can't keep hold of someone: when two people cross, the box that overlaps the old one is whoever happens
# to be there. So when the dog locks on it takes a fingerprint of the person's clothes (colour histograms of the upper
# body and of the lower body, from the camera picture) and from then on only accepts a person who matches it. If nobody
# matches, the person counts as "not seen" (the controller coasts, searches, then gives up): it NEVER switches to
# someone else. Limits: two people dressed alike look the same to it, and a big lighting change can make the right person
# stop matching (then it stops rather than guesses).
def _region_hist(rgb: np.ndarray, box, y0: float, y1: float):
    import cv2

    h, w = rgb.shape[:2]
    bx1, by1, bx2, by2 = box[:4]
    bw, bh = bx2 - bx1, by2 - by1
    xa, xb = int(max(0, bx1 + 0.2 * bw)), int(min(w, bx2 - 0.2 * bw))         # the middle of the body: not background at the sides
    ya, yb = int(max(0, by1 + y0 * bh)), int(min(h, by1 + y1 * bh))
    if xb - xa < 3 or yb - ya < 3:
        return None, None
    crop = np.ascontiguousarray(rgb[ya:yb, xa:xb])
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue = (hsv[:, :, 0].astype(np.int16) + 7) % 180                                   # rotate so red (0 and 180) sits mid-bin, not on an edge
    hs = cv2.calcHist([hue.astype(np.uint8), hsv[:, :, 1]], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()   # what colour ...
    v = cv2.calcHist([hsv], [2], None, [3], [0, 256]).flatten()                       # ... and how dark (black/white/grey have no hue)
    hs, v = hs / max(hs.sum(), 1.0), v / max(v.sum(), 1.0)
    ang = hsv[:, :, 0].astype(np.float32) * (2 * np.pi / 180)                         # average hue on the circle (red wraps around)
    mean_hue = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * 180 / (2 * np.pi)
    mean = np.array([mean_hue, hsv[:, :, 1].mean(), hsv[:, :, 2].mean()])
    return np.concatenate((0.85 * hs, 0.15 * v)).astype(np.float32), mean


def signature(rgb: np.ndarray, box):
    """(upper-body histogram, lower-body histogram), or None when the box is too small to read."""
    up, _ = _region_hist(rgb, box, 0.15, 0.50)
    lo, _ = _region_hist(rgb, box, 0.55, 0.90)
    return None if up is None or lo is None else (up, lo)


def signature_distance(a, b) -> float:
    """0 = identical clothes, ~1 = nothing in common (mean Bhattacharyya distance of the two regions)."""
    import cv2

    return float(np.mean([cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)]))


_HUES = ((10, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (100, "teal"), (130, "blue"), (160, "purple"), (172, "pink"), (181, "red"))


def _colour_name(hsv_mean) -> str:
    h, s, v = hsv_mean
    if v < 60:
        return "black"
    if s < 40:
        return "white" if v > 190 else "grey"
    name = next(n for lim, n in _HUES if h < lim)
    return ("dark " if v < 110 else "light " if v > 200 and s < 110 else "") + name


def describe(rgb: np.ndarray, box) -> str:
    """'dark blue top, grey trousers': so the person can check the dog locked onto them."""
    _, up = _region_hist(rgb, box, 0.15, 0.50)
    _, lo = _region_hist(rgb, box, 0.55, 0.90)
    if up is None or lo is None:
        return "someone (too small to read their clothes)"
    return f"{_colour_name(up)} top, {_colour_name(lo)} trousers"


@dataclass
class PersonLock:
    """Chooses which detection is the locked person. Needs the camera picture; without one it falls back to position only.

    Clothes look different as the view changes (whole body far away, only legs and hips up close), so colour alone would
    drop the right person. Two rules: someone who stays right where the person just was is accepted unless they are dressed
    clearly differently (`near_dist`), and the fingerprint keeps adapting to what the camera now sees of the person."""
    max_dist: float = 0.40        # anywhere else in the picture: a stranger must look at least this different to be turned down
    near_dist: float = 0.70       # right where the person just was: only someone clearly different (a stranger walking through) is refused
    ref: tuple | None = None
    label: str = ""               # what it locked onto, e.g. "dark blue top, grey trousers"
    announced: bool = False
    rejected: int = 0             # how many people were turned down on the last call
    closest_reject: float = 1.0   # the best colour match among those turned down (for the "why did it lose me" message)

    def reset(self) -> None:
        self.ref, self.label, self.announced, self.rejected, self.closest_reject = None, "", False, 0, 1.0

    def why_none(self, n_dets: int) -> str:
        if n_dets == 0:
            return "no person detected"
        return f"{self.rejected} person(s) seen but not matching the lock, closest colour match {self.closest_reject:.2f}"

    def choose(self, dets, frame, last_box, w: int, min_iou: float):
        self.rejected, self.closest_reject = 0, 1.0
        if not dets:
            return None
        if self.ref is None or frame is None:
            if self.ref is None:                                   # first lock: the biggest person, favouring the middle of the picture
                tgt = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]) * (1 - 0.6 * abs((d[0] + d[2]) / w - 1)))
                if frame is not None:
                    self.ref = signature(frame, tgt)
                    self.label = describe(frame, tgt) if self.ref is not None else ""
                return tgt
            return pick_target(dets, last_box, w, min_iou)         # no picture this time: position only
        best, best_cost, best_dist = None, 1e9, 1.0
        for d in dets:
            sig = signature(frame, d)
            dist = 0.5 if sig is None else signature_distance(self.ref, sig)
            iou = _iou(d, last_box) if last_box is not None else 0.0
            lc = (last_box[0] + last_box[2]) / 2 if last_box is not None else w / 2
            shift = abs((d[0] + d[2]) / 2 - lc) / (0.25 * w)       # 1.0 = a quarter of the picture away from where they were
            continuous = last_box is not None and (iou > 0.3 or shift < 0.6)
            if dist > (self.near_dist if continuous else self.max_dist):
                self.rejected += 1
                self.closest_reject = min(self.closest_reject, dist)
                continue
            cost = dist + 0.6 * (1 - iou) + 0.5 * min(1.0, shift)
            if cost < best_cost:
                best, best_cost, best_dist = d, cost, dist
        if best is not None:                                       # keep the fingerprint current: faster when the view has changed a lot
            sig = signature(frame, best)
            if sig is not None:
                k = 0.05 if best_dist < 0.25 else 0.15
                self.ref = tuple((1 - k) * r + k * n for r, n in zip(self.ref, sig))
        return best


@dataclass
class Follower:
    cfg: FollowConfig = field(default_factory=FollowConfig)
    last_box: tuple | None = None
    last_seen: float = field(default_factory=time.time)
    lock: PersonLock = field(default_factory=PersonLock)

    def reset(self) -> None:
        self.last_box, self.last_seen = None, time.time()
        self.lock.reset()

    def _pick(self, dets, w, frame=None):
        return self.lock.choose(dets, frame, self.last_box, w, self.cfg.min_iou_or_dist)

    def step(self, dets, frame_shape, now: float | None = None, frame=None) -> FollowResult:
        now = time.time() if now is None else now
        h, w = frame_shape[:2]
        c = self.cfg
        tgt = self._pick(dets, w, frame)
        if tgt is None:
            why = self.lock.why_none(len(dets))
            if now - self.last_seen > c.lost_after:
                return FollowResult(status=f"lost the person: {why}", lost=True)
            return FollowResult(status=f"searching ... ({why})")      # brief dropout: stand still, keep the lock
        self.last_box, self.last_seen = tgt, now

        err = ((tgt[0] + tgt[2]) / 2 - w / 2) / w            # + means the person is right of centre
        hr = (tgt[3] - tgt[1]) / h
        yaw = 0.0 if abs(err) < c.turn_deadband else float(np.clip(-c.k_turn * err, -c.max_turn, c.max_turn))
        vx = 0.0
        if abs(err) <= c.turn_first and hr < c.target_height:
            vx = float(np.clip(c.k_forward * (c.target_height - hr), 0.0, c.max_forward))
        status = "close enough" if hr >= c.target_height else ("turning" if abs(err) > c.turn_first else "following")
        return FollowResult(cmd=(vx, 0.0, yaw), box=tgt, status=status)


# ---- heel controller --------------------------------------------------------------------------------------
# "Heel" = walk at the person's side. The only sensor is the front camera, and that is the catch: it is only ~78 deg
# wide (dimos/robot/unitree/go2/front_camera_720.yaml: 1280x720, fx 797), so a person exactly BESIDE the dog (90 deg)
# is invisible to it. What it can do is keep the person near the edge of the picture on the chosen side and hold that
# spot as they walk, so the dog ends up at their side but roughly a metre BEHIND their hip (`lead`). Getting truly
# level needs another sensor (the dog's lidar, or a wearable tag). How far away they are comes from where their feet
# meet the floor, which needs the camera height/pitch below: those two are ESTIMATES until measured on the real dog.
@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    fx: float = 797.5
    fy: float = 796.5
    cx: float = 643.5
    cy: float = 349.3
    x_off: float = 0.30      # m the lens sits ahead of the dog's centre (DimOS: base_link -> camera_link)
    z: float = 0.35          # m, ESTIMATE: lens height above the floor while standing
    pitch: float = 0.0       # rad, ESTIMATE: how far the camera looks down


@dataclass(frozen=True)
class Placement:
    x: float                 # m ahead of the dog's centre
    y: float                 # m to the dog's left (negative = its right)
    bearing: float           # rad off the lens axis, + = left of the picture's centre
    feet_clipped: bool       # the feet are below the picture: they are closer than x says


BODY_WIDTH = 0.5              # m, a person's width across the shoulders/arms (+-20%): used only when their feet are out of frame
MAX_RANGE = 8.0             # metres: anything at/above the horizon line is "far", not infinite


def locate(box, frame_shape, cam: Camera | None = None) -> Placement:
    """Where a detected person's feet are on the floor, relative to the dog, from the box alone (flat-floor geometry)."""
    cam = cam or Camera()
    h, w = frame_shape[:2]
    s = w / cam.width                                    # same lens, different resolution
    fx, fy, cx, cy = cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s
    X = ((box[0] + box[2]) / 2 - cx) / fx                # ray through the box centre, right-positive
    Y = (min(box[3], h) - cy) / fy                       # ray through the feet row, down-positive
    sp, cp = math.sin(cam.pitch), math.cos(cam.pitch)
    fwd, down = cp - Y * sp, Y * cp + sp                 # the feet ray in level (floor-aligned) axes; lateral is X
    clipped = box[3] >= h - 3
    if down > 0.035 * fwd:
        t = min(cam.z / down, MAX_RANGE / fwd)           # ray length to the floor
    else:
        t = MAX_RANGE / fwd
    bw = box[2] - box[0]
    if clipped and bw > 20 and box[0] > 2 and box[2] < w - 2:
        t = min(t, (fx * BODY_WIDTH / bw) / fwd)         # feet are out of the picture: judge the distance from how wide they are
    elif clipped:
        t *= 0.7
    return Placement(x=cam.x_off + fwd * t, y=-X * t, bearing=math.atan(-X), feet_clipped=clipped)


CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heel_calibration.json")


def solve_camera(samples, frame_h: int = 720, cam: Camera | None = None):
    """Work out the camera height (m) and downward tilt (rad) from people standing at KNOWN distances.

    samples: [(row of the person's feet in the picture, their distance straight ahead of the lens in m), ...]. Two or more
    at clearly different distances give both numbers; a single one assumes the camera is level. Returns (height, tilt, rms
    error in m) or None when the numbers make no physical sense (someone measured wrong, feet not actually visible)."""
    cam = cam or Camera()
    sc = frame_h / cam.height
    fy, cy = cam.fy * sc, cam.cy * sc
    ys = [(row - cy) / fy for row, _ in samples]
    ds = [d for _, d in samples]
    tilts = [0.0] if len(samples) == 1 else [math.radians(t / 4) for t in range(-40, 101)]      # -10 .. +25 degrees
    best = None
    for tilt in tilts:
        sp, cp = math.sin(tilt), math.cos(tilt)
        if any(y * cp + sp <= 1e-3 for y in ys):
            continue
        ks = [(cp - y * sp) / (y * cp + sp) for y in ys]                     # forward distance = height * k
        z = sum(d * k for d, k in zip(ds, ks)) / sum(k * k for k in ks)
        err = sum((d - z * k) ** 2 for d, k in zip(ds, ks)) + 1e-3 * tilt ** 2   # (tiny pull toward level: measurement noise)
        if best is None or err < best[0]:
            best = (err, z, tilt)
    if best is None or not (0.15 <= best[1] <= 0.6) or not (math.radians(-8) <= best[2] <= math.radians(22)):
        return None
    rms = math.sqrt(best[0] / len(samples))
    return None if rms > 0.12 else (best[1], best[2], rms)      # the numbers don't fit any camera: a measurement was wrong


def load_calibration(path: str = CAL_PATH):
    """(camera height m, tilt degrees) saved by the in-app calibration, or None."""
    try:
        import json

        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["cam_height"]), float(d["cam_pitch_deg"])
    except (OSError, ValueError, KeyError):
        return None


def save_calibration(height: float, pitch_deg: float, path: str = CAL_PATH) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cam_height": round(height, 3), "cam_pitch_deg": round(pitch_deg, 2)}, f)


def _dead(e: float, d: float) -> float:
    return 0.0 if abs(e) <= d else e - math.copysign(d, e)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class HeelConfig:
    side: str = "left"           # the side of the PERSON the dog walks on ("left" is the traditional heel side)
    lead: float = 1.3            # m the person is ahead of the dog's centre. Smaller = closer, but more of them is out of the picture
    gap: float = 0.35            # m sideways between them (bigger = more beside you, but nearer the edge of the picture)
    max_forward: float = 0.8     # m/s hard caps
    max_back: float = 0.25
    max_strafe: float = 0.4
    max_turn: float = 1.0        # rad/s
    kx: float = 1.2              # m/s per metre of along-track error
    ix: float = 0.5              # leaky integral term: learns the person's walking speed so the dog doesn't trail them
    ky: float = 1.0              # m/s per metre of sideways error
    k_turn: float = 3.0          # rad/s per rad the person is outside the turn window
    turn_window: float = math.radians(6)    # only turn to keep them in view; the dog otherwise keeps its own heading
    lead_time: float = 0.4       # s: steer toward where they WILL be in the picture (their swing rate x this): makes up for detector + network lag
    dead_x: float = 0.10
    dead_y: float = 0.08
    accel: float = 1.5           # m/s^2 limit on speeding up (slowing down is immediate)
    smooth: float = 0.5          # weight of the newest position measurement (1 = no smoothing)
    coast: float = 0.25          # s a missed detection keeps the last command (no stutter)
    search_turn: float = 0.3     # rad/s: after that, turn slowly toward the side they were last seen
    lost_after: float = 1.5      # s without a matching person before giving up
    min_iou_or_dist: float = 0.10
    cam: Camera = field(default_factory=Camera)
    # lidar: an optional distance sharpener (needs a cloud passed to step()); the CAMERA still decides who and where
    use_lidar: bool = False
    lidar_window: float = 0.6    # m: the lidar may move the camera's distance estimate by at most this much
    beside_hold: float = 6.0     # s: if the camera loses them but the lidar sees someone where they were, stand still this long before giving up
    personal_space: float = 0.55  # m: closer than this the dog backs away from the person, whatever else it is doing
    max_repel: float = 0.5       # m/s: fastest it will back off (a person stepping toward it is faster than the normal backing cap)


@dataclass
class Heeler:
    """Drop-in for Follower (same reset() / step()): keeps the person at a fixed spot beside/ahead of the dog.

    The CAMERA finds the person, decides which one they are, and gives their direction and (from their feet) a distance.
    With use_lidar and a cloud passed to step(), the lidar may sharpen that distance along the same line of sight
    (obstacles.refine_range: compact, isolated blob near the camera's estimate; walls are refused). It never tracks
    anyone on its own: if the camera loses the person the dog stops, it does not go looking for a lidar blob."""
    cfg: HeelConfig = field(default_factory=HeelConfig)
    last_box: tuple | None = None
    last_seen: float = field(default_factory=time.time)
    pos: tuple | None = None                 # smoothed (x, y) of the person in the dog's frame
    acc: float = 0.0                         # leaky integral of the along-track error
    last_cmd: tuple = (0.0, 0.0, 0.0)
    last_bearing: float = 0.0
    last_step: float | None = None
    source: str = "camera"                   # "camera" or "camera+lidar" (the lidar sharpened the distance this step)
    prev_bearing: float | None = None
    prev_bearing_t: float = 0.0
    bearing_rate: float = 0.0                # how fast they swing across the picture, rad/s (smoothed)
    lock: PersonLock = field(default_factory=PersonLock)

    def reset(self) -> None:
        self.last_box, self.last_seen = None, time.time()
        self.pos, self.acc, self.last_cmd, self.last_step = None, 0.0, (0.0, 0.0, 0.0), None
        self.source = "camera"
        self.prev_bearing, self.bearing_rate = None, 0.0
        self.lock.reset()

    @property
    def target(self) -> tuple[float, float]:
        """Where the person should be, in the dog's frame: (ahead, left)."""
        return self.cfg.lead, (-self.cfg.gap if self.cfg.side == "left" else self.cfg.gap)

    def step(self, dets, frame_shape, now: float | None = None, cloud=None, frame=None) -> FollowResult:
        """dets: person boxes from the camera. cloud: lidar points in the dog's frame (x ahead, y left, z up from the
        floor), or None when there is no fresh lidar."""
        now = time.time() if now is None else now
        c = self.cfg
        dt = 0.05 if self.last_step is None else _clip(now - self.last_step, 0.02, 0.5)
        self.last_step = now
        tgt = self.lock.choose(dets, frame, self.last_box, frame_shape[1], c.min_iou_or_dist)
        if tgt is None:
            gone = now - self.last_seen
            why = self.lock.why_none(len(dets))
            if gone <= c.coast:
                return FollowResult(cmd=self.last_cmd, status=f"searching ... ({why})")     # one missed frame: keep walking
            if (c.use_lidar and cloud is not None and self.pos is not None and gone <= c.beside_hold
                    and obstacles.person_near(cloud, self.pos, radius=1.3)):   # they've walked on/back to the dog's side since
                self.last_cmd = (0.0, 0.0, 0.0)                    # the lidar sees someone right where they were: wait, don't give up
                return FollowResult(status="waiting: the lidar sees someone where you were (out of the camera's view)")
            if gone > c.lost_after:
                return FollowResult(status=f"lost the person: {why}", lost=True)
            self.last_cmd = (0.0, 0.0, 0.0)
            return FollowResult(cmd=(0.0, 0.0, math.copysign(c.search_turn, self.last_bearing)), status=f"searching ... ({why})")
        self.last_box, self.last_seen = tgt, now

        p = locate(tgt, frame_shape, c.cam)
        self.last_bearing = p.bearing
        if self.prev_bearing is not None and 0.01 < now - self.prev_bearing_t < 0.3:      # how fast they are swinging across the picture
            gap = now - self.prev_bearing_t
            k = 1 - math.exp(-gap / 0.25)
            self.bearing_rate += k * (_clip((p.bearing - self.prev_bearing) / gap, -2.0, 2.0) - self.bearing_rate)
        elif self.prev_bearing is None or now - self.prev_bearing_t >= 0.3:
            self.bearing_rate = 0.0
        self.prev_bearing, self.prev_bearing_t = p.bearing, now
        x, y, self.source = p.x, p.y, "camera"
        if c.use_lidar and cloud is not None:
            lens_x = c.cam.x_off
            s_cam = math.hypot(p.x - lens_x, p.y)
            s = obstacles.refine_range(cloud, (lens_x, 0.0), math.atan2(p.y, p.x - lens_x), s_cam, window=c.lidar_window)
            if s is not None:                    # same direction as the camera said, distance from the lidar
                k = s / max(s_cam, 1e-3)
                x, y, self.source = lens_x + (p.x - lens_x) * k, p.y * k, "camera+lidar"
        a = c.smooth
        self.pos = (x, y) if self.pos is None else (a * x + (1 - a) * self.pos[0], a * y + (1 - a) * self.pos[1])
        tx, ty = self.target
        ex, ey = self.pos[0] - tx, self.pos[1] - ty                  # + = the person is further ahead / to the left than wanted
        self.acc = _clip(self.acc * (1 - dt / 5.0) + _dead(ex, c.dead_x) * dt, -1.0, 1.0)
        vx = _clip(c.kx * _dead(ex, c.dead_x) + c.ix * self.acc, -c.max_back, c.max_forward)
        vy = _clip(c.ky * _dead(ey, c.dead_y), -c.max_strafe, c.max_strafe)
        d = math.hypot(self.pos[0], self.pos[1])                     # safety first: never walk into them, back off if they step in
        if d < c.personal_space:
            push = _clip(2.0 * (c.personal_space - d) / max(d, 0.1), 0.0, c.max_repel)
            vx = _clip(vx - push * self.pos[0] / max(d, 0.1), -c.max_repel, c.max_forward)
            vy = _clip(vy - push * self.pos[1] / max(d, 0.1), -c.max_strafe, c.max_strafe)
        if self.pos[0] > 0:
            vx = min(vx, max(0.0, d - 0.4))                          # someone in front and close: no forward speed left
        want = math.atan2(ty, tx - c.cam.x_off)                      # the bearing that spot has from the lens
        ahead = p.bearing + self.bearing_rate * c.lead_time             # where they will be in the picture by the time the dog acts
        yaw = _clip(c.k_turn * _dead(ahead - want, c.turn_window), -c.max_turn, c.max_turn)   # turn only to keep them in view
        cmd = tuple(n if abs(n) <= abs(o) or n * o < 0 else o + math.copysign(min(abs(n - o), lim * dt), n - o)
                    for n, o, lim in zip((vx, vy, yaw), self.last_cmd, (c.accel, c.accel, 3.0)))
        self.last_cmd = cmd
        if p.feet_clipped:
            status = "too close"
        elif abs(ex) <= c.dead_x and abs(ey) <= c.dead_y:
            status = "in position"
        else:
            status = "catching up" if ex > c.dead_x else "backing off" if ex < -c.dead_x else "adjusting"
        return FollowResult(cmd=cmd, box=tgt, status=f"{status} ({self.pos[0]:.1f} m ahead, {abs(self.pos[1]):.1f} m to the side) [{self.source}]")
