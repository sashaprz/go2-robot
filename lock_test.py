"""Checks that the person lock in follow.py keeps hold of the right person (synthetic pictures: no dog, no detector).

  python lock_test.py        (needs opencv: run it inside the DimOS environment, e.g. via WSL)
Each 'person' is a box with a shirt colour on top and trouser colour below, on a noisy grey background.
"""
from __future__ import annotations

import copy

import numpy as np

import follow

W, H = 1280, 720
RNG = np.random.default_rng(0)
RED_BLUE = ((190, 40, 40), (30, 40, 120))          # red top, dark blue trousers
GREEN_BLACK = ((40, 160, 60), (25, 25, 25))        # green top, black trousers
YELLOW_GREY = ((220, 200, 40), (120, 120, 120))
DARK_JEANS = ((60, 60, 70), (40, 55, 110))         # the kind of thing many people wear ...
DARK_BLACK = ((50, 50, 55), (30, 30, 35))          # ... and a stranger who looks a lot like it
NEAR_RED = ((170, 60, 60), (50, 60, 140))          # a stranger dressed SIMILARLY to RED_BLUE (not identical)


def scene(people, light: float = 1.0):
    """people: [(box, (top rgb, trouser rgb)[, share of the box height that shows the top])] -> an RGB frame."""
    img = np.full((H, W, 3), 150.0)
    for entry in people:
        b5, (top, trousers) = entry[0], entry[1]
        frac = entry[2] if len(entry) > 2 else 0.5
        x1, y1, x2, y2 = b5[:4]
        h = y2 - y1
        img[int(y1):int(y1 + frac * h), int(x1):int(x2)] = top
        img[int(y1 + frac * h):int(y2), int(x1):int(x2)] = trousers
    img = img * light + RNG.normal(0, 12, img.shape)                     # texture / sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def box(cx, h=380, w=140, y=200):
    return (cx - w / 2, y, cx + w / 2, y + h, 0.9)


def lock_on(lock, dets, frame, n=None):
    """Feed the same picture until the lock commits; returns (the picks it made on the way, the final pick)."""
    picks = [lock.choose(dets, frame, None, W, 0.1) for _ in range(n or lock.lock_frames)]
    return picks[:-1], picks[-1]


def main() -> int:
    checks = {}
    a, b = box(640), box(300, h=300, w=110, y=250)
    frame = scene([(a, RED_BLUE), (b, GREEN_BLACK)])

    # --- first lock
    lock = follow.PersonLock()
    early, first = lock_on(lock, [a, b], frame)
    checks["locks on the person nearest the middle, only after holding still for a few frames (not on the first glance)"] = (
        all(p is None for p in early) and first == a)
    print("  locked onto:", lock.label)
    checks["...and describes their clothes (red top, blue trousers)"] = "red" in lock.label and "blue" in lock.label
    big_side = box(1000, h=420, w=160)                             # a bigger person off to the side must not win
    l2 = follow.PersonLock()
    _, pick = lock_on(l2, [a, big_side], scene([(a, RED_BLUE), (big_side, GREEN_BLACK)]))
    checks["a bigger person off to the side doesn't beat the one in front of the dog"] = pick == a

    d_same = follow.signature_distance(lock.ref, follow.signature(scene([(a, RED_BLUE)], light=0.65), a))
    d_other = follow.signature_distance(lock.ref, follow.signature(frame, b))
    d_near = follow.signature_distance(lock.ref, follow.signature(scene([(b, NEAR_RED)]), b))
    d_alike = follow.signature_distance(lock.ref, follow.signature(scene([(b, RED_BLUE)]), b))
    print(f"  colour distances from the lock: same person in dimmer light {d_same:.2f}, differently dressed {d_other:.2f}, "
          f"similar clothes {d_near:.2f}, identical clothes {d_alike:.2f} (accepted below {lock.max_dist})")
    checks["the same person in 35% dimmer light still matches (where they just were: the more forgiving limit)"] = d_same < lock.near_dist
    checks["a differently dressed person does not"] = d_other > lock.near_dist

    # --- two people crossing paths: never the other one
    lock, last, wrong = follow.PersonLock(), None, 0
    pa0 = box(400)
    lock_on(lock, [pa0], scene([(pa0, RED_BLUE)]))
    last = pa0
    for i in range(41):
        pa, pb = box(400 + i * 12), box(880 - i * 12, h=370, w=135)         # they overlap around i = 20
        pick = lock.choose([pa, pb], scene([(pa, RED_BLUE), (pb, GREEN_BLACK)]), last, W, 0.1)
        wrong += pick == pb
        last = pick if pick is not None else last
    checks["two people crossing paths: it never switched to the other one"] = wrong == 0

    # --- the person leaves; a stranger stays
    pick = lock.choose([box(900)], scene([(box(900), GREEN_BLACK)]), last, W, 0.1)
    checks["the locked person leaves and a stranger stays: nobody is chosen (no switching)"] = pick is None and lock.rejected == 1

    # --- plausibility: a similarly dressed stranger somewhere the person could not have got to
    lock = follow.PersonLock()
    lock_on(lock, [a], scene([(a, RED_BLUE)]))
    far = box(1100)
    fr = scene([(far, NEAR_RED)])
    far_no = lock.choose([far], fr, a, W, 0.1, accept=lambda d: False)
    far_yes = lock.choose([far], fr, None, W, 0.1, accept=lambda d: True)
    print(f"  similar-clothes stranger far away: refused when the person couldn't have got there: {far_no is None}; "
          f"accepted if they could ({far_yes is not None}) [colour distance {d_near:.2f}]")
    checks["a similarly dressed stranger who appears where the person could NOT have got to is refused"] = far_no is None

    # --- a look-alike standing right next to the person: keep the target, frame after frame
    lock = follow.PersonLock()
    t0 = box(600)
    lock_on(lock, [t0], scene([(t0, DARK_JEANS)]))
    last, swaps = t0, 0
    for i in range(30):
        tgt, look = box(600 + i * 3), box(800 - i * 3)
        pick = lock.choose([tgt, look], scene([(tgt, DARK_JEANS), (look, DARK_BLACK)]), last, W, 0.1)
        swaps += pick != tgt
        last = pick if pick is not None else last
    checks["a look-alike (dark top, dark trousers) standing beside the target: it stays on the target for 30 frames"] = swaps == 0
    checks["...and the fingerprint bank did not grow from an ambiguous scene"] = len(lock.refs) == 1

    # --- the dog closes in: less and less of the body shows (gradual), so the colours drift, but it's still them
    lock = follow.PersonLock()
    t0 = box(650)
    lock_on(lock, [t0], scene([(t0, RED_BLUE, 0.5)]))
    last, lost = t0, 0
    for i, frac in enumerate([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0] * 2):
        cur = box(650 + (i % 3) * 4, h=380 + i * 8, w=140 + i * 3)
        pick = lock.choose([cur], scene([(cur, RED_BLUE, frac)]), last, W, 0.1)
        lost += pick is None
        last = pick if pick is not None else last
    d_end = follow.signature_distance(lock.ref, follow.signature(scene([(last, RED_BLUE, 0.0)]), last))
    print(f"  close-up, gradual: colour distance of the final view from the FIRST fingerprint {d_end:.2f}; fingerprints kept: {len(lock.refs)}")
    checks["as the view changes gradually (whole body -> legs only) the same person is never dropped"] = lost == 0
    checks["...although the final view alone would fail the strict test against the first fingerprint (the bank of views is what saved it)"] = d_end > lock.max_dist
    swap = box(650)
    sudden = lock.choose([swap], scene([(swap, GREEN_BLACK, 0.5)]), last, W, 0.1)
    checks["but a sudden switch to a differently dressed person in the same spot is refused"] = sudden is None

    # --- lost and found again: the dog turns to look for the person; they come back into view somewhere else, from another angle, in other light
    def tracked():
        lk, t0 = follow.PersonLock(), box(650)
        lock_on(lk, [t0], scene([(t0, RED_BLUE)]))
        return lk

    def found(lk, dets, fr, relax, frames=2):
        """The picks over `frames` consecutive frames while searching (no last position: they can be anywhere)."""
        return [copy.deepcopy(lk).choose(dets, fr, None, W, 0.1, relax=relax)] if frames == 1 else _run(lk, dets, fr, relax, frames)

    def _run(lk, dets, fr, relax, frames):
        lk = copy.deepcopy(lk)
        return [lk.choose(dets, fr, None, W, 0.1, relax=relax) for _ in range(frames)]

    base = tracked()
    legs = box(400, h=640, w=230, y=40)                                   # close up: only the legs are in view
    dim = box(640)
    far_dim = box(500, h=200, w=75, y=260)
    edge = (0, 200, 90, 580, 0.9)                                          # cut by the frame's edge
    p_legs = _run(base, [legs], scene([(legs, RED_BLUE, 0.0)]), 0.0, 2)
    checks["found again from CLOSE UP with only their legs in view (the trousers still match the first fingerprint), straight away"] = p_legs[0] is not None
    p_edge = _run(base, [edge], scene([(edge, RED_BLUE)]), 0.0, 2)
    checks["found again half out of the picture at its edge"] = p_edge[0] is not None
    fr_dim = scene([(dim, RED_BLUE)], light=0.5)
    p0, p1 = _run(base, [dim], fr_dim, 0.0, 2), _run(base, [dim], fr_dim, 1.0, 2)
    print(f"  same person in 50% light: picked at once? {p0[0] is not None}; after searching a while (relax 1): first frame {p1[0] is not None}, second frame {p1[1] is not None}")
    checks["in much dimmer light (the camera's exposure changed) they are found once it has been searching a while, and only after 2 frames in a row"] = p1[0] is None and p1[1] is not None
    p_far = _run(base, [far_dim], scene([(far_dim, RED_BLUE)], light=0.7), 0.5, 2)
    checks["further away and in dimmer light: found"] = p_far[1] is not None
    strangers = {"a similarly dressed stranger": NEAR_RED, "a differently dressed stranger": GREEN_BLACK, "someone in the same trousers but another top": ((40, 160, 60), (30, 40, 120))}
    refused = {k: all(pk is None for pk in _run(base, [dim], scene([(dim, c)]), 1.0, 6)) for k, c in strangers.items()}
    print("  refused after searching as long as it likes (6 frames):", refused)
    checks["...but strangers are never taken for them, however long it has been searching (similar clothes, different clothes, same trousers)"] = all(refused.values())
    a_, b_ = box(400), box(900, h=350, w=125, y=230)
    both = _run(base, [b_, a_], scene([(a_, RED_BLUE), (b_, GREEN_BLACK)]), 1.0, 3)
    checks["with the right person and a stranger in view together it picks the right one"] = all(pk == a_ for pk in both[1:])

    # --- the catalogue of views grows while they are tracked, is capped, and never loses the first fingerprint
    lk = follow.PersonLock()
    t0 = box(650)
    lock_on(lk, [t0], scene([(t0, RED_BLUE, 0.5)]))
    first_ref, last = lk.ref, t0
    sizes = []
    for i in range(240):
        frac = 0.5 - 0.5 * ((i // 4) % 11) / 10                              # 0.5 ... 0.0: whole body to legs only, and around again
        light = 1.0 - 0.25 * ((i // 30) % 3)                                  # ... in three different lights
        h = 380 + 12 * ((i // 4) % 11)
        cur = box(650 + (i % 5) * 3, h=h, w=int(h * 0.37), y=200 + (380 - h) // 2)
        pk = lk.choose([cur], scene([(cur, RED_BLUE, frac)], light=light), last, W, 0.1)
        last = pk if pk is not None else last
        sizes.append(len(lk.refs))
    print(f"  catalogue of views over 240 frames of changing view and light: {sizes[0]} -> {max(sizes)} (cap {lk.max_refs})")
    checks["the catalogue of views grows as the person is tracked through different views and lights"] = max(sizes) >= 3
    checks["...never past its cap, and the first fingerprint is never replaced"] = max(sizes) <= lk.max_refs and lk.ref is first_ref

    # --- no picture: falls back to position only (never crashes)
    checks["without a picture it still works (position only)"] = follow.PersonLock().choose([a, b], None, None, W, 0.1) == a

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
