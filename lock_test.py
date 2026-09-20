"""Checks that the person lock in follow.py keeps hold of the right person (synthetic pictures: no dog, no detector).

  python lock_test.py        (needs opencv: run it inside the DimOS environment, e.g. via WSL)
Each 'person' is a box with a shirt colour on top and trouser colour below, on a noisy grey background.
"""
from __future__ import annotations

import numpy as np

import follow

W, H = 1280, 720
RNG = np.random.default_rng(0)
RED_BLUE = ((190, 40, 40), (30, 40, 120))          # red top, dark blue trousers
GREEN_BLACK = ((40, 160, 60), (25, 25, 25))        # green top, black trousers
YELLOW_GREY = ((220, 200, 40), (120, 120, 120))


def scene(people, light: float = 1.0):
    """people: [(box, (top rgb, trouser rgb))] -> an RGB frame."""
    img = np.full((H, W, 3), 150.0)
    for b5, (top, trousers) in people:
        x1, y1, x2, y2 = b5[:4]
        h = y2 - y1
        img[int(y1):int(y1 + 0.5 * h), int(x1):int(x2)] = top
        img[int(y1 + 0.5 * h):int(y2), int(x1):int(x2)] = trousers
    img = img * light + RNG.normal(0, 12, img.shape)                     # texture / sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def box(cx, h=380, w=140, y=200):
    return (cx - w / 2, y, cx + w / 2, y + h, 0.9)


def main() -> int:
    checks = {}
    lock = follow.PersonLock()
    a, b = box(640), box(300, h=300, w=110, y=250)
    frame = scene([(a, RED_BLUE), (b, GREEN_BLACK)])
    first = lock.choose([a, b], frame, None, W, 0.1)
    checks["locks on the big, central person first"] = first == a
    print("  locked onto:", lock.label)
    checks["...and describes their clothes (red top, blue trousers)"] = "red" in lock.label and "blue" in lock.label

    d_same = follow.signature_distance(lock.ref, follow.signature(scene([(a, RED_BLUE)], light=0.65), a))
    d_other = follow.signature_distance(lock.ref, follow.signature(frame, b))
    d_alike = follow.signature_distance(lock.ref, follow.signature(scene([(b, RED_BLUE)]), b))
    print(f"  distances: same person in dimmer light {d_same:.2f}, a different person {d_other:.2f}, lookalike {d_alike:.2f} (turned down above {lock.max_dist})")
    checks["the same person in 35% dimmer light still matches"] = d_same < lock.max_dist
    checks["a differently dressed person does not"] = d_other > lock.max_dist

    # two people walk toward each other and cross: the lock must stay with A the whole way
    lock, last, wrong, gone = follow.PersonLock(), None, 0, 0
    for i in range(41):
        ax, bx = 400 + i * 12, 880 - i * 12                       # A left->right, B right->left; they overlap around i = 20
        pa, pb = box(ax), box(bx, h=370, w=135)
        pick = lock.choose([pa, pb], scene([(pa, RED_BLUE), (pb, GREEN_BLACK)]), last, W, 0.1)
        if i == 0:
            lock.reset()
            pick = lock.choose([pa], scene([(pa, RED_BLUE)]), None, W, 0.1)        # A is the locked person
        wrong += pick == pb                                       # picking the OTHER person is the failure
        gone += pick is None                                      # (nobody chosen while B stands in front of A is fine)
        last = pick if pick is not None else last
    print(f"  crossing: switched to the other person {wrong} times, no one chosen on {gone} of 41 frames (while blocked)")
    checks["two people crossing paths: it never switched to the other one"] = wrong == 0

    # A leaves the picture while B stays: it must NOT switch to B
    pick = lock.choose([box(900)], scene([(box(900), GREEN_BLACK)]), last, W, 0.1)
    checks["the locked person leaves and a stranger stays: nobody is chosen (no switching)"] = pick is None and lock.rejected == 1

    # a lookalike is a known limit; just show what happens
    lock2 = follow.PersonLock()
    lock2.choose([a], scene([(a, RED_BLUE)]), None, W, 0.1)
    twin = box(900)
    got = lock2.choose([twin], scene([(twin, RED_BLUE)]), a, W, 0.1)
    print(f"  (known limit) a stranger dressed exactly alike, far from where the person was: {'accepted' if got else 'turned down'}")

    # a second person appears right next to A (both in view) and A keeps being chosen
    lock3 = follow.PersonLock()
    pa = box(640)
    lock3.choose([pa], scene([(pa, YELLOW_GREY)]), None, W, 0.1)
    pc = box(760, h=390, w=150)                                   # bigger and closer to the middle than A
    pick = lock3.choose([pa, pc], scene([(pa, YELLOW_GREY), (pc, GREEN_BLACK)]), pa, W, 0.1)
    checks["a bigger stranger appears next to them: still the original person"] = pick == pa

    # no picture: falls back to position only (never crashes)
    lock4 = follow.PersonLock()
    checks["without a picture it still works (position only)"] = lock4.choose([a, b], None, None, W, 0.1) == a

    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
