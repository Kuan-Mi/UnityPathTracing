"""
Convert glTF/donut/RTXPT scene-graph transforms (right-handed, Y-up) into
Unity transforms (left-handed, Y-up).

This project's glTF models are imported with glTFast, which mirrors the X axis
(M = diag(-1, 1, 1), det = -1). Every transform ported from the scene JSON must
use the SAME convention so it lines up with the imported meshes.

  position (x,y,z)        -> (-x, y, z)
  uniform scale s         -> s   (unchanged)

Rotation uses ONE rule for EVERY object (mesh and camera alike): treat the
object exactly like a camera. Convert its local forward (-Z) and up (+Y) basis
vectors with M (negate-X), then rebuild with Unity's LookRotation(forward, up).
There is no per-object forward difference.

  (Verified: camera quat [-0.044203,0.863331,0.076813,0.496796] -> euler
  [10.1689, 59.8358, 359.9998].)
"""

import json
import math
import sys


# ---------- low-level quaternion / matrix helpers (xyzw order) -------------
def quat_to_mat(q):
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ]


def mat_to_quat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def _mv(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def unity_euler(q):
    """Unity Quaternion.eulerAngles (ZXY order), degrees in [0,360)."""
    x, y, z, w = q
    sinx = max(-1.0, min(1.0, 2 * (w * x - y * z)))
    ex = math.asin(sinx)
    ey = math.atan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))
    ez = math.atan2(2 * (w * z + x * y), 1 - 2 * (x * x + z * z))
    return [math.degrees(a) % 360 for a in (ex, ey, ez)]


M = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]   # negate-X


# ---------- public conversion API -----------------------------------------
def convert_position(p):
    x, y, z = p
    return [-x, y, z]


def convert_rotation(q, forward=(0, 0, -1), up=(0, 1, 0)):
    """One rotation conversion for EVERY object (treat it like a camera).

    Convert the local `forward`(-Z)/`up`(+Y) basis vectors with M (negate-X) and
    rebuild via Unity LookRotation (left-handed: +Z=forward, +Y=up).
    """
    R = quat_to_mat(q)
    fwd = _mv(M, _mv(R, list(forward)))
    upv = _mv(M, _mv(R, list(up)))
    z = _norm(fwd)
    x = _norm(_cross(upv, z))
    y = _cross(z, x)
    Rm = [[x[0], y[0], z[0]],
          [x[1], y[1], z[1]],
          [x[2], y[2], z[2]]]
    return mat_to_quat(Rm)


def fmt(v, nd=6):
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.{nd}f}" for x in v) + "]"
    return f"{v:.{nd}f}"


if __name__ == "__main__":
    # --- self-checks against the values the user verified ---
    print("VERIFY position [7.548285, 1.611793, -4.109409] ->",
          fmt(convert_position([7.548285, 1.611793, -4.109409])))
    cam = convert_rotation([-0.044203, 0.863331, 0.076813, 0.496796])
    print("VERIFY camera rotation -> quat", fmt(cam))
    print("                       -> euler", fmt(unity_euler(cam), 4),
          " (target [10.1689, 59.8358, 359.9998])")
    print()

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            scene = json.load(f)

        def walk(nodes):
            for n in nodes:
                has = any(k in n for k in ("translation", "rotation", "scaling"))
                if has:
                    print(n.get("name", "<unnamed>"))
                    if "translation" in n:
                        print("  position:", fmt(convert_position(n["translation"])))
                    r = n.get("rotation")
                    if r and len(r) == 4:
                        uq = convert_rotation(r)   # same rule for every object
                        print("  rot quat (xyzw):", fmt(uq))
                        print("  rot euler (xyz):", fmt(unity_euler(uq), 3))
                    elif r and len(r) != 4:
                        print("  rotation: SKIPPED (malformed, %d components)" % len(r))
                    if "scaling" in n:
                        print("  scale:", fmt(n["scaling"]))
                    print()
                if "children" in n:
                    walk(n["children"])

        walk(scene.get("graph", []))
