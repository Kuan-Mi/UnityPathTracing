"""Decode and compare the analytic light list of the raytracing passes in the two Other/ captures.

RTXPT stores lights in three SRVs bound on the DXR global root signature (see
Shaders/Bindings/LightingBindings.hlsli):
    t12  StructuredBuffer<LightingControlData>      -- light counts (TotalLightCount, AnalyticLightCount, ...)
    t13  StructuredBuffer<PolymorphicLightInfo>     -- 32-byte packed light records
    t14  StructuredBuffer<PolymorphicLightInfoEx>   -- 16-byte shaping (spot) records

describe_dispatch only tracks ->Dispatch, not ->DispatchRays, so (like compare_materials.py) we
walk the exported command list: for the PathTrace DispatchRays we resolve the active
SetComputeRootDescriptorTable whose root-sig SRV range covers each register, map to a heap slot,
read the bound resource id, and pull its bytes from the whole-capture export.

The PolymorphicLight packing (UnpackColor / UnpackRadiance / oct direction / f16 scalars / light
shaping) is replicated here exactly from PolymorphicLight.hlsli + LightShaping.hlsli +
Utils.hlsli, so the decoded values are what the path tracer itself sees.

Run from tools/wpix_mcp:  python compare_lights.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, glob, math, os, re, struct, sys
import wpix_core as w

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt-bistro.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity-bistro.wpix")

LIGHT_TYPES = {0: "Sphere", 1: "Triangle", 2: "Directional", 3: "Environment",
               4: "Point", 5: "EnvironmentQuad"}

SHAPING_ENABLE_BIT   = 1 << 28
IES_ENABLE_BIT       = 1 << 29
USE_MIN_FALLOFF_BIT  = 1 << 30
TYPE_SHIFT = 24
TYPE_MASK  = 0xf

MIN_LOG2 = -8.0
MAX_LOG2 = 40.0


# ---- packing helpers, ported 1:1 from the RTXPT shaders -------------------------------------

def f16(u16):
    """f16tof32 of the low 16 bits."""
    return struct.unpack("<e", struct.pack("<H", u16 & 0xffff))[0]


def unpack_r8(v):
    return (v & 0xff) / 255.0


def unpack_color(color_type_flags, log_radiance):
    rgb = (unpack_r8(color_type_flags), unpack_r8(color_type_flags >> 8), unpack_r8(color_type_flags >> 16))
    lr = log_radiance & 0xffff
    radiance = 0.0 if lr == 0 else 2.0 ** ((float(lr - 1) / 65534.0) * (MAX_LOG2 - MIN_LOG2) + MIN_LOG2)
    return tuple(c * radiance for c in rgb), radiance, rgb


def decode_oct(fx, fy):
    """Decode_Oct(float2) — input already in [-1,1] (OctToNDirUnorm32 pre-scales)."""
    fx = fx * 2.0 - 1.0
    fy = fy * 2.0 - 1.0
    nx, ny = fx, fy
    nz = 1.0 - abs(fx) - abs(fy)
    t = max(0.0, min(1.0, -nz))
    nx += -t if nx >= 0.0 else t
    ny += -t if ny >= 0.0 else t
    n = (nx, ny, nz)
    ln = math.sqrt(sum(c * c for c in n)) or 1.0
    return tuple(c / ln for c in n)


def oct_to_ndir_unorm32(p):
    ux = max(0.0, min(1.0, (p & 0xffff) / 0xfffe))
    uy = max(0.0, min(1.0, (p >> 16) / 0xfffe))
    return decode_oct(ux * 2.0 - 1.0, uy * 2.0 - 1.0)


def decode_light(base, ex):
    """base = 32 bytes PolymorphicLightInfo, ex = 16 bytes PolymorphicLightInfoEx."""
    cx, cy, cz = struct.unpack_from("<3f", base, 0)
    color_type_flags = struct.unpack_from("<I", base, 12)[0]
    direction1, direction2, scalars, log_radiance = struct.unpack_from("<4I", base, 16)

    ltype = (color_type_flags >> TYPE_SHIFT) & TYPE_MASK
    radiance, intensity, hue = unpack_color(color_type_flags, log_radiance)
    has_shaping = bool(color_type_flags & SHAPING_ENABLE_BIT)

    d = {
        "type": ltype, "type_name": LIGHT_TYPES.get(ltype, f"?{ltype}"),
        "center": (cx, cy, cz), "radiance": radiance, "intensity": intensity, "hue": hue,
        "has_shaping": has_shaping,
        "use_min_falloff": bool(color_type_flags & USE_MIN_FALLOFF_BIT),
        "ies": bool(color_type_flags & IES_ENABLE_BIT),
        "raw": (color_type_flags, direction1, direction2, scalars, log_radiance),
    }
    if ltype == 0:    # Sphere
        d["radius"] = f16(scalars)
    elif ltype == 4:  # Point  (radius==0 sphere; direction/angles unused unless shaped)
        d["radius"] = 0.0
        d["direction"] = oct_to_ndir_unorm32(direction1)
        d["outer_angle"] = f16(direction2)
        d["inner_angle"] = f16(direction2 >> 16)
    elif ltype == 2:  # Directional
        d["direction"] = oct_to_ndir_unorm32(direction1)
        half = f16(scalars)
        d["angular_size"] = 2 * half
        d["solid_angle"] = f16(scalars >> 16)

    if has_shaping and ex is not None:
        ies_idx, primary_axis, cos_cone_and_soft, unique_id = struct.unpack_from("<4I", ex, 0)
        cos_cone = f16(cos_cone_and_soft)
        cos_soft = f16(cos_cone_and_soft >> 16)
        d["spot_axis"] = oct_to_ndir_unorm32(primary_axis)
        d["cos_cone_angle"] = cos_cone
        d["cos_cone_softness"] = cos_soft
        d["cone_half_angle_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cos_cone))))
        d["unique_id"] = unique_id
    return d


# ---- capture buffer resolution --------------------------------------------------------------

def srv_buffer_ids(ed, registers):
    """{register: resource_id} resolved at the PathTrace DispatchRays for the given SRV registers."""
    text = ""
    for f in glob.glob(os.path.join(ed.dir, "CommandLists_*.cpp")):
        text += open(f, encoding="utf-8", errors="replace").read()
    cur_sig = None
    tables = {}
    found = {}
    for ln in text.split("\n"):
        m = re.search(r"SetComputeRootSignature\(GetRootSignature\((\d+)\)", ln)
        if m:
            cur_sig = int(m.group(1))
        m = re.search(r"SetComputeRootDescriptorTable\((\d+),\s*GetGpuDescriptor\([^,]+,\s*(\d+)\)", ln)
        if m:
            tables[int(m.group(1))] = int(m.group(2))
        if "->DispatchRays(" in ln:
            sig = ed.root_sigs.get(cur_sig, {})
            for param, base in tables.items():
                for rg in sig.get(param, {}).get("ranges") or []:
                    if rg["range_type"] != "SRV" or rg["space"] != 0:
                        continue
                    for reg in registers:
                        if rg["base"] <= reg < rg["base"] + rg["num"]:
                            slot = base + rg["offset"] + (reg - rg["base"])
                            desc = ed.descriptors.get(slot, {})
                            if desc.get("resource") is not None:
                                found[reg] = desc["resource"]
    return found


def read_lights(wpix):
    ed = w.load_export(w.export_full(wpix))
    ids = srv_buffer_ids(ed, (12, 13, 14))
    ctrl = lights = ex = None
    if 12 in ids:
        ctrl = w.decompress_blob(ed.dir, ids[12], ed)
    if 13 in ids:
        lights = w.decompress_blob(ed.dir, ids[13], ed)
    if 14 in ids:
        ex = w.decompress_blob(ed.dir, ids[14], ed)
    counts = {}
    if ctrl and len(ctrl) >= 16:
        total, envquad, analytic, tri = struct.unpack_from("<4I", ctrl, 0)
        counts = {"TotalLightCount": total, "EnvmapQuadNodeCount": envquad,
                  "AnalyticLightCount": analytic, "TriangleLightCount": tri}
    # Buffer order is [EnvmapQuad ... | Analytic ... | Triangle ...]; decode just the analytic slice.
    decoded = []
    if lights and counts:
        start = counts["EnvmapQuadNodeCount"]
        end = start + counts["AnalyticLightCount"]
        for i in range(start, end):
            b = lights[i * 32:(i + 1) * 32]
            e = ex[i * 16:(i + 1) * 16] if ex and len(ex) >= (i + 1) * 16 else None
            decoded.append(decode_light(b, e))
    return {"ids": ids, "counts": counts, "lights": decoded}


def _v(t, n=4):
    return "(" + ", ".join(f"{x:.{n}g}" for x in t) + ")"


def describe(tag, cap):
    print(f"== {tag} ==")
    print(f"   buffers: t12(ctrl)={cap['ids'].get(12)}  t13(lights)={cap['ids'].get(13)}  t14(ex)={cap['ids'].get(14)}")
    print(f"   counts:  {cap['counts']}")
    print(f"   decoded {len(cap['lights'])} non-empty light record(s):")
    for i, L in enumerate(cap["lights"]):
        line = (f"   [{i}] {L['type_name']:11} center={_v(L['center'])} "
                f"radiance={_v(L['radiance'])} (intensity={L['intensity']:.4g}, hue={_v(L['hue'],3)})")
        print(line)
        extra = []
        if "radius" in L:
            extra.append(f"radius={L['radius']:.5g}")
        if "direction" in L:
            extra.append(f"dir={_v(L['direction'])}")
        if "outer_angle" in L:
            extra.append(f"outer={L['outer_angle']:.4g} inner={L['inner_angle']:.4g}")
        if "angular_size" in L:
            extra.append(f"angularSize={L['angular_size']:.4g} solidAngle={L['solid_angle']:.4g}")
        if extra:
            print("        " + "  ".join(extra))
        if L["has_shaping"]:
            print(f"        SPOT axis={_v(L['spot_axis'])} coneHalfAngle={L['cone_half_angle_deg']:.3g} deg "
                  f"(cosCone={L['cos_cone_angle']:.5g} cosSoftness={L['cos_cone_softness']:.5g}) "
                  f"minFalloff={L['use_min_falloff']}")
    print()


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    A = read_lights(rtxpt)
    B = read_lights(unity)
    describe("RTXPT", A)
    describe("UNITY", B)

    # Order-independent pairing by physical center. Unity mirrors X (known convention, see
    # accel-struct-tlas-instance-parity memory), so match on (type, |x|, y, z).
    def sig(L):
        x, y, z = L["center"]
        return (L["type_name"], round(abs(x), 2), round(y, 2), round(z, 2))
    print("== Pairwise comparison (matched by type + |center|; Unity mirrors X) ==")
    used = [False] * len(B["lights"])
    for i, La in enumerate(A["lights"]):
        j = next((k for k, Lb in enumerate(B["lights"])
                  if not used[k] and sig(Lb) == sig(La)), None)
        if j is None:
            print(f"   RTXPT[{i}] {La['type_name']} center={_v(La['center'])}  -> NO UNITY MATCH")
            continue
        used[j] = True
        Lb = B["lights"][j]
        diffs = []
        for key in ("intensity", "radius", "cos_cone_angle", "cos_cone_softness", "cone_half_angle_deg"):
            a, b = La.get(key), Lb.get(key)
            if a is not None and b is not None and abs(a - b) > 1e-3:
                ratio = f"  (x{a / b:.3g})" if b else ""
                diffs.append(f"{key}: {a:.5g} vs {b:.5g}{ratio}")
        # hue compared directly; center/direction/axis compared after un-mirroring X on Unity
        if max(abs(a - b) for a, b in zip(La["hue"], Lb["hue"])) > 5e-3:
            diffs.append(f"hue: {_v(La['hue'],3)} vs {_v(Lb['hue'],3)}")
        for key in ("center", "direction", "spot_axis"):
            if key in La and key in Lb:
                bx = (-Lb[key][0], Lb[key][1], Lb[key][2])
                if max(abs(a - b) for a, b in zip(La[key], bx)) > 2e-2:
                    diffs.append(f"{key}: {_v(La[key])} vs {_v(Lb[key])} (Unity, pre-mirror)")
        if La["use_min_falloff"] != Lb["use_min_falloff"]:
            diffs.append(f"useMinFalloff: {La['use_min_falloff']} vs {Lb['use_min_falloff']}")
        tag = ("\n        ".join(diffs)) if diffs else "identical (within tolerance)"
        print(f"   RTXPT[{i}] <-> UNITY[{j}]  {La['type_name']:11} center~{_v(La['center'],3)}\n        {tag}")
    for k, Lb in enumerate(B["lights"]):
        if not used[k]:
            print(f"   UNITY[{k}] {Lb['type_name']} center={_v(Lb['center'])}  -> NO RTXPT MATCH")

    # physical-flux note: sphere-light brightness ~ radiance * surface area (r^2)
    print("\n== Effective emitted flux (radiance * 4*pi*r^2, per channel max) ==")
    for tag, cap in (("RTXPT", A), ("UNITY", B)):
        for i, L in enumerate(cap["lights"]):
            if "radius" in L:
                flux = L["intensity"] * 4 * math.pi * L["radius"] ** 2
                print(f"   {tag}[{i}] intensity={L['intensity']:.5g} * area(r={L['radius']:.4g}) "
                      f"-> flux~{flux:.5g}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT)
    p.add_argument("unity", nargs="?", default=DEF_UNITY)
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
