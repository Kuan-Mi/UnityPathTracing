"""
wpix_core — read shader inputs/outputs from PIX .wpix GPU captures, for pixel-level
comparison between two captures.

Approach (no PIX GUI required, CLI only):
  1. `pixtool save-event-list`      -> find a shader event (Dispatch/Draw) by marker name.
  2. `pixtool recapture-region`     -> isolate one event (its global-id) into a tiny capture.
  3. `pixtool export-to-cpp`        -> a C++ replay project whose `resources.bin` holds the
                                       exact GPU bytes of every resource *as of the start of
                                       the recaptured region*.
  4. parse the exported C++         -> resource descs, subresource placed-footprints, the
                                       descriptor heap (slot -> resource/view), the dispatch's
                                       root bindings, and the byte offset of each resource's
                                       (XPRESS-compressed) blob inside resources.bin.
  5. XPRESS-decompress (Cabinet.dll) the wanted blob and decode the DXGI format -> numpy.

Key idea for inputs vs outputs of a dispatch at global-id N:
  * INPUTS  = resource state at the *start* of N  -> recapture region [N, N].
  * OUTPUTS = resource state *after* N ran        -> recapture region [M, M] where M is the
              next existing global-id after N.
The bound resource ids are read from the [N,N] export (which contains the dispatch); the
output *bytes* are read from the [M,M] export. Resource api-object-ids are stable across
recaptures of the same source capture.

Only the Windows Compression API (Cabinet.dll) and numpy are required at runtime.
"""

import os, re, glob, csv, ctypes, struct, hashlib, subprocess, tempfile
from ctypes import wintypes
import numpy as np

# --------------------------------------------------------------------------------------
# pixtool discovery
# --------------------------------------------------------------------------------------

def find_pixtool():
    p = os.environ.get("WPIX_PIXTOOL")
    if p and os.path.isfile(p):
        return p
    cands = glob.glob(r"C:\Program Files\Microsoft PIX\*\pixtool.exe")
    cands += glob.glob(r"C:\Program Files (x86)\Microsoft PIX\*\pixtool.exe")
    if not cands:
        raise FileNotFoundError("pixtool.exe not found; set WPIX_PIXTOOL env var")
    # latest version dir wins
    cands.sort()
    return cands[-1]

def find_dxc():
    """Locate a dxc.exe able to disassemble this project's shaders. Prefers the RTXPT-bundled
    compiler (matches what built them; new enough for recent shader models), then falls back to
    the Windows SDK. WPIX_DXC overrides. dxc.exe needs dxcompiler.dll beside it — the bundled
    bin/x64 has it."""
    p = os.environ.get("WPIX_DXC")
    if p and os.path.isfile(p):
        return p
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bundled = sorted(glob.glob(os.path.join(repo, "RenderingPlugin", "_deps", "*",
                                            "bin", "x64", "dxc.exe")))
    if bundled:
        return bundled[-1]
    sdk = sorted(glob.glob(r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxc.exe"))
    if sdk:
        return sdk[-1]
    raise FileNotFoundError("dxc.exe not found; set WPIX_DXC env var")

def _cache_root():
    root = os.environ.get("WPIX_CACHE") or os.path.join(tempfile.gettempdir(), "wpix_mcp_cache")
    os.makedirs(root, exist_ok=True)
    return root

def _run(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pixtool failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout

# --------------------------------------------------------------------------------------
# event list
# --------------------------------------------------------------------------------------

def list_events(wpix):
    """Return list of dicts {queue, parent, name, global_id(int|None)} in capture order."""
    pixtool = find_pixtool()
    key = hashlib.sha1((os.path.abspath(wpix) + str(os.path.getmtime(wpix))).encode()).hexdigest()[:16]
    csv_path = os.path.join(_cache_root(), f"events_{key}.csv")
    if not os.path.isfile(csv_path):
        _run([pixtool, "open-capture", wpix, "save-event-list", csv_path])
    out = []
    with open(csv_path, encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        for row in rd:
            if len(row) < 4:
                continue
            gid = row[3].strip()
            out.append({
                "queue": int(row[0]) if row[0].strip() else None,
                "parent": int(row[1]) if row[1].strip() else None,
                "name": row[2].strip(),
                "global_id": int(gid) if gid else None,
            })
    return out

def find_shader_events(wpix, name_filter=None, dispatches_only=False):
    """Events that have a global id (Dispatch/Draw/etc), optionally filtered by parent
    marker name. Returns each with its nearest enclosing named marker for context.
    Set dispatches_only=True to drop ResourceBarrier/Draw and keep only Dispatch events
    (the common case when comparing compute passes)."""
    ev = list_events(wpix)
    by_queue = {}
    for e in ev:
        by_queue.setdefault(e["queue"], {})[e["queue"]] = e  # not used; kept simple below
    # build parent-name lookup by walking; markers carry name but no global id.
    idx = {}
    for e in ev:
        idx[(e["queue"], e["parent"])] = e
    # Map queue-id (row 0) -> event for parent resolution
    qmap = {e["queue"]: e for e in ev}
    res = []
    for e in ev:
        if e["global_id"] is None:
            continue
        # climb parents to collect marker names
        chain = []
        p = e["parent"]
        guard = 0
        while p is not None and p in qmap and guard < 64:
            par = qmap[p]
            if par["name"]:
                chain.append(par["name"])
            if par["parent"] == p:
                break
            p = par["parent"]
            guard += 1
        marker_path = "/".join(reversed(chain))
        if name_filter and name_filter.lower() not in (marker_path + "/" + e["name"]).lower():
            continue
        if dispatches_only and e["name"] != "Dispatch":
            continue
        res.append({"global_id": e["global_id"], "name": e["name"], "marker": marker_path})
    return res

def next_global_id(wpix, gid):
    ids = sorted({e["global_id"] for e in list_events(wpix) if e["global_id"] is not None})
    for i in ids:
        if i > gid:
            return i
    return None

# --------------------------------------------------------------------------------------
# recapture + export (cached)
# --------------------------------------------------------------------------------------

def export_region(wpix, start, end):
    """Recapture [start,end] and export to C++. Returns the export directory (cached)."""
    pixtool = find_pixtool()
    key = hashlib.sha1((os.path.abspath(wpix) + str(os.path.getmtime(wpix)) +
                        f"|{start}|{end}").encode()).hexdigest()[:16]
    export_dir = os.path.join(_cache_root(), f"export_{key}")
    done = os.path.join(export_dir, ".done")
    if os.path.isfile(done):
        return export_dir
    region = os.path.join(_cache_root(), f"region_{key}.wpix")
    _run([pixtool, "open-capture", wpix, "recapture-region", region,
          f"--start={start}", f"--end={end}"])
    os.makedirs(export_dir, exist_ok=True)
    _run([pixtool, "open-capture", region, "export-to-cpp", "--force", export_dir])
    try:
        os.remove(region)  # the 200MB+ recapture is no longer needed once exported
    except OSError:
        pass
    open(done, "w").close()
    return export_dir

def export_full(wpix):
    """Export the ORIGINAL capture to C++ directly (no recapture-region). Cached per capture.

    Use this for static frame metadata — resource debug names, formats, descriptor bindings.
    recapture-region (export_region) truncates resource debug names (e.g. 'simplebluesky'
    -> 'simple') and re-encodes resource formats (e.g. BC6H -> R32G32B32A32_FLOAT), so it is
    unreliable for names/formats regardless of how wide the region is. A direct export of the
    capture preserves both. It does NOT give per-event resource state, so pixel data at a
    specific event still requires export_region."""
    pixtool = find_pixtool()
    key = hashlib.sha1((os.path.abspath(wpix) + str(os.path.getmtime(wpix)) +
                        "|full").encode()).hexdigest()[:16]
    export_dir = os.path.join(_cache_root(), f"full_{key}")
    done = os.path.join(export_dir, ".done")
    if os.path.isfile(done):
        return export_dir
    os.makedirs(export_dir, exist_ok=True)
    _run([pixtool, "open-capture", wpix, "export-to-cpp", "--force", export_dir])
    open(done, "w").close()
    return export_dir

_EXPORT_DATA_CACHE = {}

def load_export(export_dir):
    """ExportData for a directory, parsed once and reused within the process.

    Parsing an export dir (regex over every *.cpp + the recursive blob-offset
    simulation in _simulate_blob_offsets) is expensive, so all callers — extract,
    compare and describe_dispatch — must go through here rather than constructing
    ExportData directly, or each call re-parses the whole exported C++ project."""
    ed = _EXPORT_DATA_CACHE.get(export_dir)
    if ed is None:
        ed = _EXPORT_DATA_CACHE[export_dir] = ExportData(export_dir)
    return ed

def canonical_name(wpix, res):
    """Map a region-recapture resource (truncated name + desc) to its full debug name from
    the original-capture export. Returns the original (truncated) name if no unique match."""
    tn = res.get("name") or ""
    try:
        full = load_export(export_full(wpix))
    except Exception:
        return res.get("name")
    names = {r.get("name") for r in full.resources.values()
             if r.get("width") == res.get("width")
             and r.get("height") == res.get("height")
             and r.get("array_or_depth") == res.get("array_or_depth")
             and r.get("mips") == res.get("mips")
             and (r.get("name") or "").startswith(tn)}
    return names.pop() if len(names) == 1 else res.get("name")

# --------------------------------------------------------------------------------------
# export parsing
# --------------------------------------------------------------------------------------

_READ_RE = re.compile(r'g_resourceReader->Read\(\s*\w+\s*,\s*(\d+)\s*\)')
_CALL_RE = re.compile(r'^\s*(\w+)\s*\(\s*\)\s*;')
_RESFUNC_RE = re.compile(r'CreateAndInitResource_(\d+)')

class ExportData:
    def __init__(self, export_dir):
        self.dir = export_dir
        self.resources = {}     # id -> dict
        self.blob_offset = {}   # id -> int (byte offset in resources.bin)
        self.read_size = {}     # id -> compressed size
        self.descriptors = {}   # heap slot -> dict
        self.dispatches = []    # list of dict (one per Dispatch found)
        self._parse()

    def _read_all(self, pattern):
        text = {}
        for f in glob.glob(os.path.join(self.dir, pattern)):
            text[f] = open(f, encoding="utf-8", errors="replace").read()
        return text

    def _parse(self):
        self._parse_resources()
        self._parse_names()
        self._parse_footprints()
        self._parse_psos()
        self._simulate_blob_offsets()
        self._parse_descriptors()
        self._parse_root_signatures()
        self._parse_dispatches()

    # ---- compute PSOs: id -> CS uncompressed byte size (from CreatePSOs.cpp) ----
    def _parse_psos(self):
        self.pso_cs_size = {}          # pso_id -> bytes of the CS blob
        self.pso_blob = {}             # pso_id -> (byte_offset_in_resources_bin, compressed_size)
        cs_re = re.compile(r'void CreateComputePipelineState_(\d+)\(\)(.*?)\n\}', re.S)
        size_re = re.compile(r'cpsoDesc\.CS\s*=\s*\{[^,}]+,\s*(\d+)\s*\}')
        for f, t in self._read_all("CreatePSOs*.cpp").items():
            for m in cs_re.finditer(t):
                sm = size_re.search(m.group(2))
                if sm:
                    self.pso_cs_size[int(m.group(1))] = int(sm.group(1))

    # ---- root signatures: id -> {param_index: {type, table_size}} ----
    def _parse_root_signatures(self):
        self.root_sigs = {}
        blk_re = re.compile(
            r'static D3D12_ROOT_PARAMETER1 rootParameters\[\d+\];(.*?)'
            r'CreateAndTrackRootSignature\((\d+),', re.S)
        ptype_re = re.compile(r'rootParameters\[(\d+)\]\.ParameterType\s*=\s*D3D12_ROOT_PARAMETER_TYPE_(\w+)')
        # D3D12_DESCRIPTOR_RANGE1 = { TYPE, NumDescriptors, BaseShaderRegister, RegisterSpace,
        #                             Flags, OffsetInDescriptorsFromTableStart }
        range_re = re.compile(r'descriptorRanges\[\d+\]\s*=\s*\{\s*D3D12_DESCRIPTOR_RANGE_TYPE_(\w+),'
                              r'\s*(\d+),\s*(\d+),\s*(\d+),\s*[^,]+,\s*(\d+)\s*\}')
        # root CBV/SRV/UAV descriptor: rootParameters[i].Descriptor = { ShaderRegister, RegisterSpace }
        rootdesc_re = re.compile(r'rootParameters\[(\d+)\]\.Descriptor\s*=\s*\{\s*(\d+),\s*(\d+)')
        for f, t in self._read_all("FrameResources_*.cpp").items():
            for blk in blk_re.finditer(t):
                body, sigid = blk.group(1), int(blk.group(2))
                params = {}
                cur = None
                for ln in body.split("\n"):
                    m = ptype_re.search(ln)
                    if m:
                        cur = int(m.group(1))
                        params[cur] = {"type": m.group(2), "table_size": 0,
                                       "range_type": None, "ranges": []}
                        continue
                    m = range_re.search(ln)
                    if m and cur is not None:
                        rtype, num, base, space, off = (m.group(1), int(m.group(2)),
                                                        int(m.group(3)), int(m.group(4)),
                                                        int(m.group(5)))
                        params[cur]["table_size"] += num
                        params[cur]["ranges"].append({"range_type": rtype, "num": num,
                                                       "base": base, "space": space, "offset": off})
                        if params[cur]["range_type"] is None:
                            params[cur]["range_type"] = rtype
                        continue
                    m = rootdesc_re.search(ln)
                    if m and int(m.group(1)) in params:
                        params[int(m.group(1))]["reg"] = (int(m.group(2)), int(m.group(3)))
                self.root_sigs[sigid] = params

    # ---- resource debug names: GetObject(id)->SetName(LR"(name)") ----
    def _parse_names(self):
        name_re = re.compile(r'GetObject\((\d+)\)->SetName\(LR"\(([^)]*)\)"\)')
        for f, t in self._read_all("FrameResources_*.cpp").items():
            for m in name_re.finditer(t):
                rid = int(m.group(1))
                if rid in self.resources:
                    self.resources[rid]["name"] = m.group(2)

    def find_resource(self, name=None, like=None):
        """Find a resource id by debug name and/or a desc dict `like`
        (keys among width,height,array_or_depth,mips,format,dimension). Returns id or None."""
        for rid, r in self.resources.items():
            if name is not None and r.get("name") != name:
                continue
            if like is not None and any(r.get(k) != v for k, v in like.items()):
                continue
            return rid
        return None

    def find_resource_loose(self, name, like=None):
        """Like find_resource but tolerant of the 8-char name truncation that
        export_region applies: matches when either the stored name is a prefix of
        `name` or vice versa. Prefers an exact match, then a desc-disambiguated
        prefix match. Returns id or None."""
        exact = self.find_resource(name=name, like=like)
        if exact is not None:
            return exact
        cands = []
        for rid, r in self.resources.items():
            rn = r.get("name") or ""
            if not rn or not (name.startswith(rn) or rn.startswith(name)):
                continue
            if like is not None and any(r.get(k) != v for k, v in like.items()):
                continue
            cands.append(rid)
        return cands[0] if len(cands) == 1 else (cands[0] if cands and like else None)

    # ---- resource descs + read sizes ----
    def _parse_resources(self):
        desc_re = re.compile(
            r'void CreateAndInitResource_(\d+)\(\).*?'
            r'D3D12_RESOURCE_DESC\s+resourceDesc\s*=\s*\{\s*'
            r'(D3D12_RESOURCE_DIMENSION_\w+),\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*'
            r'(DXGI_FORMAT_\w+)',
            re.S)
        for f, t in self._read_all("CreateAndInitResources_*.cpp").items():
            for m in desc_re.finditer(t):
                rid = int(m.group(1))
                self.resources[rid] = {
                    "id": rid,
                    "dimension": m.group(2),
                    "width": int(m.group(3)),
                    "height": int(m.group(4)),
                    "array_or_depth": int(m.group(5)),
                    "mips": int(m.group(6)),
                    "format": m.group(7),
                    "footprints": [],
                }
            for m in re.finditer(r'void CreateAndInitResource_(\d+)\(\)(.*?)\n\}', t, re.S):
                rid = int(m.group(1))
                rm = _READ_RE.search(m.group(2))
                if rm:
                    self.read_size[rid] = int(rm.group(1))

    # ---- subresource placed footprints ----
    def _parse_footprints(self):
        fp_re = re.compile(
            r'g_resourceInitInfo_(\d+)_0\[\]\s*=\s*\{(.*?)\};', re.S)
        entry_re = re.compile(
            r'\{\s*(\d+),\s*\{\s*(DXGI_FORMAT_\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\},\s*(\d+)\s*\}')
        for f, t in self._read_all("CapturedAssets.h").items():
            for m in fp_re.finditer(t):
                rid = int(m.group(1))
                fps = []
                for e in entry_re.finditer(m.group(2)):
                    fps.append({
                        "offset": int(e.group(1)),
                        "format": e.group(2),
                        "width": int(e.group(3)),
                        "height": int(e.group(4)),
                        "depth": int(e.group(5)),
                        "row_pitch": int(e.group(6)),
                        "subresource": int(e.group(7)),
                    })
                if rid in self.resources:
                    self.resources[rid]["footprints"] = fps

    # ---- byte offset of each resource blob in resources.bin (runtime read order) ----
    def _simulate_blob_offsets(self):
        funcs = {}
        for f, t in {**self._read_all("*.cpp")}.items():
            lines = t.split("\n"); i, n = 0, len(lines)
            while i < n:
                mm = re.match(r'^(?:inline\s+)?void\s+(\w+)\s*\(', lines[i])
                if mm:
                    name = mm.group(1); j = i
                    while j < n and "{" not in lines[j]:
                        j += 1
                    depth = 1; k = j + 1; body = []
                    while k < n:
                        depth += lines[k].count("{") - lines[k].count("}")
                        if depth <= 0:
                            break
                        body.append(lines[k]); k += 1
                    funcs[name] = body; i = k + 1
                else:
                    i += 1
        self._funcs = funcs
        has_read = {}
        def chr_(name, stack):
            if name in has_read: return has_read[name]
            if name in stack or name not in funcs:
                return False
            stack.add(name); res = False
            for ln in funcs[name]:
                if _READ_RE.search(ln): res = True; break
                cm = _CALL_RE.match(ln)
                if cm and cm.group(1) in funcs and cm.group(1) != name and chr_(cm.group(1), stack):
                    res = True; break
            stack.discard(name); has_read[name] = res; return res
        for nm in list(funcs):
            chr_(nm, set())

        offset = [0]
        pso_re = re.compile(r'CreateComputePipelineState_(\d+)')
        def walk(name):
            cur_rid = None
            mrid = _RESFUNC_RE.fullmatch(name)
            if mrid: cur_rid = int(mrid.group(1))
            pso_m = pso_re.fullmatch(name)
            cur_pso = int(pso_m.group(1)) if pso_m else None
            for ln in funcs.get(name, []):
                rm = _READ_RE.search(ln)
                if rm:
                    if cur_rid is not None and cur_rid not in self.blob_offset:
                        self.blob_offset[cur_rid] = offset[0]
                    # the CS is the first blob read inside a compute-PSO function
                    if cur_pso is not None and cur_pso not in self.pso_blob:
                        self.pso_blob[cur_pso] = (offset[0], int(rm.group(1)))
                    offset[0] += int(rm.group(1))
                    continue
                cm = _CALL_RE.match(ln)
                if cm:
                    cn = cm.group(1)
                    if cn in funcs and has_read.get(cn) and cn != name:
                        walk(cn)
        if "CreateAppResources_000" in funcs:
            walk("CreateAppResources_000")

    # ---- descriptor heap: slot -> resource/view ----
    def _parse_descriptors(self):
        # CreateXxxView_*(GetResource(R).Get(), [nullptr,] GetCpuDescriptor(heap, SLOT), FORMAT, DIM, ... , mip..)
        view_re = re.compile(
            r'Create(\w+?)View_(\w+)\(\s*GetResource\((\d+)\)\.Get\(\)\s*,\s*'
            r'(?:nullptr\s*,\s*)?GetCpuDescriptor\([^,]+,\s*(\d+)\)\s*,\s*'
            r'(DXGI_FORMAT_\w+)\s*,\s*(D3D12_\w+_DIMENSION_\w+)([^;]*)\)')
        for f, t in self._read_all("Descriptors_*.cpp").items():
            for m in view_re.finditer(t):
                view_kind, view_shape, rid, slot, fmt, dim, tail = m.groups()
                nums = re.findall(r'-?\d+\.?\d*f?', tail)
                self.descriptors[int(slot)] = {
                    "slot": int(slot),
                    "view": view_kind,            # ShaderResource / UnorderedAccess / RenderTarget
                    "shape": view_shape,          # Tex2D / Tex2DArray / TexCube / Buffer ...
                    "resource": int(rid),
                    "format": fmt,
                    "dimension": dim,
                    "view_args": tail.strip(),
                }

    # ---- dispatches and their root bindings ----
    def _parse_dispatches(self):
        for f, t in self._read_all("CommandLists_*.cpp").items():
            for fn in re.finditer(r'void (PopulateCommandList_\w+)\(\)\s*\{(.*?)\n\}', t, re.S):
                body = fn.group(2)
                root_tables = {}; root_cbv = {}; cur_sig = None; cur_pso = None
                last_gid = None; rays_dims = None
                for ln in body.split("\n"):
                    m = re.search(r'SetPipelineState\(GetPipelineState\((\d+)\)', ln)
                    if m: cur_pso = int(m.group(1))
                    m = re.search(r'SetPipelineState1\(GetStateObject\((\d+)\)', ln)
                    if m: cur_pso = None  # DXR state object: no single compute kernel to reflect
                    m = re.search(r'SetComputeRootSignature\(GetRootSignature\((\d+)\)', ln)
                    if m: cur_sig = int(m.group(1))
                    m = re.search(r'SetComputeRootDescriptorTable\((\d+),\s*GetGpuDescriptor\([^,]+,\s*(\d+)\)', ln)
                    if m: root_tables[int(m.group(1))] = int(m.group(2))
                    m = re.search(r'SetComputeRootConstantBufferView\((\d+),\s*GetGpuva\((\d+),\s*(\d+)\)', ln)
                    if m: root_cbv[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
                    m = re.search(r'//\s*GlobalId\s*=\s*(\d+)', ln)
                    if m: last_gid = int(m.group(1))
                    m = re.search(r'->Dispatch\((\d+),\s*(\d+),\s*(\d+)\)', ln)
                    if m:
                        self.dispatches.append({
                            "func": fn.group(1),
                            "global_id": last_gid,
                            "groups": (int(m.group(1)), int(m.group(2)), int(m.group(3))),
                            "root_signature": cur_sig,
                            "pso": cur_pso,
                            "root_tables": dict(root_tables),   # param -> start slot
                            "root_cbv": dict(root_cbv),         # param -> (res, offset)
                        })
                    # DXR: a DispatchRays carries the same compute root bindings (CBV/tables)
                    # as a Dispatch, so register it too -> describe_dispatch/extract/compare
                    # work on ray-dispatch global ids. The D3D12_DISPATCH_RAYS_DESC trailing
                    # ints are the ray-grid Width/Height/Depth (used as 'groups').
                    m = re.search(r'D3D12_DISPATCH_RAYS_DESC\b.*?(\d+),\s*(\d+),\s*(\d+)\s*\}\s*;', ln)
                    if m: rays_dims = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if '->DispatchRays(' in ln:
                        self.dispatches.append({
                            "func": fn.group(1),
                            "global_id": last_gid,
                            "groups": rays_dims or (0, 0, 0),
                            "root_signature": cur_sig,
                            "pso": cur_pso,
                            "root_tables": dict(root_tables),   # param -> start slot
                            "root_cbv": dict(root_cbv),         # param -> (res, offset)
                            "ray_dispatch": True,
                        })
                        rays_dims = None

    # ---- enumerate populated descriptor slots of a table (start slot..next table/gap) ----
    def table_slots(self, start_slot, stop_slots, max_count=64):
        out = []
        s = start_slot
        while s in self.descriptors and len(out) < max_count:
            if s != start_slot and s in stop_slots:
                break
            out.append(self.descriptors[s])
            s += 1
        return out

    # ---- shader reflection: resources a compute kernel actually declares (PSV0) ----
    def shader_resource_bindings(self, pso):
        """[{class, space, lo, hi}] the dispatch's compute shader declares, or None when
        the shader blob / PSV0 can't be located (so callers can degrade gracefully)."""
        if pso is None or pso not in getattr(self, "pso_blob", {}):
            return None
        off, csize = self.pso_blob[pso]
        try:
            data = _xpress_decompress_at(self.dir, off, csize)
        except Exception:
            return None
        cs = data[:self.pso_cs_size.get(pso, len(data))]
        return psv0_resource_bindings(cs)

# --------------------------------------------------------------------------------------
# XPRESS decompression (Windows Compression API, Cabinet.dll) — matches the exporter
# --------------------------------------------------------------------------------------

_COMPRESS_ALGORITHM_XPRESS = 3

# Decompressed resource blobs keyed by (export_dir, resource_id). A single blob can be
# 100MB+ (e.g. a 2048^2 cube, all faces+mips), and one blob is reused across every
# mip/face of the same resource, so without this each face re-inflated the whole thing.
# Bounded to keep memory in check; clear_cache() also drops it.
_BLOB_CACHE = {}
_BLOB_CACHE_MAX = 6

def _xpress_decompress_at(export_dir, off, csize):
    """XPRESS-decompress `csize` bytes at byte offset `off` in resources.bin."""
    cab = ctypes.WinDLL("Cabinet.dll")
    hdec = wintypes.HANDLE()
    if not cab.CreateDecompressor(_COMPRESS_ALGORITHM_XPRESS, None, ctypes.byref(hdec)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with open(os.path.join(export_dir, "resources.bin"), "rb") as fh:
            fh.seek(off); comp = fh.read(csize)
        final = ctypes.c_size_t(0)
        cab.Decompress(hdec, comp, ctypes.c_size_t(csize), None, 0, ctypes.byref(final))
        out = (ctypes.c_ubyte * final.value)()
        got = ctypes.c_size_t(0)
        if not cab.Decompress(hdec, comp, ctypes.c_size_t(csize), out,
                              ctypes.c_size_t(final.value), ctypes.byref(got)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bytes(out[:got.value])
    finally:
        cab.CloseDecompressor(hdec)

def decompress_blob(export_dir, resource_id, exp=None):
    ckey = (export_dir, resource_id)
    cached = _BLOB_CACHE.get(ckey)
    if cached is not None:
        return cached
    exp = exp or load_export(export_dir)
    if resource_id not in exp.blob_offset:
        raise KeyError(f"resource {resource_id} has no initial data in this region")
    blob = _xpress_decompress_at(export_dir, exp.blob_offset[resource_id],
                                 exp.read_size[resource_id])
    if len(_BLOB_CACHE) >= _BLOB_CACHE_MAX:
        _BLOB_CACHE.pop(next(iter(_BLOB_CACHE)))  # FIFO eviction
    _BLOB_CACHE[ckey] = blob
    return blob

# --------------------------------------------------------------------------------------
# Shader reflection: pull the resource bindings a compute kernel actually declares from
# its DXBC container's PSV0 (pipeline-state-validation) chunk. Used to mark which of a
# dispatch's bound descriptors the shader really touches (vs. bound-but-unused, common
# when many kernels share one root signature).
# --------------------------------------------------------------------------------------

# PSVResourceType -> HLSL register class. 1=Sampler 2=CBV 3-5=SRV(typed/raw/struct) 6-9=UAV
_PSV_RTYPE_CLASS = {1: "s", 2: "b", 3: "t", 4: "t", 5: "t", 6: "u", 7: "u", 8: "u", 9: "u"}
# D3D12_DESCRIPTOR_RANGE_TYPE -> HLSL register class
_RANGE_TYPE_CLASS = {"SRV": "t", "UAV": "u", "CBV": "b", "SAMPLER": "s"}

def dxbc_chunk(blob, fourcc):
    """Return the bytes of a named chunk in a DXBC container, or None."""
    if len(blob) < 32 or blob[:4] != b"DXBC":
        return None
    n = struct.unpack_from("<I", blob, 28)[0]
    for o in struct.unpack_from("<%dI" % n, blob, 32):
        if blob[o:o + 4] == fourcc:
            sz = struct.unpack_from("<I", blob, o + 4)[0]
            return blob[o + 8:o + 8 + sz]
    return None

def psv0_resource_bindings(dxil):
    """Parse a shader's PSV0 chunk -> list of {class, space, lo, hi} declared resources.
    Returns None if there's no DXBC/PSV0 (e.g. a stripped or non-DXIL blob)."""
    psv = dxbc_chunk(dxil, b"PSV0")
    if psv is None:
        return None
    p = 0
    ri_size = struct.unpack_from("<I", psv, p)[0]; p += 4 + ri_size  # skip PSVRuntimeInfo
    rcount = struct.unpack_from("<I", psv, p)[0]; p += 4
    out = []
    if rcount:
        bind_size = struct.unpack_from("<I", psv, p)[0]; p += 4
        for i in range(rcount):
            rt, space, lo, hi = struct.unpack_from("<IIII", psv, p + i * bind_size)[:4]
            out.append({"class": _PSV_RTYPE_CLASS.get(rt, "?"),
                        "space": space, "lo": lo, "hi": hi})
    return out

# --------------------------------------------------------------------------------------
# Shader compile-time info: read identity/profile/features straight from the DXIL
# container, and recover the original DXC command line (entry/defines/flags) from the
# embedded debug metadata via a dxc.exe -dumpbin disassembly.
# --------------------------------------------------------------------------------------

# DXIL program-version shader-kind nibble -> profile prefix.
_SHADER_KIND = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs",
                6: "lib", 7: "ms", 8: "as"}

# DXIL ShaderFeatureInfo (SFI0) bit index -> feature name.
_SFI0_FEATURES = {
    0: "Doubles", 1: "ComputeShadersPlusRawAndStructuredBuffersViaShader4X",
    2: "UAVsAtEveryStage", 3: "64UAVs", 4: "MinimumPrecision", 5: "DoubleExtensions",
    6: "ShaderExtensions", 7: "ComparisonFiltering", 8: "TiledResources",
    9: "PSOutStencilRef", 10: "InnerCoverage", 11: "TypedUAVLoadAdditionalFormats",
    12: "ROVs", 13: "ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer",
    14: "WaveOps", 15: "Int64Ops", 16: "ViewID", 17: "Barycentrics",
    18: "NativeLowPrecision", 19: "ShadingRate", 20: "Raytracing_Tier_1_1",
    21: "SamplerFeedback", 22: "AtomicInt64OnTypedResource",
    23: "AtomicInt64OnGroupShared", 24: "DerivativesInMeshAndAmpShaders",
    25: "ResourceDescriptorHeapIndexing", 26: "SamplerDescriptorHeapIndexing",
    27: "AtomicInt64OnHeapResource", 28: "AdvancedTextureOps",
    29: "WriteableMSAATextures",
}

def container_parts(blob):
    """{fourcc_bytes: (data_offset, size)} for a DXBC/DXIL container, or {} if not one."""
    if len(blob) < 32 or blob[:4] != b"DXBC":
        return {}
    n = struct.unpack_from("<I", blob, 28)[0]
    parts = {}
    for o in struct.unpack_from("<%dI" % n, blob, 32):
        if o + 8 > len(blob):
            continue
        parts[blob[o:o + 4]] = (o + 8, struct.unpack_from("<I", blob, o + 4)[0])
    return parts

def container_compile_info(blob):
    """Compile-time fields readable from the container itself (no disassembler):
    shader_hash (HASH part digest — the value PIX/RenderDoc show, NOT the header checksum),
    target/profile + shader_kind (DXIL program version), debug_name (ILDN), and the SFI0
    required-feature flags."""
    parts = container_parts(blob)
    info = {"container_parts": sorted(c.decode("ascii", "replace") for c in parts)}
    if b"HASH" in parts:                      # DxilShaderHash { uint32 Flags; byte Digest[16] }
        o, _ = parts[b"HASH"]
        info["shader_hash"] = blob[o + 4:o + 20].hex()
    for part in (b"DXIL", b"ILDB"):           # program-version header gives the profile
        if part in parts:
            ver = struct.unpack_from("<I", blob, parts[part][0])[0]
            kind, major, minor = (ver >> 16) & 0xFFFF, (ver >> 4) & 0xF, ver & 0xF
            info["shader_kind"] = _SHADER_KIND.get(kind, str(kind))
            info["target"] = "%s_%d_%d" % (_SHADER_KIND.get(kind, "?"), major, minor)
            break
    if b"ILDN" in parts:                      # DxilShaderDebugName { u16 Flags; u16 NameLen; char[] }
        o, _ = parts[b"ILDN"]
        nl = struct.unpack_from("<H", blob, o + 2)[0]
        info["debug_name"] = blob[o + 4:o + 4 + nl].split(b"\x00")[0].decode("utf-8", "replace")
    if b"SFI0" in parts and parts[b"SFI0"][1] >= 8:
        mask = struct.unpack_from("<Q", blob, parts[b"SFI0"][0])[0]
        info["feature_flags_mask"] = hex(mask)
        info["feature_flags"] = [v for b, v in _SFI0_FEATURES.items() if mask & (1 << b)]
    return info

def _llvm_unescape(s):
    """Decode LLVM-IR metadata-string escapes (\\\\ and \\XX hex), e.g. DXC writes '\\5C' for '\\'."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n:
            if s[i + 1] == "\\":
                out.append("\\"); i += 2; continue
            h = s[i + 1:i + 3]
            if len(h) == 2 and all(c in "0123456789abcdefABCDEF" for c in h):
                out.append(chr(int(h, 16))); i += 3; continue
            out.append(s[i + 1]); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)

def _structure_args(toks):
    """Fold a DXC argument token list into {raw_args, entry, target, output, defines,
    includes, flags}. Shared by the embedded-debug and external-PDB code paths."""
    out = {"raw_args": list(toks), "defines": [], "includes": [], "flags": []}
    i = 0
    valued = {"-E": "entry", "-T": "target", "-Fo": "output"}
    listed = {"-D": "defines", "-I": "includes"}
    while i < len(toks):
        t = toks[i]
        if t in valued and i + 1 < len(toks):
            out[valued[t]] = toks[i + 1]; i += 2; continue
        if t in listed and i + 1 < len(toks):
            if toks[i + 1] not in out[listed[t]]:   # build scripts often pass -D/-I twice
                out[listed[t]].append(toks[i + 1])
            i += 2; continue
        if t.startswith("-"):
            out["flags"].append(t)
        i += 1
    return out

def _parse_compile_args(disasm):
    """Pull the embedded DXC command line out of a -dumpbin disassembly. The args live in a
    one-line metadata node like `!{!"-E", !"main", !"-T", !"cs_6_9", !"-D", !"FOO", ...}` that
    DXC emits only with -Zi/-Qembed_debug. Returns the structured dict or {} when absent."""
    arg_line = next((ln for ln in disasm.splitlines() if '!"-T"' in ln and "!{" in ln), None) \
        or next((ln for ln in disasm.splitlines() if '!"-E"' in ln and "!{" in ln), None)
    if not arg_line:
        return {}
    toks = [_llvm_unescape(t) for t in re.findall(r'!"((?:[^"\\]|\\.)*)"', arg_line)]
    return _structure_args(toks) if toks else {}

def _parse_required_features(disasm):
    """The `; Note: shader requires additional functionality:` block -> list of feature lines."""
    lines = disasm.splitlines()
    for i, ln in enumerate(lines):
        if "requires additional functionality" in ln:
            feats = []
            for ln2 in lines[i + 1:]:
                s = ln2.strip().lstrip(";").strip()
                if not s:
                    break
                feats.append(s)
            return feats
    return []

def _default_pdb_dir():
    """Unity writes side-car shader PDBs (hash-named, MSF/DIA) to UnityProject\\ShaderPDB.
    WPIX_PDB_DIR overrides; returns None if neither exists."""
    p = os.environ.get("WPIX_PDB_DIR")
    if p and os.path.isdir(p):
        return p
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = os.path.join(repo, "UnityProject", "ShaderPDB")
    return cand if os.path.isdir(cand) else None

def _compile_args_from_pdb(shader_hash, pdb_dir, dxc):
    """Recover compile args from an external DXC PDB named <shader_hash>.pdb (Unity's
    layout). Returns (structured_args, pdb_path, main_file) or (None, path_or_None, None)."""
    if not shader_hash or not pdb_dir:
        return None, None, None
    pdb_path = os.path.join(pdb_dir, f"{shader_hash}.pdb")
    if not os.path.isfile(pdb_path):
        return None, None, None
    import dxc_pdb  # lazy: only loads dxcompiler.dll when an external PDB is actually needed
    dll = os.path.join(os.path.dirname(dxc), "dxcompiler.dll")
    raw = dxc_pdb.read_pdb_compile_info(pdb_path, dll)
    structured = _structure_args(raw.get("args") or [])
    return structured, pdb_path, raw.get("main_file")

def describe_shader(wpix, global_id, disassemble=True, pdb_dir=None):
    """Compile-time info for the compute shader bound at a Dispatch: shader_hash, target,
    shader_kind, required-feature flags and debug name (always, from the DXIL container), plus
    the original DXC command line — entry, defines, includes, flags.

    The command line is recovered from whichever source carries it:
      * embedded debug info (-Zi/-Qembed_debug) in the captured blob, via dxc -dumpbin; else
      * an external side-car PDB named <shader_hash>.pdb (Unity's UnityProject\\ShaderPDB
        layout), read through IDxcPdbUtils2 in dxcompiler.dll.
    'args_source' in the result says which was used. Set disassemble=False for container-only
    fields. pdb_dir overrides the PDB folder (default: auto-detected ShaderPDB, or WPIX_PDB_DIR);
    WPIX_DXC overrides the dxc.exe / dxcompiler.dll used."""
    # The shader container is static frame metadata (identical wherever the PSO is bound),
    # so a single whole-capture export serves every dispatch and is cached once per capture.
    # That avoids a per-global_id recapture-region + export (the slow path): only fall back
    # to it if the full export somehow lacks this PSO's blob.
    def _resolve(export_dir):
        A = load_export(export_dir)
        d = next((x for x in A.dispatches if x["global_id"] == global_id),
                 A.dispatches[-1] if A.dispatches else None)
        if d is None:
            return None
        pso = d.get("pso")
        if pso is None or pso not in getattr(A, "pso_blob", {}):
            return None
        return A, pso

    resolved = _resolve(export_full(wpix)) or _resolve(export_region(wpix, global_id, global_id))
    if resolved is None:
        raise RuntimeError(f"dispatch {global_id} has no compute-shader blob to inspect")
    A, pso = resolved
    off, csize = A.pso_blob[pso]
    blob = _xpress_decompress_at(A.dir, off, csize)
    blob = blob[:A.pso_cs_size.get(pso, len(blob))]
    info = {"global_id": global_id, "pso": pso, "container_bytes": len(blob)}
    info.update(container_compile_info(blob))
    if not disassemble:
        return info
    try:
        dxc = find_dxc()
    except FileNotFoundError as e:
        info["disasm_note"] = f"{e}; entry/defines/flags unavailable"
        return info
    info["dxc"] = dxc

    # 1) embedded compile args (-Zi/-Qembed_debug), via disassembly — also gives features.
    tmp = os.path.join(_cache_root(), f"cs_{info.get('shader_hash', 'blob')}.bin")
    with open(tmp, "wb") as fh:
        fh.write(blob)
    proc = subprocess.run([dxc, "-dumpbin", tmp], capture_output=True, text=True)
    if proc.returncode == 0:
        rf = _parse_required_features(proc.stdout)
        if rf:
            info["required_features"] = rf
        args = _parse_compile_args(proc.stdout)
        if args:
            info.update(args)
            info["args_source"] = "embedded_debug"
            return info
    else:
        info["disasm_note"] = f"dxc -dumpbin failed: {(proc.stdout + proc.stderr).strip()[:300]}"

    # 2) fall back to an external side-car PDB (Unity's no-embed-debug layout).
    pdb_dir = pdb_dir or _default_pdb_dir()
    try:
        args, pdb_path, main_file = _compile_args_from_pdb(info.get("shader_hash"), pdb_dir, dxc)
    except Exception as e:
        info["pdb_note"] = f"external PDB read failed: {type(e).__name__}: {e}"
        return info
    if args:
        info.update(args)
        info["args_source"] = "external_pdb"
        info["pdb_path"] = pdb_path
        if main_file:
            info["main_file"] = main_file
    elif "disasm_note" not in info:
        where = f" (looked in {pdb_dir})" if pdb_dir else " (no ShaderPDB dir found)"
        info["disasm_note"] = ("no embedded compile args (built without -Zi/-Qembed_debug) "
                               f"and no <hash>.pdb side-car{where}; entry/defines/flags unavailable")
    return info

# --------------------------------------------------------------------------------------
# DXGI format decoding -> numpy (H, W, C) float for color formats
# --------------------------------------------------------------------------------------

# (numpy dtype, channels, bytes-per-pixel, normalizer or None, srgb)
_FORMATS = {
    "DXGI_FORMAT_R32G32B32A32_FLOAT": (np.float32, 4, 16, None, False),
    "DXGI_FORMAT_R16G16B16A16_FLOAT": (np.float16, 4, 8, None, False),
    "DXGI_FORMAT_R16G16B16A16_UNORM": (np.uint16, 4, 8, 65535.0, False),
    "DXGI_FORMAT_R32G32_FLOAT":       (np.float32, 2, 8, None, False),
    "DXGI_FORMAT_R32_FLOAT":          (np.float32, 1, 4, None, False),
    "DXGI_FORMAT_R16_FLOAT":          (np.float16, 1, 2, None, False),
    "DXGI_FORMAT_R8G8B8A8_UNORM":      (np.uint8, 4, 4, 255.0, False),
    "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB": (np.uint8, 4, 4, 255.0, True),
    "DXGI_FORMAT_R8G8B8A8_TYPELESS":   (np.uint8, 4, 4, 255.0, False),
    "DXGI_FORMAT_B8G8R8A8_UNORM":      (np.uint8, 4, 4, 255.0, False),  # BGRA, swizzled below
    "DXGI_FORMAT_R8_UNORM":           (np.uint8, 1, 1, 255.0, False),
    # integer formats — read as raw ints (no normalization), e.g. lookup/index maps
    "DXGI_FORMAT_R32G32B32A32_UINT":  (np.uint32, 4, 16, None, False),
    "DXGI_FORMAT_R32G32_UINT":        (np.uint32, 2, 8, None, False),
    "DXGI_FORMAT_R32_UINT":           (np.uint32, 1, 4, None, False),
    "DXGI_FORMAT_R32_SINT":           (np.int32, 1, 4, None, False),
    "DXGI_FORMAT_R16G16B16A16_UINT":  (np.uint16, 4, 8, None, False),
    "DXGI_FORMAT_R16G16_UINT":        (np.uint16, 2, 4, None, False),
    "DXGI_FORMAT_R16_UINT":           (np.uint16, 1, 2, None, False),
    "DXGI_FORMAT_R8G8B8A8_UINT":      (np.uint8, 4, 4, None, False),
    "DXGI_FORMAT_R8_UINT":            (np.uint8, 1, 1, None, False),
}
_BC = {"DXGI_FORMAT_BC1", "DXGI_FORMAT_BC2", "DXGI_FORMAT_BC3", "DXGI_FORMAT_BC4",
       "DXGI_FORMAT_BC5", "DXGI_FORMAT_BC6H", "DXGI_FORMAT_BC7"}

def _is_bc(fmt):
    base = fmt.replace("DXGI_FORMAT_", "")
    for k in ("BC1", "BC2", "BC3", "BC4", "BC5", "BC6H", "BC7"):
        if base.startswith(k):
            return True
    return False

def decode_subresource(blob, fp):
    """Decode one placed-subresource (dict from footprints) -> numpy array (H,W,C) float32.
    Raises NotImplementedError for block-compressed formats (no decoder available)."""
    fmt = fp["format"]; w = fp["width"]; h = fp["height"]; rp = fp["row_pitch"]; off = fp["offset"]
    if _is_bc(fmt):
        raise NotImplementedError(f"{fmt} is block-compressed; export raw with dump_subresource_raw()")
    if fmt == "DXGI_FORMAT_R11G11B10_FLOAT":
        return _decode_r11g11b10(blob, off, w, h, rp)
    if fmt not in _FORMATS:
        raise NotImplementedError(f"format {fmt} not supported yet")
    dtype, ch, bpp, norm, srgb = _FORMATS[fmt]
    rows = []
    for y in range(h):
        start = off + y * rp
        row = np.frombuffer(blob, dtype=dtype, count=w * ch, offset=start)
        rows.append(row.reshape(w, ch))
    arr = np.stack(rows, 0).astype(np.float32)
    if norm: arr = arr / norm
    if fmt.startswith("DXGI_FORMAT_B8G8R8A8"):
        arr = arr[..., [2, 1, 0, 3]]
    if srgb:  # to linear
        a = arr.copy()
        lo = a <= 0.04045
        arr = np.where(lo, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return arr

def _decode_r11g11b10(blob, off, w, h, rp):
    out = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        u = np.frombuffer(blob, dtype=np.uint32, count=w, offset=off + y * rp)
        r = u & 0x7FF; g = (u >> 11) & 0x7FF; b = (u >> 22) & 0x3FF
        out[y, :, 0] = _f11(r); out[y, :, 1] = _f11(g); out[y, :, 2] = _f10(b)
    return out

def _float_from_bits(mant, exp, mbits):
    # minifloat with no sign; exp bias = 15
    val = np.where(exp == 0,
                   mant.astype(np.float32) / (1 << mbits) * (2.0 ** -14),
                   (1.0 + mant.astype(np.float32) / (1 << mbits)) * (2.0 ** (exp.astype(np.float32) - 15)))
    return val

def _f11(u):  # 5 exp, 6 mant
    return _float_from_bits(u & 0x3F, (u >> 6) & 0x1F, 6)

def _f10(u):  # 5 exp, 5 mant
    return _float_from_bits(u & 0x1F, (u >> 5) & 0x1F, 5)

def subresource_index(mip, array_slice, mip_count):
    return array_slice * mip_count + mip

def get_footprint(exp_res, subresource):
    for fp in exp_res["footprints"]:
        if fp["subresource"] == subresource:
            return fp
    raise KeyError(f"subresource {subresource} not found")

def array_stats(arr):
    a = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
    return {
        "shape": list(arr.shape),
        "min": [float(x) for x in np.nanmin(a, 0)],
        "max": [float(x) for x in np.nanmax(a, 0)],
        "mean": [float(x) for x in np.nanmean(a, 0)],
        "has_nan": bool(np.isnan(a).any()),
    }

def is_buffer_resource(res):
    """True for raw/structured buffers (no texture footprints to decode). These are read
    as raw bytes / 32-bit words rather than a (H,W,C) image."""
    return res.get("dimension") == "D3D12_RESOURCE_DIMENSION_BUFFER"

def _safe_float(x):
    # JSON has no NaN/Infinity; null them out so the transport stays valid.
    x = float(x)
    return x if (x == x and abs(x) != float("inf")) else None

def _dump_words(raw):
    """Interpret a byte string as little-endian 32-bit words, returning parallel
    float / uint views (plus a NaN-sanitized float list and a had_nan flag)."""
    nwords = len(raw) // 4
    floats = list(struct.unpack_from("<%df" % nwords, raw, 0)) if nwords else []
    uints = list(struct.unpack_from("<%dI" % nwords, raw, 0)) if nwords else []
    had_nan = any(f != f for f in floats)  # NaN != NaN
    safe = [_safe_float(f) for f in floats]
    return floats, uints, safe, had_nan

# --------------------------------------------------------------------------------------
# Struct overlay: parse an HLSL/C++ struct definition string and map a buffer's bytes
# onto named fields. Assumes tight 4-byte scalar packing (StructuredBuffer layout, NOT
# cbuffer 16-byte packing). Supports scalar/vector/matrix types, fixed arrays, nested
# structs, #define integer constants, and #if/#else (takes the #if branch, skips #else).
# --------------------------------------------------------------------------------------

_QUALIFIERS = {"row_major", "column_major", "ROW_MAJOR", "COLUMN_MAJOR", "const",
               "static", "inline", "volatile", "precise", "globallycoherent", "unsigned"}
_TYPE_ALIASES = {"dword": "uint", "DWORD": "uint", "float1": "float",
                 "uint1": "uint", "int1": "int", "bool": "uint"}

def _scalar_kind_count(typename):
    """('f'|'i'|'u', scalar_count) for a scalar/vector/matrix type, else None."""
    t = _TYPE_ALIASES.get(typename, typename)
    kinds = {"float": "f", "half": "f", "double": "f", "int": "i", "uint": "u"}
    m = re.fullmatch(r"(float|half|double|int|uint)(\d)x(\d)", t)
    if m:
        return kinds[m.group(1)], int(m.group(2)) * int(m.group(3))
    m = re.fullmatch(r"(float|half|double|int|uint)([2-4])?", t)
    if m:
        return kinds[m.group(1)], int(m.group(2)) if m.group(2) else 1
    return None

def _strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)

def _select_preprocessor_branches(s):
    """Drop preprocessor directive lines, taking the #if branch and skipping the #else
    branch (so e.g. `#if BAKER_ONLY <real> #else <padding> #endif` keeps <real>)."""
    out, stack = [], []  # stack entries: True = emitting this branch, False = skipping
    for ln in s.split("\n"):
        st = ln.strip()
        if re.match(r"#\s*(if|ifdef|ifndef)\b", st):
            stack.append(all(stack) if stack else True)  # take if-branch unless already skipping
        elif re.match(r"#\s*elif\b", st):
            if stack:
                stack[-1] = False  # already took the if-branch; skip the rest
        elif re.match(r"#\s*else\b", st):
            if stack:
                stack[-1] = (not stack[-1]) and all(stack[:-1])
        elif re.match(r"#\s*endif\b", st):
            if stack:
                stack.pop()
        elif st.startswith("#"):
            continue  # #define / #include / #pragma — handled elsewhere or ignored
        elif all(stack) if stack else True:
            out.append(ln)
    return "\n".join(out)

def _parse_defines(s):
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"#\s*define\s+(\w+)\s+(\d+)\b", s)}

def _eval_dim(expr, defines):
    for k, v in defines.items():
        expr = re.sub(r"\b%s\b" % re.escape(k), str(v), expr)
    if re.fullmatch(r"[\d\s/*+\-()]+", expr or ""):
        try:
            return int(eval(expr))  # charset-restricted above
        except Exception:
            pass
    raise ValueError(f"cannot evaluate array dimension {expr!r} (define unknown?)")

def _extract_struct_bodies(s):
    """Brace-matched {name: body} for every `struct Name { ... }` (handles method bodies)."""
    bodies = {}
    for m in re.finditer(r"struct\s+(\w+)\s*\{", s):
        depth, j = 0, m.end() - 1
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        bodies[m.group(1)] = s[m.end():j]
    return bodies

def _iter_decls(body, defines):
    """Yield (type, name, count) for each field declaration in a struct body."""
    body = _select_preprocessor_branches(body)
    for _ in range(6):  # strip nested method bodies { ... }
        body, n = re.subn(r"\{[^{}]*\}", " ", body)
        if not n:
            break
    for decl in body.split(";"):
        decl = decl.strip()
        if not decl or "(" in decl:  # blank or method
            continue
        parts = [p for p in decl.split() if p not in _QUALIFIERS]
        if len(parts) < 2:
            continue
        typ = parts[0]
        m = re.match(r"(\w+)\s*(?:\[([^\]]*)\])?", " ".join(parts[1:]))
        if not m:
            continue
        count = _eval_dim(m.group(2), defines) if m.group(2) else 1
        yield typ, m.group(1), count

def parse_struct_layout(struct_def, struct_name=None):
    """Flatten a struct definition string to [{name,type,kind,scalars,offset}], total_size.
    `struct_name` selects the root struct (default: the last one defined, which in C/HLSL
    is the top-level type since dependencies precede it)."""
    s = _strip_comments(struct_def)
    defines = _parse_defines(s)
    bodies = _extract_struct_bodies(s)
    if not bodies:
        raise ValueError("no `struct Name { ... }` found in struct_def")
    root = struct_name or list(bodies)[-1]
    if root not in bodies:
        raise ValueError(f"struct {root!r} not found; defined: {list(bodies)}")
    out = []

    def flatten(struct, prefix, off):
        for typ, name, count in _iter_decls(bodies[struct], defines):
            typ = _TYPE_ALIASES.get(typ, typ)
            for idx in range(count):
                fn = f"{prefix}{name}" if count == 1 else f"{prefix}{name}[{idx}]"
                if typ in bodies:
                    off = flatten(typ, fn + ".", off)
                else:
                    kc = _scalar_kind_count(typ)
                    if kc is None:
                        raise ValueError(f"unknown type {typ!r} for field {fn!r}")
                    kind, n = kc
                    out.append({"name": fn, "type": typ, "kind": kind,
                                "scalars": n, "offset": off})
                    off += n * 4
        return off

    total = flatten(root, "", 0)
    return out, total

def _field_value(blob, fld):
    """Decode a field's scalar(s) from the buffer bytes -> scalar or list."""
    off, n, kind = fld["offset"], fld["scalars"], fld["kind"]
    code = {"f": "<f", "i": "<i", "u": "<I"}[kind]
    vals = []
    for k in range(n):
        raw = blob[off + k * 4: off + k * 4 + 4]
        if len(raw) < 4:
            vals.append(None)
        else:
            v = struct.unpack(code, raw)[0]
            vals.append(_safe_float(v) if kind == "f" else int(v))
    return vals[0] if n == 1 else vals

def _val_str(v):
    if isinstance(v, list):
        return ", ".join(_val_str(x) for x in v)
    if v is None:
        return "nan"
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)

def _render_table(headers, rows):
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])

# --------------------------------------------------------------------------------------
# High-level: resolve a dispatch's bindings, then extract / compare resources
# --------------------------------------------------------------------------------------

def _the_dispatch(exp):
    if not exp.dispatches:
        raise RuntimeError("no Dispatch found in this region")
    return exp.dispatches[-1]

def _slot_register(ranges, rel):
    """Map a table-relative descriptor index to (reg_class, register, space) using the
    root signature's descriptor ranges, or (None, None, None) if unknown."""
    for rg in ranges or []:
        if rg["offset"] <= rel < rg["offset"] + rg["num"]:
            return (_RANGE_TYPE_CLASS.get(rg["range_type"]),
                    rg["base"] + (rel - rg["offset"]), rg["space"])
    return (None, None, None)

def binding_table(exp, dispatch=None):
    """Flatten a dispatch's root bindings into ordered cbv/srv/uav lists with resource info.
    Each srv/uav item also carries its HLSL register ('reg' like 'u8', plus register/space)
    resolved from the root signature ranges, when available."""
    d = dispatch or _the_dispatch(exp)
    sig = getattr(exp, "root_sigs", {}).get(d["root_signature"], {})
    srv, uav = [], []
    for param in sorted(d["root_tables"]):
        start = d["root_tables"][param]
        ranges = sig.get(param, {}).get("ranges") or []
        size = sig.get(param, {}).get("table_size") or 0
        # An unbounded/bindless table reports an enormous table_size (e.g. ~2^33), so never
        # iterate range(start, start+size) -- walk only the heap slots that actually exist,
        # stopping at the next table's start (same boundary table_slots uses).
        if size:
            end = start + size
            stop = set(d["root_tables"].values()) - {start}
            slots = []
            for s in sorted(exp.descriptors):
                if s < start:
                    continue
                if s >= end or (s != start and s in stop):
                    break
                slots.append(exp.descriptors[s])
                if len(slots) >= 8192:
                    break
        else:
            slots = exp.table_slots(start, set(d["root_tables"].values()))
        for desc in slots:
            r = exp.resources.get(desc["resource"], {})
            cls, reg, space = _slot_register(ranges, desc["slot"] - start)
            item = {
                "slot": desc["slot"], "root_param": param,
                "resource_id": desc["resource"], "resource_name": r.get("name"),
                "view_format": desc["format"], "view_shape": desc["shape"],
                "res_format": r.get("format"), "width": r.get("width"),
                "height": r.get("height"), "array_or_depth": r.get("array_or_depth"),
                "mips": r.get("mips"),
                "reg": (f"{cls}{reg}" if cls is not None else None),
                "register": reg, "space": space, "reg_class": cls,
            }
            (uav if desc["view"] == "UnorderedAccess" else srv).append(item)
    cbv = []
    for param in sorted(d["root_cbv"]):
        res, off = d["root_cbv"][param]
        rd = sig.get(param, {}).get("reg")
        cbv.append({"root_param": param, "resource_id": res, "offset": off,
                    "reg": (f"b{rd[0]}" if rd else None),
                    "register": (rd[0] if rd else None),
                    "space": (rd[1] if rd else None), "reg_class": ("b" if rd else None)})
    return {"groups": d["groups"], "root_signature": d["root_signature"],
            "cbv": cbv, "srv": srv, "uav": uav}

def _mark_used(bt, bindings):
    """Annotate each srv/uav/cbv item with used=True/False from the shader's declared
    resource bindings (list of {class,space,lo,hi}). No-op if bindings is None."""
    if bindings is None:
        return 0
    def is_used(item):
        c, reg, sp = item.get("reg_class"), item.get("register"), item.get("space")
        if c is None or reg is None:
            return None  # can't tell
        return any(b["class"] == c and b["space"] == sp and b["lo"] <= reg <= b["hi"]
                   for b in bindings)
    n_used = 0
    for role in ("cbv", "srv", "uav"):
        for item in bt[role]:
            u = is_used(item)
            item["used"] = u
            if u:
                n_used += 1
    return n_used

def describe_dispatch(wpix, global_id, used_only=False):
    """Bindings for a dispatch, with cbv/srv/uav lists in the SAME order that
    extract()/compare() selectors use — so a {'uav':2} shown here resolves to the
    same binding there.

    The table is built from the per-event region recapture (the source extract/compare
    read from); each resource's 8-char-truncated name is then upgraded to its full debug
    name from a direct full-capture export, where that metadata is intact. If the region
    recapture failed to parse the root CBV, the CBV is back-filled from the full export
    (flagged with "from":"full_export") so it is at least visible.

    Shader reflection (the compute kernel's PSV0 chunk) marks each binding 'used' True/
    False — useful when many kernels share one root signature that binds far more than any
    single kernel touches. used=None means it couldn't be determined. used_only=True drops
    the bound-but-unused srv/uav/cbv entries (selector indices then refer to the kept set)."""
    A = load_export(export_region(wpix, global_id, global_id))
    d = next((x for x in A.dispatches if x["global_id"] == global_id), None) \
        or (A.dispatches[-1] if A.dispatches else None)
    bt = binding_table(A, d)
    full = None
    try:
        full = load_export(export_full(wpix))
    except Exception:
        pass
    for role in ("srv", "uav"):
        for item in bt[role]:
            rid = item.get("resource_id")
            if rid is not None and rid in A.resources:
                item["resource_name"] = canonical_name(wpix, A.resources[rid])
    if not bt["cbv"] and full is not None:
        fd = next((d for d in full.dispatches if d["global_id"] == global_id), None)
        if fd is not None:
            for param in sorted(fd["root_cbv"]):
                res, off = fd["root_cbv"][param]
                bt["cbv"].append({"root_param": param, "resource_id": res,
                                  "offset": off, "from": "full_export"})
    bindings = A.shader_resource_bindings(d.get("pso") if d else None)
    if bindings is not None:
        n_used = _mark_used(bt, bindings)
        bt["reflection"] = {"available": True, "pso": d.get("pso"),
                            "declared_resources": len(bindings), "bound_used": n_used}
        if used_only:
            for role in ("cbv", "srv", "uav"):
                bt[role] = [it for it in bt[role] if it.get("used") is not False]
    else:
        bt["reflection"] = {"available": False,
                            "note": "no PSV0/DXIL for this dispatch's PSO; bindings unfiltered"}
    return bt

def _resolve(exp, selector, bt):
    """selector: one of {'srv':i},{'uav':i},{'cbv':i},{'slot':n},{'name':str}."""
    if "slot" in selector:
        desc = exp.descriptors[int(selector["slot"])]
        r = exp.resources.get(desc["resource"], {})
        role = "uav" if desc["view"] == "UnorderedAccess" else "srv"
        return {"role": role, "resource_id": desc["resource"],
                "resource_name": r.get("name"), "view_format": desc["format"]}
    if "name" in selector:
        # region exports truncate debug names to 8 chars, so match loosely
        rid = exp.find_resource_loose(selector["name"])
        if rid is None:
            raise KeyError(
                f"no resource named {selector['name']!r} in this region "
                f"(region names are truncated to 8 chars; try a {{'uav':i}}/{{'srv':i}} "
                f"index from describe_dispatch instead)")
        role = "uav" if any(u["resource_id"] == rid for u in bt["uav"]) else "srv"
        return {"role": role, "resource_id": rid,
                "resource_name": exp.resources[rid].get("name"),
                "view_format": exp.resources[rid].get("format")}
    for role in ("srv", "uav", "cbv"):
        if role in selector:
            item = bt[role][int(selector[role])]
            return {"role": role, "resource_id": item["resource_id"],
                    "resource_name": item.get("resource_name"),
                    "view_format": item.get("view_format"),
                    "offset": item.get("offset")}
    raise ValueError(f"bad selector {selector}")

def _resolve_cbv_via_full(wpix, global_id, selector):
    """Resolve a {'cbv':i} selector against the full-capture export, used when the
    per-event region recapture failed to parse the dispatch's root CBV. Resource ids
    are stable across exports of the same capture, so the returned id reads correctly
    from the region blob."""
    full = load_export(export_full(wpix))
    fd = next((d for d in full.dispatches if d["global_id"] == global_id), None)
    if fd is None:
        raise KeyError(f"dispatch {global_id} not found in full export for cbv resolution")
    params = sorted(fd["root_cbv"])
    idx = int(selector.get("cbv", 0))
    if idx >= len(params):
        raise IndexError(f"cbv index {idx} out of range ({len(params)} cbv bindings)")
    res, off = fd["root_cbv"][params[idx]]
    return {"role": "cbv", "resource_id": res, "offset": off,
            "resource_name": full.resources.get(res, {}).get("name")}

_MATCH_KEYS = ("name", "width", "height", "array_or_depth", "mips", "format", "dimension")

def _match_output_resource(A, rid_a, B):
    """Find resource `rid_a` (a UAV in region A) inside the post-dispatch region B.

    Region recaptures renumber resource ids, so the counterpart is matched by (name,
    descriptor). But sibling resources can share an identical name AND descriptor — e.g.
    the two `HistoryRemap` buffers (currentToPast / pastToCurrent) — which a plain
    'first match wins' lookup silently confuses (and a stale cache then serves the wrong
    sibling's bytes). Disambiguate by ordinal: the k-th same-(name,desc) resource in A,
    ordered by id, maps to the k-th in B. Falls back to the legacy name+desc match."""
    sig = tuple(A.resources[rid_a].get(k) for k in _MATCH_KEYS)
    same_a = sorted(r for r in A.resources
                    if tuple(A.resources[r].get(k) for k in _MATCH_KEYS) == sig)
    same_b = sorted(r for r in B.resources
                    if tuple(B.resources[r].get(k) for k in _MATCH_KEYS) == sig)
    if rid_a in same_a and same_a.index(rid_a) < len(same_b):
        return same_b[same_a.index(rid_a)]
    like = {k: A.resources[rid_a].get(k) for k in
            ("width", "height", "array_or_depth", "mips", "format", "dimension")}
    return B.find_resource(name=A.resources[rid_a].get("name"), like=like)

def _save_raw_bin(path, raw):
    """Write exact bytes to a file (creating parent dirs) for byte-accurate interop with
    other programs. Returns {"saved_bin","saved_bytes"} to merge into the result dict."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(raw)
    return {"saved_bin": path, "saved_bytes": len(raw)}

def extract(wpix, global_id, selector, mip=0, array_slice=0, out_npy=None,
            cbv_size=None, struct_def=None, struct_name=None, save_bin=None):
    """Extract a dispatch input/output as numpy (texture), or bytes/32-bit words
    (cbv and raw/structured buffers). Returns a JSON-able dict with stats; optionally
    saves .npy. For buffers, cbv_size caps how many bytes are read (default: whole buffer).
    If struct_def (an HLSL/C++ struct definition string) is given for a buffer, the bytes
    are overlaid onto named fields ('fields' list + a readable 'table').

    save_bin writes the EXACT raw bytes (no decoding, no truncation) to a .bin file, the
    portable format other programs can read directly (e.g. C fread / numpy.fromfile). For a
    CBV/buffer that is the byte window actually read (size capped by cbv_size); for a texture
    it is the raw decoded subresource buffer. The JSON result is unchanged (still truncated);
    the .bin holds the full data."""
    A = load_export(export_region(wpix, global_id, global_id))
    bt = binding_table(A)
    is_cbv = ("cbv" in selector) or (selector.get("role") == "cbv")
    if is_cbv and not bt["cbv"]:
        tgt = _resolve_cbv_via_full(wpix, global_id, selector)
    else:
        tgt = _resolve(A, selector, bt)

    if tgt["role"] == "cbv":
        off = tgt["offset"]
        rid = tgt["resource_id"]
        blob = decompress_blob(A.dir, rid, A) if rid in A.resources else b""
        avail = max(0, len(blob) - off)
        if avail == 0:
            raise RuntimeError(
                f"constant buffer (resource {rid}) bytes are not in the region recapture of "
                f"event {global_id}; PIX did not capture this CBV's contents here. The bindings "
                f"are visible via describe_dispatch, but the values cannot be read for this event.")
        # default to a single 256B constant-buffer-ish window; never read past the buffer.
        # NOTE: there is no per-CBV-view size in the capture, so pass cbv_size=sizeof(struct)
        # to avoid reading adjacent constants/uninitialized bytes.
        n = min(cbv_size or 256, avail)
        raw = blob[off:off + n]
        nwords = len(raw) // 4
        floats = list(struct.unpack_from("<%df" % nwords, raw, 0))
        uints = list(struct.unpack_from("<%dI" % nwords, raw, 0))
        had_nan = any(f != f for f in floats)  # NaN != NaN
        # JSON has no NaN/Infinity; null them out (uints/hex keep the raw values).
        safe_floats = [f if (f == f and abs(f) != float("inf")) else None for f in floats]
        out = {"role": "cbv", "resource_name": tgt.get("resource_name"),
               "offset": off, "byte_size": len(raw), "buffer_bytes_after_offset": avail,
               "had_nan": had_nan, "floats": safe_floats[:256], "uints": uints[:256],
               "hex": raw[:128].hex()}
        if had_nan:
            out["note"] = ("some words are NaN when read as float — this is normal for uint/"
                           "padding fields (e.g. 0xCCCCCCCC); if unexpected, you may be reading "
                           "past the struct — pass cbv_size=sizeof(struct).")
        if save_bin:
            out.update(_save_raw_bin(save_bin, raw))
        return out

    is_output = (tgt["role"] == "uav")
    if is_output:
        # outputs = state AFTER the dispatch -> next event's region; match by name+desc
        nxt = next_global_id(wpix, global_id)
        if nxt is None:
            raise RuntimeError("no event after this dispatch to snapshot its output")
        B = load_export(export_region(wpix, nxt, nxt))
        bid = _match_output_resource(A, tgt["resource_id"], B)
        if bid is None:
            raise RuntimeError("output resource not found in post-dispatch snapshot")
        exp, rid = B, bid
    else:
        exp, rid = A, tgt["resource_id"]

    res = exp.resources[rid]
    if is_buffer_resource(res):
        # raw/structured buffer: no texture footprint, dump as bytes / 32-bit words.
        blob = decompress_blob(exp.dir, rid, exp)
        n = min(cbv_size or len(blob), len(blob))
        raw = blob[:n]
        _, uints, safe, had_nan = _dump_words(raw)
        info = {"role": tgt["role"], "resource_name": canonical_name(wpix, res),
                "resource_id_in_region": rid, "region": "after" if is_output else "before",
                "kind": "buffer", "byte_size": len(blob), "bytes_read": len(raw),
                "had_nan": had_nan, "floats": safe[:256], "uints": uints[:256],
                "hex": raw[:128].hex()}
        if had_nan:
            info["note"] = ("some words are NaN as float — normal for uint/padding fields; "
                            "uints/hex hold the raw values.")
        if struct_def:
            layout, parsed = parse_struct_layout(struct_def, struct_name)
            if parsed != len(blob):
                info["struct_warning"] = (f"parsed struct is {parsed} bytes but buffer is "
                                          f"{len(blob)}; field offsets may be misaligned.")
            info["fields"] = [{"name": f["name"], "type": f["type"], "offset": f["offset"],
                               "value": _field_value(blob, f)} for f in layout]
            info["table"] = _render_table(
                ["offset", "field", "type", "value"],
                [[f["offset"], f["name"], f["type"], _val_str(_field_value(blob, f))]
                 for f in layout])
        if out_npy:
            np.save(out_npy, np.frombuffer(raw[:len(raw) // 4 * 4], np.uint32))
            info["saved_npy"] = os.path.abspath(out_npy)
        if save_bin:
            info.update(_save_raw_bin(save_bin, raw))
        return info
    sub = subresource_index(mip, array_slice, res["mips"])
    fp = get_footprint(res, sub)
    arr = decode_subresource(decompress_blob(exp.dir, rid, exp), fp)
    info = {"role": tgt["role"], "resource_name": canonical_name(wpix, res),
            "resource_id_in_region": rid, "region": "after" if is_output else "before",
            "subresource": sub, "mip": mip, "array_slice": array_slice,
            "format": res["format"], "stats": array_stats(arr)}
    if out_npy:
        np.save(out_npy, arr)
        info["saved_npy"] = os.path.abspath(out_npy)
    if save_bin:
        info.update(_save_raw_bin(save_bin, np.ascontiguousarray(arr).tobytes()))
        info["bin_dtype"] = str(arr.dtype)
        info["bin_shape"] = list(arr.shape)
    return info

def _load_for_compare(wpix, global_id, selector, mip, array_slice):
    A = load_export(export_region(wpix, global_id, global_id))
    bt = binding_table(A)
    tgt = _resolve(A, selector, bt)
    is_output = (tgt["role"] == "uav")
    if is_output:
        nxt = next_global_id(wpix, global_id)
        B = load_export(export_region(wpix, nxt, nxt))
        rid = _match_output_resource(A, tgt["resource_id"], B); exp = B
    else:
        rid = tgt["resource_id"]; exp = A
    res = exp.resources[rid]
    if is_buffer_resource(res):
        # buffers can't be decoded as a texture; hand back raw bytes for byte/word diffing.
        return decompress_blob(exp.dir, rid, exp), res
    fp = get_footprint(res, subresource_index(mip, array_slice, res["mips"]))
    return decode_subresource(decompress_blob(exp.dir, rid, exp), fp), res

def _buffer_diff(a, b, ra, rb, out_npy_diff=None, struct_def=None, struct_name=None):
    """Byte/word-level diff of two raw buffers. Reports identical/equal, differing word &
    byte counts, and a sample of the first differing 32-bit words shown three ways
    (uint / hex / float) so a structured-buffer field can be identified by offset.
    If struct_def is given, also returns a per-field 'differing_fields' list and a
    readable 'table' (field | type | a | b) instead of relying on raw word offsets."""
    a = bytes(a); b = bytes(b)
    na, nb = len(a), len(b)
    n = min(na, nb)
    wn = n // 4
    aw = np.frombuffer(a[:wn * 4], np.uint32)
    bw = np.frombuffer(b[:wn * 4], np.uint32)
    word_diff = aw != bw
    nwd = int(word_diff.sum())
    byte_diff = np.frombuffer(a[:n], np.uint8) != np.frombuffer(b[:n], np.uint8)
    identical = (na == nb) and not bool(byte_diff.any())
    res = {
        "kind": "buffer",
        "equal": bool(identical),
        "identical": bool(identical),
        "format": ra.get("format"),
        "byte_size_a": na, "byte_size_b": nb,
        "num_differing_words": nwd,
        "num_differing_bytes": int(byte_diff.sum()) + abs(na - nb),
    }
    if na != nb:
        res["note"] = f"buffers differ in size; compared common prefix of {n} bytes"
    if struct_def:
        layout, parsed = parse_struct_layout(struct_def, struct_name)
        if parsed != na or parsed != nb:
            res["struct_warning"] = (f"parsed struct is {parsed} bytes but buffers are "
                                     f"{na}/{nb}; field offsets may be misaligned.")
        diffs = []
        for f in layout:
            o, w = f["offset"], f["scalars"] * 4
            if a[o:o + w] != b[o:o + w]:
                diffs.append({"name": f["name"], "type": f["type"], "offset": o,
                              "a": _field_value(a, f), "b": _field_value(b, f)})
        res["num_differing_fields"] = len(diffs)
        res["differing_fields"] = diffs
        res["table"] = _render_table(
            ["offset", "field", "type", "Rtxpt/A", "Unity/B"],
            [[d["offset"], d["name"], d["type"], _val_str(d["a"]), _val_str(d["b"])]
             for d in diffs]) if diffs else "(all fields identical)"
    if nwd:
        af = np.frombuffer(a[:wn * 4], np.float32)
        bf = np.frombuffer(b[:wn * 4], np.float32)
        idx = np.nonzero(word_diff)[0]
        res["first_diff_word"] = int(idx[0])
        res["first_diff_byte_offset"] = int(idx[0]) * 4
        res["differing_words"] = [{
            "word": int(i), "byte_offset": int(i) * 4,
            "a_uint": int(aw[i]), "b_uint": int(bw[i]),
            "a_hex": "0x%08x" % int(aw[i]), "b_hex": "0x%08x" % int(bw[i]),
            "a_float": _safe_float(af[i]), "b_float": _safe_float(bf[i]),
        } for i in idx[:32]]
    if out_npy_diff:
        np.save(out_npy_diff, aw.astype(np.int64) - bw.astype(np.int64))
        res["saved_diff_npy"] = os.path.abspath(out_npy_diff)
    return res

def compare(spec_a, spec_b, out_npy_diff=None, struct_def=None, struct_name=None):
    """spec_*: dict(wpix, global_id, selector, mip=0, array_slice=0). Returns diff stats.
    For buffers, pass struct_def (HLSL/C++ struct text) to get a per-field 'differing_fields'
    list and a readable 'table' instead of raw word offsets."""
    wa = spec_a.get("wpix_path") or spec_a["wpix"]
    wb = spec_b.get("wpix_path") or spec_b["wpix"]
    a, ra = _load_for_compare(wa, spec_a["global_id"], spec_a["selector"],
                              spec_a.get("mip", 0), spec_a.get("array_slice", 0))
    b, rb = _load_for_compare(wb, spec_b["global_id"], spec_b["selector"],
                              spec_b.get("mip", 0), spec_b.get("array_slice", 0))
    if isinstance(a, (bytes, bytearray)) or isinstance(b, (bytes, bytearray)):
        return _buffer_diff(a, b, ra, rb, out_npy_diff, struct_def, struct_name)
    if a.shape != b.shape:
        return {"equal": False, "reason": "shape mismatch",
                "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = (a.astype(np.float64) - b.astype(np.float64))
    ad = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    rng = float(max(np.nanmax(a), np.nanmax(b)) - min(np.nanmin(a), np.nanmin(b))) or 1.0
    # psnr is undefined (infinite) when mse==0; leave it null and rely on identical/equal,
    # because JSON has no Infinity (emitting it would break the MCP transport).
    psnr = float(20 * np.log10(rng) - 10 * np.log10(mse)) if mse > 0 else None
    equal = bool(np.array_equal(a, b))
    res = {
        "equal": equal,
        "identical": equal,  # explicit: psnr_db is null when identical (mse==0, infinite PSNR)
        "shape": list(a.shape),
        "format": ra.get("format"),
        "max_abs_diff": float(ad.max()),
        "mean_abs_diff": float(ad.mean()),
        "mse": mse, "psnr_db": psnr,
        "num_differing_texels": int((ad.max(-1) > 0).sum()),
        "per_channel_max_abs": [float(x) for x in ad.reshape(-1, ad.shape[-1]).max(0)],
    }
    # When the stored format is fp16, classify the difference in ULPs of the half grid:
    # max_fp16_ulp ~1 means the two only differ by last-bit half-float rounding.
    if not equal and "16" in (ra.get("format") or "") and "FLOAT" in (ra.get("format") or ""):
        af = a.astype(np.float16)
        ulp = np.abs(np.nextafter(af, np.float16(np.inf)).astype(np.float64) - af.astype(np.float64))
        ulp = np.where(ulp > 0, ulp, np.finfo(np.float16).tiny)
        res["max_fp16_ulp"] = float(np.nanmax(ad / ulp))
    if out_npy_diff:
        np.save(out_npy_diff, diff.astype(np.float32))
        res["saved_diff_npy"] = os.path.abspath(out_npy_diff)
    return res

def _spec_wpix(spec):
    return spec.get("wpix_path") or spec["wpix"]

def _spec_resource(spec):
    """The resource desc (dims/format/mips) a compare-spec points at, without decoding —
    used to enumerate mips/array-slices for compare_subresources."""
    wpix = _spec_wpix(spec); gid = spec["global_id"]; sel = spec["selector"]
    A = load_export(export_region(wpix, gid, gid)); bt = binding_table(A)
    tgt = _resolve(A, sel, bt)
    if tgt["role"] == "uav":
        nxt = next_global_id(wpix, gid)
        B = load_export(export_region(wpix, nxt, nxt))
        rid = _match_output_resource(A, tgt["resource_id"], B); exp = B
    else:
        rid = tgt["resource_id"]; exp = A
    return dict(exp.resources[rid], role=tgt["role"])

def compare_subresources(spec_a, spec_b, mips=None, array_slices=None,
                         struct_def=None, struct_name=None):
    """Compare the same binding across two captures over many subresources in one call.

    spec_*: {wpix_path, global_id, selector}. `mips`/`array_slices` are iterables; when
    omitted they default to the full ranges of the resource (all mips, all faces/slices).
    Returns {per_subresource:[...], aggregate:{...}} where aggregate rolls up the worst
    max_abs_diff, total differing texels, and worst fp16-ULP across all subresources.
    Replaces the manual `for face: for mip: compare(...)` loops. For buffers, pass
    struct_def to also roll up the field-level diff (differing_fields + a readable table)."""
    res = _spec_resource(spec_a)
    if mips is None:
        mips = range(int(res.get("mips", 1)))
    if array_slices is None:
        array_slices = range(int(res.get("array_or_depth", 1)))
    rows = []
    worst_abs = 0.0; total_diff = 0; worst_ulp = 0.0; all_equal = True
    is_buffer = False; total_diff_bytes = 0
    buffer_fields = None; buffer_table = None
    for m in mips:
        for s in array_slices:
            a = dict(spec_a, mip=m, array_slice=s)
            b = dict(spec_b, mip=m, array_slice=s)
            c = compare(a, b, struct_def=struct_def, struct_name=struct_name)
            if c.get("kind") == "buffer":  # buffers have byte/word diffs, not texel stats
                is_buffer = True
                total_diff_bytes += c.get("num_differing_bytes") or 0
                if c.get("differing_fields") is not None:
                    buffer_fields = c.get("differing_fields")
                    buffer_table = c.get("table")
                rows.append({"mip": m, "array_slice": s,
                             "num_differing_bytes": c.get("num_differing_bytes"),
                             "num_differing_words": c.get("num_differing_words"),
                             "equal": c.get("equal")})
            else:
                rows.append({"mip": m, "array_slice": s,
                             "max_abs_diff": c.get("max_abs_diff"),
                             "num_differing_texels": c.get("num_differing_texels"),
                             "max_fp16_ulp": c.get("max_fp16_ulp"),
                             "equal": c.get("equal")})
                worst_abs = max(worst_abs, c.get("max_abs_diff") or 0.0)
                total_diff += c.get("num_differing_texels") or 0
                if c.get("max_fp16_ulp") is not None:
                    worst_ulp = max(worst_ulp, c["max_fp16_ulp"])
            all_equal = all_equal and bool(c.get("equal"))
    agg = {"identical": all_equal, "format": res.get("format"),
           "subresources_compared": len(rows)}
    if is_buffer:
        agg["kind"] = "buffer"
        agg["total_differing_bytes"] = total_diff_bytes
        if buffer_fields is not None:
            agg["num_differing_fields"] = len(buffer_fields)
            agg["differing_fields"] = buffer_fields
            agg["field_table"] = buffer_table
    else:
        agg["max_abs_diff"] = worst_abs
        agg["total_differing_texels"] = total_diff
        if worst_ulp:
            agg["max_fp16_ulp"] = worst_ulp
            agg["within_1_ulp"] = worst_ulp <= 1.0 + 1e-6
    return {"aggregate": agg, "per_subresource": rows}

def _match_struct_def(struct_defs, name):
    """Pick a struct_def for an output resource `name` from a {key: text} map: exact name
    first, then any key that is a substring of the name (or vice-versa)."""
    if not struct_defs or not name:
        return None
    if name in struct_defs:
        return struct_defs[name]
    for k, v in struct_defs.items():
        if k and (k in name or name in k):
            return v
    return None

def diff_stage(wpix_a, wpix_b, marker, dispatch="last", mips="base", used_only=False,
               struct_defs=None):
    """Compare a whole pipeline stage between two captures in one call.

    Finds the Dispatch(es) whose marker path contains `marker` in each capture, picks
    one (`dispatch`: "last" (default, = final resource state) or "first"), then compares
    every UAV output — matched across captures by debug name — and reports a per-output
    verdict: identical / within-1-fp16-ULP / divergent.

    `mips` controls which mip levels to compare (all array slices/faces are always
    compared):
      * "base" (default) — mip 0 only. Correct for a single dispatch, which usually
        writes only the top mip; higher mips at that event are stale (written by a later
        pass) and would produce a spurious "divergent".
      * "all" — every mip. Use only when the chosen dispatch is the last of a chain that
        populated the full mip pyramid (e.g. the final EnvMapBakerMIPs dispatch).

    `used_only` (default False): skip UAVs the kernel doesn't actually reference (per PSV0
    shader reflection). Useful when a capture binds a big shared table (e.g. RTXPT's
    LightsBaker u0–u18) — it confines the diff to real outputs and disambiguates duplicate
    names (e.g. two HistoryRemap views) to the one the shader uses. Degrades to no-op when
    reflection is unavailable.

    `struct_defs` (default None): a {resource_name: HLSL/C++ struct text} map. When a buffer
    output diverges and a struct_def matches its name (exact, else substring), the output
    entry gains a field-level diff — 'num_differing_fields' and a readable 'field_table'
    (field | type | A | B) — instead of just a byte count."""
    def pick(wpix):
        ds = [e["global_id"] for e in find_shader_events(wpix, marker, dispatches_only=True)]
        if not ds:
            raise KeyError(f"no Dispatch under marker {marker!r} in {wpix}")
        return ds[-1] if dispatch == "last" else ds[0]
    gid_a, gid_b = pick(wpix_a), pick(wpix_b)
    # NOTE: keep the UNFILTERED describe so enumerate() indices still match the {uav:i}
    # selectors compare() resolves; used_only only gates which names participate.
    da = describe_dispatch(wpix_a, gid_a)
    db = describe_dispatch(wpix_b, gid_b)
    # index UAVs by full name on each side
    def uav_index(desc):
        out = {}
        for i, u in enumerate(desc["uav"]):
            if used_only and u.get("used") is False:
                continue
            out.setdefault(u.get("resource_name"), i)
        return out
    ia, ib = uav_index(da), uav_index(db)
    mip_arg = None if mips == "all" else [0]
    outputs = []
    for name in sorted(set(ia) & set(ib)):
        c = compare_subresources(
            {"wpix_path": wpix_a, "global_id": gid_a, "selector": {"uav": ia[name]}},
            {"wpix_path": wpix_b, "global_id": gid_b, "selector": {"uav": ib[name]}},
            mips=mip_arg, struct_def=_match_struct_def(struct_defs, name))
        agg = c["aggregate"]
        if agg["identical"]:
            verdict = "identical"
        elif agg.get("within_1_ulp"):
            verdict = "within_1_fp16_ulp"
        else:
            verdict = "divergent"
        outputs.append({"output": name, "verdict": verdict, **agg})
    return {"marker": marker, "dispatch": dispatch, "used_only": used_only,
            "reflection_a": da.get("reflection", {}).get("available"),
            "reflection_b": db.get("reflection", {}).get("available"),
            "global_id_a": gid_a, "global_id_b": gid_b,
            "unmatched_a": sorted(set(ia) - set(ib)),
            "unmatched_b": sorted(set(ib) - set(ia)),
            "outputs": outputs}

def clear_cache():
    import shutil
    # Drop in-process caches too, or the on-disk export dirs they point at are deleted
    # while parsed ExportData / decompressed blobs (100MB+) stay resident in memory.
    _EXPORT_DATA_CACHE.clear()
    _BLOB_CACHE.clear()
    root = _cache_root()
    n = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            n += 1
        except OSError:
            pass
    return {"cleared": n, "cache_dir": root}
