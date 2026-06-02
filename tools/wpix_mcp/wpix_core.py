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

def find_shader_events(wpix, name_filter=None):
    """Events that have a global id (Dispatch/Draw/etc), optionally filtered by parent
    marker name. Returns each with its nearest enclosing named marker for context."""
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
        self._simulate_blob_offsets()
        self._parse_descriptors()
        self._parse_root_signatures()
        self._parse_dispatches()

    # ---- root signatures: id -> {param_index: {type, table_size}} ----
    def _parse_root_signatures(self):
        self.root_sigs = {}
        blk_re = re.compile(
            r'static D3D12_ROOT_PARAMETER1 rootParameters\[\d+\];(.*?)'
            r'CreateAndTrackRootSignature\((\d+),', re.S)
        ptype_re = re.compile(r'rootParameters\[(\d+)\]\.ParameterType\s*=\s*D3D12_ROOT_PARAMETER_TYPE_(\w+)')
        range_re = re.compile(r'descriptorRanges\[\d+\]\s*=\s*\{\s*D3D12_DESCRIPTOR_RANGE_TYPE_(\w+),\s*(\d+)')
        tbl_re = re.compile(r'rootParameters\[(\d+)\]\.DescriptorTable\s*=\s*\{\s*\d+,')
        for f, t in self._read_all("FrameResources_*.cpp").items():
            for blk in blk_re.finditer(t):
                body, sigid = blk.group(1), int(blk.group(2))
                params = {}
                cur = None; cur_count = 0; cur_rtype = None
                for ln in body.split("\n"):
                    m = ptype_re.search(ln)
                    if m:
                        cur = int(m.group(1)); cur_count = 0; cur_rtype = None
                        params[cur] = {"type": m.group(2), "table_size": 0, "range_type": None}
                        continue
                    m = range_re.search(ln)
                    if m and cur is not None:
                        params[cur]["table_size"] += int(m.group(2))
                        if params[cur]["range_type"] is None:
                            params[cur]["range_type"] = m.group(1)
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
        def walk(name):
            cur_rid = None
            mrid = _RESFUNC_RE.fullmatch(name)
            if mrid: cur_rid = int(mrid.group(1))
            for ln in funcs.get(name, []):
                rm = _READ_RE.search(ln)
                if rm:
                    if cur_rid is not None and cur_rid not in self.blob_offset:
                        self.blob_offset[cur_rid] = offset[0]
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
                root_tables = {}; root_cbv = {}; cur_sig = None
                last_gid = None
                for ln in body.split("\n"):
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
                            "root_tables": dict(root_tables),   # param -> start slot
                            "root_cbv": dict(root_cbv),         # param -> (res, offset)
                        })

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

def decompress_blob(export_dir, resource_id, exp=None):
    ckey = (export_dir, resource_id)
    cached = _BLOB_CACHE.get(ckey)
    if cached is not None:
        return cached
    exp = exp or load_export(export_dir)
    if resource_id not in exp.blob_offset:
        raise KeyError(f"resource {resource_id} has no initial data in this region")
    off = exp.blob_offset[resource_id]
    csize = exp.read_size[resource_id]
    cab = ctypes.WinDLL("Cabinet.dll")
    hdec = wintypes.HANDLE()
    if not cab.CreateDecompressor(_COMPRESS_ALGORITHM_XPRESS, None, ctypes.byref(hdec)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with open(os.path.join(export_dir, "resources.bin"), "rb") as fh:
            fh.seek(off); comp = fh.read(csize)
        # query size
        final = ctypes.c_size_t(0)
        cab.Decompress(hdec, comp, ctypes.c_size_t(csize), None, 0, ctypes.byref(final))
        out = (ctypes.c_ubyte * final.value)()
        got = ctypes.c_size_t(0)
        if not cab.Decompress(hdec, comp, ctypes.c_size_t(csize), out,
                              ctypes.c_size_t(final.value), ctypes.byref(got)):
            raise ctypes.WinError(ctypes.get_last_error())
        blob = bytes(out[:got.value])
    finally:
        cab.CloseDecompressor(hdec)
    if len(_BLOB_CACHE) >= _BLOB_CACHE_MAX:
        _BLOB_CACHE.pop(next(iter(_BLOB_CACHE)))  # FIFO eviction
    _BLOB_CACHE[ckey] = blob
    return blob

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

# --------------------------------------------------------------------------------------
# High-level: resolve a dispatch's bindings, then extract / compare resources
# --------------------------------------------------------------------------------------

def _the_dispatch(exp):
    if not exp.dispatches:
        raise RuntimeError("no Dispatch found in this region")
    return exp.dispatches[-1]

def binding_table(exp, dispatch=None):
    """Flatten a dispatch's root bindings into ordered cbv/srv/uav lists with resource info."""
    d = dispatch or _the_dispatch(exp)
    sig = getattr(exp, "root_sigs", {}).get(d["root_signature"], {})
    srv, uav = [], []
    for param in sorted(d["root_tables"]):
        start = d["root_tables"][param]
        size = sig.get(param, {}).get("table_size") or 0
        slots = ([exp.descriptors[s] for s in range(start, start + size)
                  if s in exp.descriptors] if size
                 else exp.table_slots(start, set(d["root_tables"].values())))
        for desc in slots:
            r = exp.resources.get(desc["resource"], {})
            item = {
                "slot": desc["slot"], "root_param": param,
                "resource_id": desc["resource"], "resource_name": r.get("name"),
                "view_format": desc["format"], "view_shape": desc["shape"],
                "res_format": r.get("format"), "width": r.get("width"),
                "height": r.get("height"), "array_or_depth": r.get("array_or_depth"),
                "mips": r.get("mips"),
            }
            (uav if desc["view"] == "UnorderedAccess" else srv).append(item)
    cbv = []
    for param in sorted(d["root_cbv"]):
        res, off = d["root_cbv"][param]
        cbv.append({"root_param": param, "resource_id": res, "offset": off})
    return {"groups": d["groups"], "root_signature": d["root_signature"],
            "cbv": cbv, "srv": srv, "uav": uav}

def describe_dispatch(wpix, global_id):
    # Names/formats/bindings are static frame metadata: read them from a direct export of the
    # original capture so debug names and formats are intact (export_region truncates names and
    # re-encodes formats). Fall back to a region recapture if the dispatch isn't in the export.
    exp = load_export(export_full(wpix))
    d = next((d for d in exp.dispatches if d["global_id"] == global_id), None)
    if d is not None:
        return binding_table(exp, d)
    return binding_table(load_export(export_region(wpix, global_id, global_id)))

def _resolve(exp, selector, bt):
    """selector: one of {'srv':i},{'uav':i},{'cbv':i},{'slot':n},{'name':str}."""
    if "slot" in selector:
        desc = exp.descriptors[int(selector["slot"])]
        r = exp.resources.get(desc["resource"], {})
        role = "uav" if desc["view"] == "UnorderedAccess" else "srv"
        return {"role": role, "resource_id": desc["resource"],
                "resource_name": r.get("name"), "view_format": desc["format"]}
    if "name" in selector:
        rid = exp.find_resource(name=selector["name"])
        if rid is None:
            raise KeyError(f"no resource named {selector['name']}")
        role = "uav" if any(u["resource_id"] == rid for u in bt["uav"]) else "srv"
        return {"role": role, "resource_id": rid, "resource_name": selector["name"],
                "view_format": exp.resources[rid].get("format")}
    for role in ("srv", "uav", "cbv"):
        if role in selector:
            item = bt[role][int(selector[role])]
            return {"role": role, "resource_id": item["resource_id"],
                    "resource_name": item.get("resource_name"),
                    "view_format": item.get("view_format"),
                    "offset": item.get("offset")}
    raise ValueError(f"bad selector {selector}")

def extract(wpix, global_id, selector, mip=0, array_slice=0, out_npy=None,
            cbv_size=None):
    """Extract a dispatch input/output as numpy (texture) or bytes/floats (cbv).
    Returns a JSON-able dict with stats; optionally saves .npy."""
    A = load_export(export_region(wpix, global_id, global_id))
    bt = binding_table(A)
    tgt = _resolve(A, selector, bt)

    if tgt["role"] == "cbv":
        off = tgt["offset"]
        n = cbv_size or 1024
        blob = decompress_blob(A.dir, tgt["resource_id"], A)
        raw = blob[off:off + n]
        nwords = len(raw) // 4
        floats = list(struct.unpack_from("<%df" % nwords, raw, 0))
        uints = list(struct.unpack_from("<%dI" % nwords, raw, 0))
        return {"role": "cbv", "resource_name": tgt["resource_name"],
                "offset": off, "byte_size": len(raw),
                "floats": floats[:256], "uints": uints[:256], "hex": raw[:128].hex()}

    is_output = (tgt["role"] == "uav")
    if is_output:
        # outputs = state AFTER the dispatch -> next event's region; match by name+desc
        nxt = next_global_id(wpix, global_id)
        if nxt is None:
            raise RuntimeError("no event after this dispatch to snapshot its output")
        B = load_export(export_region(wpix, nxt, nxt))
        ra = A.resources[tgt["resource_id"]]
        like = {k: ra[k] for k in ("width", "height", "array_or_depth", "mips",
                                   "format", "dimension")}
        bid = B.find_resource(name=ra.get("name"), like=like)
        if bid is None:
            raise RuntimeError("output resource not found in post-dispatch snapshot")
        exp, rid = B, bid
    else:
        exp, rid = A, tgt["resource_id"]

    res = exp.resources[rid]
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
    return info

def _load_for_compare(wpix, global_id, selector, mip, array_slice):
    A = load_export(export_region(wpix, global_id, global_id))
    bt = binding_table(A)
    tgt = _resolve(A, selector, bt)
    is_output = (tgt["role"] == "uav")
    if is_output:
        nxt = next_global_id(wpix, global_id)
        B = load_export(export_region(wpix, nxt, nxt))
        ra = A.resources[tgt["resource_id"]]
        like = {k: ra[k] for k in ("width", "height", "array_or_depth", "mips",
                                   "format", "dimension")}
        rid = B.find_resource(name=ra.get("name"), like=like); exp = B
    else:
        rid = tgt["resource_id"]; exp = A
    res = exp.resources[rid]
    fp = get_footprint(res, subresource_index(mip, array_slice, res["mips"]))
    return decode_subresource(decompress_blob(exp.dir, rid, exp), fp), res

def compare(spec_a, spec_b, out_npy_diff=None):
    """spec_*: dict(wpix, global_id, selector, mip=0, array_slice=0). Returns diff stats."""
    wa = spec_a.get("wpix_path") or spec_a["wpix"]
    wb = spec_b.get("wpix_path") or spec_b["wpix"]
    a, ra = _load_for_compare(wa, spec_a["global_id"], spec_a["selector"],
                              spec_a.get("mip", 0), spec_a.get("array_slice", 0))
    b, rb = _load_for_compare(wb, spec_b["global_id"], spec_b["selector"],
                              spec_b.get("mip", 0), spec_b.get("array_slice", 0))
    if a.shape != b.shape:
        return {"equal": False, "reason": "shape mismatch",
                "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = (a.astype(np.float64) - b.astype(np.float64))
    ad = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    rng = float(max(np.nanmax(a), np.nanmax(b)) - min(np.nanmin(a), np.nanmin(b))) or 1.0
    psnr = float(20 * np.log10(rng) - 10 * np.log10(mse)) if mse > 0 else None
    res = {
        "equal": bool(np.array_equal(a, b)),
        "shape": list(a.shape),
        "max_abs_diff": float(ad.max()),
        "mean_abs_diff": float(ad.mean()),
        "mse": mse, "psnr_db": psnr,
        "num_differing_texels": int((ad.max(-1) > 0).sum()),
        "per_channel_max_abs": [float(x) for x in ad.reshape(-1, ad.shape[-1]).max(0)],
    }
    if out_npy_diff:
        np.save(out_npy_diff, diff.astype(np.float32))
        res["saved_diff_npy"] = os.path.abspath(out_npy_diff)
    return res

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
