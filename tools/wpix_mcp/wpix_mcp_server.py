"""
wpix-mcp — a self-contained MCP server (stdio / JSON-RPC, no external deps beyond numpy)
exposing PIX .wpix shader input/output inspection for pixel-level comparison.

Tools:
  wpix_find_events       find Dispatch/Draw events (optionally by marker name)
  wpix_describe_dispatch list a dispatch's bound CBVs / SRV inputs / UAV outputs
  wpix_describe_shader   compile-time info for a dispatch's CS (hash/target/defines/flags)
  wpix_extract           decode one input/output (or CBV) to stats + optional .npy
  wpix_compare           diff the same binding across two captures (max-abs/mse/psnr)
  wpix_clear_cache       delete cached exports

Run:  python wpix_mcp_server.py        (speaks MCP over stdin/stdout)
Logs go to stderr only; stdout carries protocol messages exclusively.
"""

import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wpix_core as core

SERVER_INFO = {"name": "wpix-mcp", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"

def log(*a):
    print("[wpix-mcp]", *a, file=sys.stderr, flush=True)

# ----------------------------------------------------------------------------------
# tool implementations -> python objects (serialized to text by the dispatcher)
# ----------------------------------------------------------------------------------

def t_find_events(wpix_path, name_filter=None, dispatches_only=False):
    return {"events": core.find_shader_events(wpix_path, name_filter, dispatches_only)}

def t_stage_times(wpix_path, name_filter=None, dispatches_only=False, work_only=True,
                  group_by="marker", duration_counter="TOP to EOP Duration (ns)", top=None):
    return core.stage_times(wpix_path, name_filter, bool(dispatches_only), bool(work_only),
                            group_by, duration_counter, top)

def t_describe_dispatch(wpix_path, global_id, used_only=False):
    return core.describe_dispatch(wpix_path, int(global_id), bool(used_only))

def t_describe_shader(wpix_path, global_id, disassemble=True, pdb_dir=None):
    return core.describe_shader(wpix_path, int(global_id), bool(disassemble), pdb_dir)

def t_extract(wpix_path, global_id, selector, mip=0, array_slice=0,
              save_npy_path=None, cbv_size=None, struct_def=None, struct_name=None,
              save_bin_path=None):
    return core.extract(wpix_path, int(global_id), selector, int(mip),
                        int(array_slice), save_npy_path,
                        int(cbv_size) if cbv_size else None, struct_def, struct_name,
                        save_bin_path)

def t_compare(a, b, save_diff_npy_path=None, struct_def=None, struct_name=None):
    return core.compare(a, b, save_diff_npy_path, struct_def, struct_name)

def t_compare_subresources(a, b, mips=None, array_slices=None, struct_def=None, struct_name=None):
    return core.compare_subresources(a, b, mips, array_slices, struct_def, struct_name)

def t_diff_stage(wpix_a, wpix_b, marker, dispatch="last", mips="base", used_only=False,
                 struct_defs=None):
    return core.diff_stage(wpix_a, wpix_b, marker, dispatch, mips, bool(used_only), struct_defs)

def t_clear_cache():
    return core.clear_cache()

TOOLS = [
    {
        "name": "wpix_find_events",
        "description": "List GPU events (Dispatch/Draw, each with a global id) in a .wpix "
                       "capture, with their enclosing PIX marker path. Filter by a marker/"
                       "event name substring (e.g. 'ProcSkyBaseBake').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string", "description": "Path to the .wpix capture."},
                "name_filter": {"type": "string", "description": "Optional case-insensitive substring of marker path or event name."},
                "dispatches_only": {"type": "boolean", "default": False, "description": "Keep only Dispatch events (drop ResourceBarrier/Draw)."},
            },
            "required": ["wpix_path"],
        },
        "handler": t_find_events,
    },
    {
        "name": "wpix_stage_times",
        "description": "Aggregate GPU duration counters from a .wpix capture by PIX marker/"
                       "stage. Uses pixtool save-event-list with duration counters, then sums "
                       "the selected counter over matching events. Default counter is "
                       "'TOP to EOP Duration (ns)', usually the closest per-event GPU execution "
                       "duration. By default work_only=true includes compute Dispatch, "
                       "raytracing DispatchRays/accel builds, raster Draw calls and indirect "
                       "work, while excluding barriers/waits/present. Set dispatches_only=true "
                       "for compute-only timing. name_filter narrows to a marker/event "
                       "substring; group_by is marker, leaf, event, or kind; top returns the "
                       "slowest N stages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string", "description": "Path to the .wpix capture."},
                "name_filter": {"type": "string", "description": "Optional case-insensitive substring of marker path or event name."},
                "dispatches_only": {"type": "boolean", "default": False, "description": "Keep only Dispatch events for compute-stage timing."},
                "work_only": {"type": "boolean", "default": True, "description": "Keep shader/dispatch/draw work only: compute, raytracing, raster and indirect events; drop barriers/waits/present."},
                "group_by": {"type": "string", "enum": ["marker", "leaf", "event", "kind"], "default": "marker", "description": "Aggregation key: full marker path, final marker segment, event name, or work kind."},
                "duration_counter": {"type": "string", "default": "TOP to EOP Duration (ns)", "description": "Timing counter column to sum, e.g. TOP to EOP Duration (ns) or EOP to EOP Duration (ns)."},
                "top": {"type": "integer", "description": "Optional: return only the slowest N stages."},
            },
            "required": ["wpix_path"],
        },
        "handler": t_stage_times,
    },
    {
        "name": "wpix_describe_dispatch",
        "description": "For a compute Dispatch identified by its global id, return its thread-"
                       "group counts and root bindings: CBVs, SRV inputs, and UAV outputs, each "
                       "resolved to a resource debug-name, view format, and dimensions. The "
                       "srv/uav/cbv list ordering matches the {srv:i}/{uav:i}/{cbv:i} selector "
                       "indices used by wpix_extract/wpix_compare (resolved from the same region "
                       "recapture, with full debug names restored from the capture export). Each "
                       "srv/uav/cbv item carries its HLSL register ('reg') and a 'used' flag from "
                       "shader reflection (the kernel's PSV0 chunk) — true/false/null. Set "
                       "used_only=true to drop bound-but-unused bindings (common when many kernels "
                       "share one root signature); selector indices then refer to the kept set.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string"},
                "global_id": {"type": "integer", "description": "Global id of the Dispatch (from wpix_find_events)."},
                "used_only": {"type": "boolean", "default": False, "description": "Drop bindings the shader doesn't reference (per PSV0 reflection)."},
            },
            "required": ["wpix_path", "global_id"],
        },
        "handler": t_describe_dispatch,
    },
    {
        "name": "wpix_describe_shader",
        "description": "Compile-time info for the compute shader bound at a Dispatch (by global "
                       "id): shader_hash (the HASH-part digest PIX/RenderDoc show — NOT the "
                       "container header checksum), target/profile, shader_kind, required-feature "
                       "flags and debug/PDB name (all read straight from the DXIL container), plus "
                       "the original DXC command line — entry, defines, includes, flags. The "
                       "command line comes from whichever source has it: embedded debug info "
                       "(-Zi/-Qembed_debug) via dxc -dumpbin, else an external side-car PDB named "
                       "<shader_hash>.pdb (Unity's UnityProject\\ShaderPDB layout) read via "
                       "IDxcPdbUtils2 in dxcompiler.dll. 'args_source' reports which was used "
                       "(embedded_debug | external_pdb). Useful for comparing how the same pass is "
                       "compiled across two apps (e.g. Unity cs_6_6 vs RTXPT cs_6_9, differing "
                       "-D defines). Set disassemble=false for just the container fields. Env: "
                       "WPIX_DXC overrides dxc.exe/dxcompiler.dll; WPIX_PDB_DIR (or the pdb_dir "
                       "arg) overrides the side-car PDB folder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string"},
                "global_id": {"type": "integer", "description": "Global id of the Dispatch (from wpix_find_events)."},
                "disassemble": {"type": "boolean", "default": True, "description": "Recover entry/defines/flags (dxc -dumpbin, then external-PDB fallback); false = container-only fields (hash/target/features)."},
                "pdb_dir": {"type": "string", "description": "Folder of side-car <hash>.pdb files for shaders built without embedded debug (default: auto-detected UnityProject/ShaderPDB, or WPIX_PDB_DIR)."},
            },
            "required": ["wpix_path", "global_id"],
        },
        "handler": t_describe_shader,
    },
    {
        "name": "wpix_extract",
        "description": "Decode one binding of a dispatch to numpy and return per-channel stats "
                       "(min/max/mean/has_nan); optionally save the array as .npy. SRV/CBV inputs "
                       "are read as the resource state before the dispatch; UAV outputs are read "
                       "as the state after it (matched across captures by debug-name). 'selector' "
                       "is one of {\"srv\":i}, {\"uav\":i}, {\"cbv\":i}, {\"slot\":n}, "
                       "{\"name\":\"EnvMapBak\"}. For textures choose mip/array_slice (cube face = "
                       "array_slice 0..5). Raw/structured buffers are dumped as bytes / 32-bit "
                       "words (floats+uints+hex), with cbv_size capping the byte count. Block-"
                       "compressed inputs (e.g. BC6H) are reported but not decoded.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string"},
                "global_id": {"type": "integer"},
                "selector": {"type": "object", "description": "One of {srv:i}|{uav:i}|{cbv:i}|{slot:n}|{name:str}."},
                "mip": {"type": "integer", "default": 0},
                "array_slice": {"type": "integer", "default": 0, "description": "Array slice / cube face (0..5)."},
                "save_npy_path": {"type": "string", "description": "Optional path to save the decoded array as .npy."},
                "save_bin_path": {"type": "string", "description": "Optional path to save the EXACT raw bytes as .bin (no decoding, no truncation) for byte-accurate interop with other programs (C fread / numpy.fromfile). For a CBV/buffer this is the byte window read (size via cbv_size); for a texture it's the raw decoded subresource (dtype/shape returned as bin_dtype/bin_shape)."},
                "cbv_size": {"type": "integer", "description": "Bytes to read for a CBV selector (default 1024)."},
                "struct_def": {"type": "string", "description": "Optional HLSL/C++ struct definition string to overlay on a buffer's bytes (tight 4-byte structured-buffer packing). Returns named 'fields' + a readable 'table'. Paste dependency structs too; nested structs are expanded."},
                "struct_name": {"type": "string", "description": "Root struct name in struct_def (default: last struct defined)."},
            },
            "required": ["wpix_path", "global_id", "selector"],
        },
        "handler": t_extract,
    },
    {
        "name": "wpix_compare",
        "description": "Decode the same binding from two captures (or two dispatches) and report "
                       "difference. Textures: pixel-level max_abs_diff, mean_abs_diff, mse, "
                       "psnr_db, num_differing_texels, per-channel max. Raw/structured buffers: "
                       "byte/word-level num_differing_words, num_differing_bytes, and a sample of "
                       "the first differing 32-bit words (uint/hex/float). Each side is {wpix_path, "
                       "global_id, selector, mip?, array_slice?}. Optionally save the signed diff as .npy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "object", "description": "{wpix_path, global_id, selector, mip?, array_slice?}"},
                "b": {"type": "object", "description": "{wpix_path, global_id, selector, mip?, array_slice?}"},
                "save_diff_npy_path": {"type": "string"},
                "struct_def": {"type": "string", "description": "Optional HLSL/C++ struct definition string. For buffer comparisons, overlays the bytes on named fields and returns a per-field 'differing_fields' list + a readable 'table' (field | type | A | B)."},
                "struct_name": {"type": "string", "description": "Root struct name in struct_def (default: last struct defined)."},
            },
            "required": ["a", "b"],
        },
        "handler": t_compare,
    },
    {
        "name": "wpix_compare_subresources",
        "description": "Compare the same binding across two captures over MANY subresources in "
                       "one call (replaces manual per-face/per-mip loops). 'a'/'b' are "
                       "{wpix_path, global_id, selector}. 'mips'/'array_slices' are optional "
                       "integer lists; omitted = full ranges (all mips, all cube faces/slices). "
                       "Returns an 'aggregate' (worst max_abs_diff, total differing texels, worst "
                       "fp16-ULP, identical flag) plus a 'per_subresource' table.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "object", "description": "{wpix_path, global_id, selector}"},
                "b": {"type": "object", "description": "{wpix_path, global_id, selector}"},
                "mips": {"type": "array", "items": {"type": "integer"}, "description": "Mip levels to compare; omit for all."},
                "array_slices": {"type": "array", "items": {"type": "integer"}, "description": "Array slices / cube faces; omit for all."},
            },
            "required": ["a", "b"],
        },
        "handler": t_compare_subresources,
    },
    {
        "name": "wpix_diff_stage",
        "description": "Compare a whole pipeline stage between two captures in ONE call. Locates "
                       "the Dispatch under marker 'marker' in each capture (dispatch='last' = "
                       "final state, or 'first'), then compares every UAV output (matched across "
                       "captures by debug name) and returns a per-output verdict: identical / "
                       "within_1_fp16_ulp / divergent. mips='base' (default) compares only mip 0 "
                       "(correct for a single dispatch; higher mips are stale there); mips='all' "
                       "compares the full pyramid (use for the final dispatch of a mip-gen chain).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_a": {"type": "string"},
                "wpix_b": {"type": "string"},
                "marker": {"type": "string", "description": "Marker substring, e.g. 'ProcSkyBaseBake', 'GenIM', 'EnvMapBakerMIPs'."},
                "dispatch": {"type": "string", "enum": ["last", "first"], "default": "last"},
                "mips": {"type": "string", "enum": ["base", "all"], "default": "base"},
                "used_only": {"type": "boolean", "default": False, "description": "Skip UAVs the kernel doesn't reference (PSV0 reflection); confines the diff to real outputs and disambiguates duplicate-named views. No-op if reflection unavailable."},
                "struct_defs": {"type": "object", "description": "Map of {resource_name: HLSL/C++ struct text}. When a buffer output diverges and its name matches a key (exact, else substring), the output gains a field-level diff ('num_differing_fields' + readable 'field_table') instead of just a byte count.", "additionalProperties": {"type": "string"}},
            },
            "required": ["wpix_a", "wpix_b", "marker"],
        },
        "handler": t_diff_stage,
    },
    {
        "name": "wpix_clear_cache",
        "description": "Delete cached recapture exports (resources.bin etc). Each cached export "
                       "can be ~200MB+, so clear when done comparing.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": t_clear_cache,
    },
]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}

# ----------------------------------------------------------------------------------
# JSON-RPC / MCP plumbing
# ----------------------------------------------------------------------------------

def make_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}

def make_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

def handle_request(msg):
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return make_result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return make_result(req_id, {})
    if method == "tools/list":
        tools = [{k: t[k] for k in ("name", "description", "inputSchema")} for t in TOOLS]
        return make_result(req_id, {"tools": tools})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = TOOL_BY_NAME.get(name)
        if not tool:
            return make_error(req_id, -32602, f"unknown tool: {name}")
        try:
            out = tool["handler"](**args)
            text = json.dumps(out, indent=2, default=str)
            return make_result(req_id, {"content": [{"type": "text", "text": text}],
                                        "isError": False})
        except Exception as e:
            tb = traceback.format_exc()
            log("tool error:", tb)
            return make_result(req_id, {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}\n{tb}"}],
                "isError": True})

    if req_id is not None:
        return make_error(req_id, -32601, f"method not found: {method}")
    return None  # notification (e.g. notifications/initialized) -> no response

def main():
    log("starting; pixtool =", _safe_pixtool())
    stdin = sys.stdin
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log("bad json:", line[:200]); continue
        msgs = msg if isinstance(msg, list) else [msg]
        for m in msgs:
            resp = handle_request(m)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

def _safe_pixtool():
    try:
        return core.find_pixtool()
    except Exception as e:
        return f"<not found: {e}>"

if __name__ == "__main__":
    main()
