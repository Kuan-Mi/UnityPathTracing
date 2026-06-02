"""
wpix-mcp — a self-contained MCP server (stdio / JSON-RPC, no external deps beyond numpy)
exposing PIX .wpix shader input/output inspection for pixel-level comparison.

Tools:
  wpix_find_events       find Dispatch/Draw events (optionally by marker name)
  wpix_describe_dispatch list a dispatch's bound CBVs / SRV inputs / UAV outputs
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

def t_find_events(wpix_path, name_filter=None):
    return {"events": core.find_shader_events(wpix_path, name_filter)}

def t_describe_dispatch(wpix_path, global_id):
    return core.describe_dispatch(wpix_path, int(global_id))

def t_extract(wpix_path, global_id, selector, mip=0, array_slice=0,
              save_npy_path=None, cbv_size=None):
    return core.extract(wpix_path, int(global_id), selector, int(mip),
                        int(array_slice), save_npy_path,
                        int(cbv_size) if cbv_size else None)

def t_compare(a, b, save_diff_npy_path=None):
    return core.compare(a, b, save_diff_npy_path)

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
            },
            "required": ["wpix_path"],
        },
        "handler": t_find_events,
    },
    {
        "name": "wpix_describe_dispatch",
        "description": "For a compute Dispatch identified by its global id, return its thread-"
                       "group counts and root bindings: CBVs, SRV inputs, and UAV outputs, each "
                       "resolved to a resource debug-name, view format, and dimensions. Use the "
                       "returned srv/uav/cbv ordering as selectors for wpix_extract/wpix_compare.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string"},
                "global_id": {"type": "integer", "description": "Global id of the Dispatch (from wpix_find_events)."},
            },
            "required": ["wpix_path", "global_id"],
        },
        "handler": t_describe_dispatch,
    },
    {
        "name": "wpix_extract",
        "description": "Decode one binding of a dispatch to numpy and return per-channel stats "
                       "(min/max/mean/has_nan); optionally save the array as .npy. SRV/CBV inputs "
                       "are read as the resource state before the dispatch; UAV outputs are read "
                       "as the state after it (matched across captures by debug-name). 'selector' "
                       "is one of {\"srv\":i}, {\"uav\":i}, {\"cbv\":i}, {\"slot\":n}, "
                       "{\"name\":\"EnvMapBak\"}. For textures choose mip/array_slice (cube face = "
                       "array_slice 0..5). Block-compressed inputs (e.g. BC6H) are reported but "
                       "not decoded (no decoder available).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wpix_path": {"type": "string"},
                "global_id": {"type": "integer"},
                "selector": {"type": "object", "description": "One of {srv:i}|{uav:i}|{cbv:i}|{slot:n}|{name:str}."},
                "mip": {"type": "integer", "default": 0},
                "array_slice": {"type": "integer", "default": 0, "description": "Array slice / cube face (0..5)."},
                "save_npy_path": {"type": "string", "description": "Optional path to save the decoded array as .npy."},
                "cbv_size": {"type": "integer", "description": "Bytes to read for a CBV selector (default 1024)."},
            },
            "required": ["wpix_path", "global_id", "selector"],
        },
        "handler": t_extract,
    },
    {
        "name": "wpix_compare",
        "description": "Decode the same binding from two captures (or two dispatches) and report "
                       "pixel-level difference: max_abs_diff, mean_abs_diff, mse, psnr_db, "
                       "num_differing_texels, per-channel max. Each side is {wpix_path, global_id, "
                       "selector, mip?, array_slice?}. Optionally save the signed diff as .npy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "object", "description": "{wpix_path, global_id, selector, mip?, array_slice?}"},
                "b": {"type": "object", "description": "{wpix_path, global_id, selector, mip?, array_slice?}"},
                "save_diff_npy_path": {"type": "string"},
            },
            "required": ["a", "b"],
        },
        "handler": t_compare,
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
