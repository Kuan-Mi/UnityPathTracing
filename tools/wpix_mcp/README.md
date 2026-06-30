# wpix-mcp

A self-contained **MCP server** for inspecting and **comparing shader inputs/outputs**
between two PIX `.wpix` GPU captures, at the pixel level — without the PIX GUI.

It was built to verify the Unity-replicated `RtxptFeature` passes against the
original C++ Rtxpt by diffing the exact GPU resources a shader reads and writes.

## How it works

`pixtool.exe` (shipped with PIX) can't dump a compute shader's UAV/SRV pixels directly
(`save-resource` only handles render targets). So for a Dispatch at global id `N`:

1. `recapture-region [N,N]` → `export-to-cpp` produces a C++ replay whose `resources.bin`
   holds the exact bytes of every resource **as of the start of N** → the dispatch's
   **inputs** (SRVs, CBV).
2. For **outputs** (UAVs), it exports `[M,M]` where `M` is the next event, capturing the
   resource state **after** N ran. The output resource is matched across the two exports
   by its debug-name + descriptor (api object-ids are renumbered per recapture).
3. The wanted resource blob is XPRESS-decompressed (`Cabinet.dll`) and decoded from its
   DXGI format into a numpy array. Descriptor-table sizes come from the parsed root
   signature, so bindings resolve exactly.

Exports are cached under `%TEMP%\wpix_mcp_cache` (or `$WPIX_CACHE`); each can be 200 MB+,
so call `wpix_clear_cache` when done.

## Requirements

- Windows + Microsoft PIX (auto-discovered at `C:\Program Files\Microsoft PIX\*\pixtool.exe`;
  override with the `WPIX_PIXTOOL` env var).
- Python 3.10+ with `numpy` (already installed here). No other packages — the MCP protocol
  is implemented directly over stdio, so `pip install mcp` is **not** needed.

## Register with Claude Code

Add to `.mcp.json` at the repo root (a ready copy is in this folder):

```json
{
  "mcpServers": {
    "wpix": {
      "command": "python",
      "args": ["f:\\UnityPathTracing\\tools\\wpix_mcp\\wpix_mcp_server.py"],
      "env": { "WPIX_PIXTOOL": "C:\\Program Files\\Microsoft PIX\\2603.25\\pixtool.exe" }
    }
  }
}
```

Then restart Claude Code (or `/mcp` to reconnect). Tools appear as `wpix_*`.

## Tools

| tool | purpose |
|---|---|
| `wpix_find_events` | list Dispatch/Draw events (filter by marker name; `dispatches_only` to drop barriers) |
| `wpix_describe_dispatch` | thread-group counts + CBV/SRV(input)/UAV(output) bindings — **indices match the selectors below** |
| `wpix_extract` | decode one binding → per-channel stats (+ optional `.npy`) |
| `wpix_compare` | diff one binding across two captures → max-abs/mse/psnr/#differing texels (+ `max_fp16_ulp` for half formats) |
| `wpix_compare_subresources` | diff one binding across **all mips/faces** in one call → per-subresource table + aggregate |
| `wpix_diff_stage` | **one-call stage diff**: find a marked dispatch in both captures, match UAV outputs by name, verdict per output |
| `wpix_clear_cache` | delete cached exports |

### Quick start: diff a whole stage

```jsonc
wpix_diff_stage { "wpix_a": "...\\Rtxpt.wpix", "wpix_b": "...\\Unity.wpix", "marker": "ProcSkyBaseBake" }
// -> outputs:[{ output:"EnvMapBakerMainCube", verdict:"identical", ... }]

// For a mip-gen chain, point at the last dispatch and compare the full pyramid:
wpix_diff_stage { "wpix_a": "...", "wpix_b": "...", "marker": "EnvMapBakerMIPs", "mips": "all" }
```

`mips`: `"base"` (default) compares only mip 0 — correct for a single dispatch, since higher
mips at that event are stale (written by a later pass). `"all"` compares the full pyramid —
use only on the final dispatch of a mip-gen chain.

### Selectors

`wpix_extract` / `wpix_compare` pick a binding with one of:
`{"srv": i}`, `{"uav": i}`, `{"cbv": i}`, `{"slot": n}`, `{"name": "EnvMapBakerMainCube"}`.
The `i` indices match `wpix_describe_dispatch`'s ordering. `{"name": …}` takes the **full**
debug name (matching is tolerant of PIX's 8-char truncation). For textures, choose `mip` and
`array_slice` (cube face = `array_slice` 0..5).

> CBV note: there is no per-view size in the capture, so `wpix_extract` reads a 256-byte
> window by default and never past the buffer. Pass `cbv_size = sizeof(struct)` to avoid
> reading adjacent constants; `had_nan` flags uint/padding fields or an over-read.

## Example: compare the ProcSkyBaseBake output cube (Unity vs Rtxpt)

The one-call way:

```jsonc
wpix_diff_stage { "wpix_a": "...\\Rtxpt.wpix", "wpix_b": "...\\Unity.wpix", "marker": "ProcSkyBaseBake" }
```

Or step by step, when you want a specific face/mip:

```jsonc
// 1. find the dispatch in each capture (dispatches_only skips ResourceBarrier rows)
wpix_find_events { "wpix_path": "...\\Unity.wpix", "name_filter": "ProcSkyBaseBake", "dispatches_only": true }
wpix_find_events { "wpix_path": "...\\Rtxpt.wpix", "name_filter": "ProcSkyBaseBake", "dispatches_only": true }

// 2. see the bindings (uav/srv/cbv indices here match the selectors in step 3)
wpix_describe_dispatch { "wpix_path": "...\\Unity.wpix", "global_id": 15 }

// 3a. all 6 faces at once
wpix_compare_subresources {
  "a": { "wpix_path": "...\\Rtxpt.wpix", "global_id": 18, "selector": {"uav":1} },
  "b": { "wpix_path": "...\\Unity.wpix", "global_id": 15, "selector": {"uav":0} }
}

// 3b. or one face, with a saved diff
wpix_compare {
  "a": { "wpix_path": "...\\Rtxpt.wpix", "global_id": 18, "selector": {"uav":1}, "array_slice": 0 },
  "b": { "wpix_path": "...\\Unity.wpix", "global_id": 15, "selector": {"uav":0}, "array_slice": 0 },
  "save_diff_npy_path": "...\\diff_face0.npy"
}
```

## CLI use (without MCP)

`wpix_core.py` is a plain library; you can script it directly (run from this folder, or add
it to `sys.path`):

```python
import wpix_core as w
# whole-stage verdict
w.diff_stage(r"...\Rtxpt.wpix", r"...\Unity.wpix", "GenIM")
# all mips/faces of one output
w.compare_subresources({"wpix_path": r"...\Rtxpt.wpix", "global_id": 32, "selector": {"uav": 1}},
                       {"wpix_path": r"...\Unity.wpix", "global_id": 29, "selector": {"uav": 0}})
# single binding
bt   = w.describe_dispatch(r"...\Unity.wpix", 15)
info = w.extract(r"...\Unity.wpix", 15, {"uav": 0}, mip=0, array_slice=0, out_npy="cube.npy")
diff = w.compare(a_spec, b_spec)
```

## Format support

Decoded to float numpy: `R32G32B32A32_FLOAT`, `R16G16B16A16_FLOAT/UNORM`, `R32G32_FLOAT`,
`R32_FLOAT`, `R16_FLOAT`, `R8G8B8A8_UNORM(_SRGB)`, `B8G8R8A8_UNORM`, `R11G11B10_FLOAT`,
`R8_UNORM`. Block-compressed inputs (BC6H etc.) are reported (name/format/size) but not
decoded — no BC decoder is bundled (no `texconv` present). The RTXPT bake **outputs** are
uncompressed `R16G16B16A16_FLOAT`, which decode fully.
