# wpix-mcp

A self-contained **MCP server** for inspecting and **comparing shader inputs/outputs**
between two PIX `.wpix` GPU captures, at the pixel level — without the PIX GUI.

It was built to verify the Unity-replicated `NativeRtxptFeature` passes against the
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
| `wpix_find_events` | list Dispatch/Draw events (filter by marker name, e.g. `ProcSkyBaseBake`) |
| `wpix_describe_dispatch` | thread-group counts + CBV/SRV(input)/UAV(output) bindings |
| `wpix_extract` | decode one binding → per-channel stats (+ optional `.npy`) |
| `wpix_compare` | diff one binding across two captures → max-abs/mse/psnr/#differing texels |
| `wpix_clear_cache` | delete cached exports |

### Selectors

`wpix_extract` / `wpix_compare` pick a binding with one of:
`{"srv": i}`, `{"uav": i}`, `{"cbv": i}`, `{"slot": n}`, `{"name": "EnvMapBak"}`.
For textures, choose `mip` and `array_slice` (cube face = `array_slice` 0..5).

## Example: compare the ProcSkyBaseBake output cube (Unity vs Rtxpt)

```jsonc
// 1. find the dispatch in each capture
wpix_find_events { "wpix_path": "...\\Unity.wpix", "name_filter": "ProcSkyBaseBake" }   // -> global_id 17
wpix_find_events { "wpix_path": "...\\Rtxpt.wpix", "name_filter": "ProcSkyBaseBake" }   // -> global_id ?

// 2. see the bindings
wpix_describe_dispatch { "wpix_path": "...\\Unity.wpix", "global_id": 17 }

// 3. diff the baked cube, +X face (array_slice 0), mip 0
wpix_compare {
  "a": { "wpix_path": "...\\Unity.wpix", "global_id": 17, "selector": {"uav":0}, "array_slice": 0 },
  "b": { "wpix_path": "...\\Rtxpt.wpix", "global_id": 42, "selector": {"uav":0}, "array_slice": 0 },
  "save_diff_npy_path": "...\\diff_face0.npy"
}
```

## CLI use (without MCP)

`wpix_core.py` is a plain library; you can script it directly:

```python
import wpix_core as w
bt = w.describe_dispatch(r"...\Unity.wpix", 17)
info = w.extract(r"...\Unity.wpix", 17, {"uav": 0}, mip=0, array_slice=0, out_npy="cube.npy")
diff = w.compare(a_spec, b_spec)
```

## Format support

Decoded to float numpy: `R32G32B32A32_FLOAT`, `R16G16B16A16_FLOAT/UNORM`, `R32G32_FLOAT`,
`R32_FLOAT`, `R16_FLOAT`, `R8G8B8A8_UNORM(_SRGB)`, `B8G8R8A8_UNORM`, `R11G11B10_FLOAT`,
`R8_UNORM`. Block-compressed inputs (BC6H etc.) are reported (name/format/size) but not
decoded — no BC decoder is bundled (no `texconv` present). The RTXPT bake **outputs** are
uncompressed `R16G16B16A16_FLOAT`, which decode fully.
