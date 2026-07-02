# Precisely measures the GPU execution time of the PathTrace (FILL) DispatchRays in
# all four captures. No elevation needed (timing counters, not the NVIDIA HF plugin).
#
# Each run is an independent GPU replay via `pixtool save-event-list --counters`, so
# multiple runs give independent hardware samples (driver/thermal jitter). Per event
# PIX reports three timing views:
#   exec       = EOP Start Time - Execution Start Time  (time the event actually executed;
#                the most precise per-event measure)
#   eop_to_eop = gap from previous event's EOP to this EOP (adds pre-idle, if any)
#   top_to_eop = from entering top-of-pipe to EOP (absorbs in-flight drain of prior work;
#                inflates small events queued behind big ones - see DenoiseSpecHitT case)
#
# Usage:  .\tools\measure_pathtrace.ps1 [-Runs 3]

param([int]$Runs = 3)

$pix = (Get-ChildItem 'C:\Program Files\Microsoft PIX\*\pixtool.exe' |
        Sort-Object FullName -Descending | Select-Object -First 1).FullName
if (-not $pix) { throw 'pixtool.exe not found' }

$captures = [ordered]@{
    'Rtxpt (SER on)'   = 'F:\UnityPathTracing\Other\Rtxpt.wpix'
    'Unity (SER on)'   = 'F:\UnityPathTracing\Other\Unity.wpix'
    'Rtxpt (SER off)'  = 'F:\UnityPathTracing\Other\Rtxpt-NoSER.wpix'
    'Unity (SER off)'  = 'F:\UnityPathTracing\Other\Unity-NoSER.wpix'
}

function Get-PathTraceTimings([string]$wpix, [string]$csv) {
    & $pix open-capture $wpix save-event-list $csv --counters=*Duration* --counters=*Time* |
        Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pixtool failed on $wpix" }

    $rows = Import-Csv $csv

    # queue-id -> row, for climbing the Parent chain to the nearest named marker
    $qmap = @{}
    foreach ($r in $rows) { $qmap[$r.'Queue ID'] = $r }

    foreach ($r in $rows) {
        if ($r.Name.Trim() -ne 'DispatchRays') { continue }

        # Nearest enclosing marker with a name.
        $p = $r.Parent
        $marker = ''
        for ($i = 0; $i -lt 64 -and $p -and $qmap.ContainsKey($p); $i++) {
            $par = $qmap[$p]
            if ($par.Name.Trim()) {
                $marker = $par.Name.Trim()
                break
            }
            $p = $par.Parent
        }

        if (-not $marker.EndsWith('PathTrace')) { continue }   # excludes ...PathTracePrePass

        $num = { param($s) [double]($s -replace '[^\d.]', '') }
        return [pscustomobject]@{
            exec_ms       = ((& $num $r.'EOP Start Time (ns)') - (& $num $r.'Execution Start Time (ns)')) / 1e6
            eop_to_eop_ms = (& $num $r.'EOP to EOP Duration (ns)') / 1e6
            top_to_eop_ms = (& $num $r.'TOP to EOP Duration (ns)') / 1e6
        }
    }

    throw "no PathTrace DispatchRays found in $wpix"
}

$tmp = Join-Path $env:TEMP 'pathtrace_timing'
New-Item -ItemType Directory -Force $tmp | Out-Null
$summary = @()

foreach ($cap in $captures.GetEnumerator()) {
    Write-Host "== $($cap.Key) ==" -ForegroundColor Cyan
    $samples = @()

    for ($i = 1; $i -le $Runs; $i++) {
        $csv = Join-Path $tmp "run_$i.csv"
        Remove-Item $csv -ErrorAction SilentlyContinue      # force a fresh GPU replay

        $t = Get-PathTraceTimings $cap.Value $csv
        $samples += $t

        Write-Host ("   run {0}:  exec {1,8:F3} ms   eop-to-eop {2,8:F3} ms   top-to-eop {3,8:F3} ms" -f `
            $i, $t.exec_ms, $t.eop_to_eop_ms, $t.top_to_eop_ms)
    }

    $mean = @{}
    foreach ($k in 'exec_ms','eop_to_eop_ms','top_to_eop_ms') {
        $vals = $samples | ForEach-Object { $_.$k }
        $mean[$k] = ($vals | Measure-Object -Average).Average
    }

    $summary += [pscustomobject]@{
        capture       = $cap.Key
        exec_ms       = [math]::Round($mean.exec_ms, 3)
        eop_to_eop_ms = [math]::Round($mean.eop_to_eop_ms, 3)
        top_to_eop_ms = [math]::Round($mean.top_to_eop_ms, 3)
    }
}

Write-Host "`n== PathTrace (FILL) DispatchRays: mean of $Runs runs ==" -ForegroundColor Green
$summary | Format-Table -AutoSize
