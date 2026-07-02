param(
    [string]$Path = "Other\report.txt",
    [int]$Top = 25
)

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Report not found: $Path"
}

$header = Get-Content -LiteralPath $Path -TotalCount 2
$reportedGroups = $null
$reportedDuplicateGroups = $null
$reportedDuplicateObjects = $null
$materialSource = ""

if ($header[0] -match 'Groups:\s*(\d+),\s*duplicate groups:\s*(\d+),\s*objects in duplicate groups:\s*(\d+)') {
    $reportedGroups = [int]$matches[1]
    $reportedDuplicateGroups = [int]$matches[2]
    $reportedDuplicateObjects = [int]$matches[3]
}
if ($header.Count -gt 1 -and $header[1] -match 'Material source:\s*(.+)$') {
    $materialSource = $matches[1].Trim()
}

$groups = New-Object System.Collections.Generic.List[object]
$current = $null

Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_
    if ($line -match '^(\d+)\s+x\s+(.+)$') {
        if ($current) { $groups.Add([pscustomobject]$current) }
        $current = [ordered]@{
            Count = [int]$matches[1]
            Mesh = $matches[2].Trim()
            Materials = ""
            Paths = 0
        }
    }
    elseif ($current -and $line -match '^Materials:\s*(.*)$') {
        $current.Materials = $matches[1].Trim()
    }
    elseif ($current -and $line -match '^\s+\S') {
        $current.Paths++
    }
}
if ($current) { $groups.Add([pscustomobject]$current) }

$listedObjects = ($groups | Measure-Object Count -Sum).Sum
$duplicateGroups = if ($reportedDuplicateGroups -ne $null) { $reportedDuplicateGroups } else { @($groups | Where-Object Count -gt 1).Count }
$duplicateObjects = if ($reportedDuplicateObjects -ne $null) { $reportedDuplicateObjects } else { $listedObjects }
$totalGroups = if ($reportedGroups -ne $null) { $reportedGroups } else { $groups.Count }
$singletonGroups = $totalGroups - $duplicateGroups
$totalObjects = $duplicateObjects + $singletonGroups
$savedEntries = $totalObjects - $totalGroups

Write-Host "== Summary =="
Write-Host ("material_source: {0}" -f $materialSource)
Write-Host ("total_objects: {0}" -f $totalObjects)
Write-Host ("unique_mesh_material_groups: {0}" -f $totalGroups)
Write-Host ("duplicate_groups: {0}" -f $duplicateGroups)
Write-Host ("singleton_groups: {0}" -f $singletonGroups)
Write-Host ("objects_in_duplicate_groups: {0}" -f $duplicateObjects)
Write-Host ("potential_saved_geometry_entries: {0}" -f $savedEntries)
Write-Host ("potential_reduction: {0:N2}%" -f (100.0 * $savedEntries / $totalObjects))
Write-Host ("largest_duplicate_group: {0}" -f (($groups | Measure-Object Count -Maximum).Maximum))

Write-Host ""
Write-Host ("== Top {0} Duplicate Groups ==" -f $Top)
$groups |
    Sort-Object Count -Descending |
    Select-Object -First $Top |
    ForEach-Object {
        Write-Host ("{0,4} x  {1}  |  {2}" -f $_.Count, $_.Mesh, $_.Materials)
    }

Write-Host ""
Write-Host "== Duplicate Group Size Distribution =="
$groups |
    Group-Object Count |
    Sort-Object { [int]$_.Name } |
    ForEach-Object {
        $objects = ($_.Group | Measure-Object Count -Sum).Sum
        Write-Host ("size {0,4}: {1,4} groups, {2,5} objects" -f $_.Name, $_.Count, $objects)
    }
