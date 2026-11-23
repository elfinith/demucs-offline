#requires -Version 5.1
<#
.SYNOPSIS
    Собрать переносимый комплект для офлайн-работы (без .venv) и, при желании, zip.

.DESCRIPTION
    Копирует в папку назначения всё, что нужно на другой машине:
      src\, models\, wheels\, *.ps1, *.cmd, README.md, LICENSES.md
    Исключает .venv (его пересоздают на месте через setup_offline.cmd),
    __pycache__ и результаты обработки.

.PARAMETER OutDir
    Куда собрать. По умолчанию <корень>\dist\demucs-offline

.PARAMETER Zip
    Дополнительно упаковать в demucs-offline.zip рядом с папкой.

.EXAMPLE
    .\pack_bundle.ps1 -Zip
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [switch]$Zip
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'dist\demucs-offline' }

function Write-Step($t) { Write-Host "==> $t" -ForegroundColor Cyan }

if (Test-Path $OutDir) {
    Write-Step "очищаю $OutDir"
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Step "копирую файлы комплекта"
foreach ($item in @('src', 'models', 'wheels')) {
    $srcPath = Join-Path $root $item
    if (-not (Test-Path $srcPath)) { throw "нет обязательной папки: $srcPath" }
    Copy-Item -LiteralPath $srcPath -Destination (Join-Path $OutDir $item) -Recurse -Force
}
foreach ($pattern in @('*.ps1', '*.cmd', 'README.md', 'LICENSES.md')) {
    Get-ChildItem -Path $root -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $OutDir -Force
    }
}

# чистим мусор
Get-ChildItem -Path $OutDir -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force
Get-ChildItem -Path $OutDir -Recurse -File -Include '*.pyc', '*.part' -ErrorAction SilentlyContinue |
    Remove-Item -Force

$files = Get-ChildItem -Path $OutDir -Recurse -File
$sizeMb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "    файлов: $($files.Count), объём: $sizeMb МБ"

Write-Step "проверяю обязательные элементы"
$required = @(
    'src\make_backing.py', 'src\offline_check.py', 'src\backing_lib.py',
    'models\htdemucs_6s.yaml', 'setup_offline.cmd', 'make_backing.cmd', 'README.md'
)
$missing = @()
foreach ($rel in $required) {
    if (-not (Test-Path (Join-Path $OutDir $rel))) { $missing += $rel }
}
$weights = @(Get-ChildItem (Join-Path $OutDir 'models') -Filter *.th -File -ErrorAction SilentlyContinue)
$wheels = @(Get-ChildItem (Join-Path $OutDir 'wheels') -Filter *.whl -File -ErrorAction SilentlyContinue)
if ($weights.Count -eq 0) { $missing += 'models\*.th' }
if ($wheels.Count -eq 0) { $missing += 'wheels\*.whl' }
if ($missing.Count -gt 0) { throw "в комплекте не хватает: $($missing -join ', ')" }
Write-Host "    весов: $($weights.Count), wheel-файлов: $($wheels.Count)"

if ($Zip) {
    $zipPath = "$OutDir.zip"
    if (Test-Path $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Write-Step "упаковываю $zipPath"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $OutDir, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $zipMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    $hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash
    Write-Host "    zip: $zipMb МБ"
    Write-Host "    sha256: $hash"
}

Write-Host ""
Write-Host "Готово: $OutDir" -ForegroundColor Green
Write-Host "На целевой машине (Windows x64, Python 3.12):"
Write-Host "    1) setup_offline.cmd        (создаст .venv из wheels, без интернета)"
Write-Host "    2) make_backing.cmd `"D:\music\song.mp3`""
