#requires -Version 5.1
<#
.SYNOPSIS
    Создать локальное окружение (.venv) для офлайн-комплекта — без интернета.

.DESCRIPTION
    Ставит python-пакеты из папки wheels строго офлайн (--no-index), затем
    запускает самопроверку. Интернет не используется ни на одном шаге.

.PARAMETER Python
    Путь к python.exe 3.12 x64. По умолчанию ищется py -3.12, затем python.

.PARAMETER VenvDir
    Куда создать окружение. По умолчанию <корень комплекта>\.venv

.PARAMETER Force
    Удалить существующее окружение и создать заново.

.PARAMETER SkipCheck
    Не запускать offline_check.py после установки.
#>
[CmdletBinding()]
param(
    [string]$Python,
    [string]$VenvDir,
    [switch]$Force,
    [switch]$SkipCheck
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
if (-not $VenvDir) { $VenvDir = Join-Path $root '.venv' }
$wheels = Join-Path $root 'wheels'
$src = Join-Path $root 'src'

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "ОШИБКА: $text" -ForegroundColor Red; exit 1 }

Write-Step "корень комплекта: $root"

if (-not (Test-Path $wheels)) { Fail "нет папки wheels (ожидалась $wheels)" }
$whl = @(Get-ChildItem -Path $wheels -Filter *.whl -ErrorAction SilentlyContinue)
if ($whl.Count -eq 0) { Fail "в $wheels нет ни одного .whl" }
Write-Host "    wheel-файлов: $($whl.Count)"

# --- поиск python 3.12 x64
if (-not $Python) {
    $candidates = @()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $found = & py -3.12 -c "import sys,platform;print(sys.executable);print('%d.%d'%sys.version_info[:2]);print(platform.machine())" 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $Python = $found[0] }
        } catch { }
    }
    if (-not $Python) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source }
    }
}
if (-not $Python) {
    Fail @"
Python 3.12 x64 не найден. Варианты:
  1) установить Python 3.12 (x64) на эту машину (можно с того же носителя);
  2) указать путь явно:  .\setup_offline.ps1 -Python "C:\Python312\python.exe"
  3) перенести уже готовую папку .venv с машины, где она создана (тот же Python 3.12 x64).
"@
}
if (-not (Test-Path $Python)) { Fail "не найден python: $Python" }

$info = & $Python -c "import sys,platform;print(sys.version.split()[0]);print(platform.machine())"
$pyVer = $info[0]; $pyArch = $info[1]
Write-Host "    python: $Python ($pyVer, $pyArch)"
if (-not $pyVer.StartsWith('3.12')) { Fail "нужен Python 3.12.x, найден $pyVer" }
if ($pyArch -notin @('AMD64', 'x86_64')) { Fail "нужен 64-битный Python, найден $pyArch" }

# --- создание venv
if (Test-Path $VenvDir) {
    if (-not $Force) {
        Write-Host "    окружение уже есть: $VenvDir (ключ -Force пересоздаст)" -ForegroundColor Yellow
    } else {
        Write-Step "удаляю старое окружение"
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}
if (-not (Test-Path (Join-Path $VenvDir 'Scripts\python.exe'))) {
    Write-Step "создаю venv: $VenvDir"
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "не удалось создать venv" }
}

$vpy = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $vpy)) { Fail "в venv нет python.exe: $vpy" }

# --- установка строго из локальных wheel-файлов
Write-Step "устанавливаю пакеты из wheels (--no-index, без сети)"
& $vpy -m pip install --no-index --find-links "$wheels" --disable-pip-version-check --no-warn-script-location demucs soundfile lameenc
if ($LASTEXITCODE -ne 0) { Fail "pip не смог поставить пакеты из $wheels" }

Write-Step "версии установленного"
& $vpy -c "import torch, demucs, soundfile, lameenc, numpy; print('torch', torch.__version__); print('demucs', demucs.__version__); print('soundfile', soundfile.__version__); print('numpy', numpy.__version__)"

if (-not $SkipCheck) {
    Write-Step "самопроверка (offline_check.py)"
    & $vpy (Join-Path $src 'offline_check.py')
    $code = $LASTEXITCODE
    if ($code -ne 0) { Fail "самопроверка не пройдена (код $code)" }
}

Write-Host ""
Write-Host "Готово. Запуск минусовки:" -ForegroundColor Green
Write-Host "    .\make_backing.ps1 `"D:\music\song.mp3`""
