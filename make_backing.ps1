#requires -Version 5.1
<#
.SYNOPSIS
    Минусовка (backing track) без гитары из произвольного аудиофайла — офлайн.

.DESCRIPTION
    Все аргументы передаются скрипту src\make_backing.py без изменений,
    поэтому работают любые его опции.

.EXAMPLE
    .\make_backing.ps1 "D:\music\song.mp3"
    .\make_backing.ps1 "D:\music\*.mp3" -o D:\backing
    .\make_backing.ps1 song.mp3 --stems --keep-sr --verify
    .\make_backing.ps1 song.mp3 --max-seconds 30
    .\make_backing.ps1 --help
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$script = Join-Path $root 'src\make_backing.py'

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path $script)) { Write-Host "ОШИБКА: нет $script" -ForegroundColor Red; exit 1 }

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $python = $cmd.Source
        Write-Host "ВНИМАНИЕ: локального .venv нет, использую системный python ($python)." -ForegroundColor Yellow
        Write-Host "          Если пакетов не хватает — запустите .\setup_offline.ps1" -ForegroundColor Yellow
    } else {
        Write-Host "ОШИБКА: не найдено окружение. Запустите .\setup_offline.ps1" -ForegroundColor Red
        exit 1
    }
}

& $python -u $script @args
exit $LASTEXITCODE
