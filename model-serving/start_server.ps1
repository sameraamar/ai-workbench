<#
.SYNOPSIS
    Start the Gemma 4 Model Server with .env loading and PYTHONPATH setup.
.DESCRIPTION
    Loads .env, sets PYTHONPATH to src/, prints config summary, runs uvicorn with reload.
#>
param(
    [string]$Host_ = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    Write-Host "`u{1F680} Starting Gemma 4 Model Server" -ForegroundColor Cyan
    Write-Host ("=" * 50)

    # --- Load .env --------------------------------------------------------
    $envFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#")) {
                $parts = $line -split "=", 2
                if ($parts.Count -eq 2) {
                    [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
                }
            }
        }
        Write-Host "`u{2705} Loaded .env configuration" -ForegroundColor Green
    } else {
        Write-Host "`u{26A0}`u{FE0F}  No .env file found - using defaults" -ForegroundColor Yellow
    }

    # --- Print config summary ---------------------------------------------
    $quantize    = if ($env:GEMMA_QUANTIZE_4BIT -eq "1") { "ENABLED" } else { "DISABLED" }
    $compile     = if ($env:GEMMA_TORCH_COMPILE -eq "0") { "DISABLED" } else { "ENABLED" }
    $memoryOpt   = if ($env:GEMMA_MEMORY_OPT    -eq "0") { "DISABLED" } else { "ENABLED" }

    Write-Host "   Quantization : $quantize"
    Write-Host "   Torch Compile: $compile"
    Write-Host "   Memory Opt   : $memoryOpt"

    # --- Set PYTHONPATH ----------------------------------------------------
    $srcPath = Join-Path $PSScriptRoot "src"
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $srcPath
    }
    Write-Host "   Python path  : $srcPath"

    Write-Host ""
    Write-Host "`u{1F310} Starting FastAPI server on http://${Host_}:${Port}" -ForegroundColor Cyan
    Write-Host "   Press Ctrl+C to stop the server"
    Write-Host ("=" * 50)

    # --- Launch uvicorn ----------------------------------------------------
    $uvicornArgs = @(
        "-m", "uvicorn",
        "gemma_serving.app:app",
        "--host", $Host_,
        "--port", $Port
    )
    if (-not $NoReload) {
        $uvicornArgs += "--reload"
        $uvicornArgs += "--reload-dir"
        $uvicornArgs += $srcPath
    }

    & python @uvicornArgs
}
finally {
    Pop-Location
}
