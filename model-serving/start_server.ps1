<#
.SYNOPSIS
    Start the Model Serving API with .env loading and PYTHONPATH setup.
.DESCRIPTION
    Loads .env, sets PYTHONPATH to src/, prints config summary, runs uvicorn with reload.
    Supports any model registered in model_profiles (Gemma 4, Mistral, etc.).
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
    Write-Host "`u{1F680} Starting Model Server" -ForegroundColor Cyan
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
    $quantize    = if ($env:MODEL_QUANTIZE_4BIT -eq "1") { "ENABLED" } else { "DISABLED" }
    $compile     = if ($env:MODEL_TORCH_COMPILE -eq "0") { "DISABLED" } else { "ENABLED" }
    $memoryOpt   = if ($env:MODEL_MEMORY_OPT    -eq "0") { "DISABLED" } else { "ENABLED" }

    Write-Host "   Quantization : $quantize"
    Write-Host "   Torch Compile: $compile"
    Write-Host "   Memory Opt   : $memoryOpt"

    if ($env:MODEL_QUANTIZE_4BIT -eq "1") {
        Write-Host ""
        Write-Host "   ╔══════════════════════════════════════════════════════════╗" -ForegroundColor Red
        Write-Host "   ║  ⚠️  WARNING: 4-BIT QUANTIZATION IS ENABLED              ║" -ForegroundColor Red
        Write-Host "   ║                                                          ║" -ForegroundColor Red
        Write-Host "   ║  Image/multimodal understanding will NOT work.           ║" -ForegroundColor Red
        Write-Host "   ║  NF4 quantization destroys the vision tower.             ║" -ForegroundColor Red
        Write-Host "   ║  The model will hallucinate or say 'provide an image'.   ║" -ForegroundColor Red
        Write-Host "   ║                                                          ║" -ForegroundColor Red
        Write-Host "   ║  Set MODEL_QUANTIZE_4BIT=0 in .env for image support.   ║" -ForegroundColor Red
        Write-Host "   ╚══════════════════════════════════════════════════════════╝" -ForegroundColor Red
        Write-Host ""
    }

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
        "model_serving.app:app",
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
