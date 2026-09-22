<#
.SYNOPSIS
    Start the vLLM model server via WSL2.
.DESCRIPTION
    Reads configuration from .env.vllm, then launches start.sh in WSL2.
    The server exposes an OpenAI-compatible API at http://localhost:<port>.
.PARAMETER Model
    Override the MODEL_ID from .env.vllm. Pass a HuggingFace model ID.
.PARAMETER Port
    Override the VLLM_PORT from .env.vllm (default: 8000).
.PARAMETER Distribution
    WSL distribution that contains the vLLM environment.
.EXAMPLE
    .\start_vllm.ps1
    .\start_vllm.ps1 -Distribution "Ubuntu-22.04"
    .\start_vllm.ps1 -Model "google/gemma-4-E4B-it"
    .\start_vllm.ps1 -Model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" -Port 8001
#>
param(
    [string]$Model,
    [int]$Port,
    [string]$Distribution = "Ubuntu-22.04"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot

# --- Check the requested WSL2 distribution is available --------------------
try {
    $null = wsl --status 2>&1
} catch {
    Write-Error "WSL2 is required but not found. Install it with: wsl --install"
    exit 1
}

$installedDistributions = @(
    wsl --list --quiet 2>$null |
        ForEach-Object { $_.Replace([string][char]0, "").Trim() } |
        Where-Object { $_ }
)
if ($Distribution -notin $installedDistributions) {
    Write-Error @"
WSL distribution '$Distribution' is not installed.
The 'docker-desktop' distribution is internal to Docker Desktop and cannot be used for this launcher.
Install Ubuntu with: wsl --install -d $Distribution
Or use Docker Desktop directly with: .\start-docker.ps1
"@
    exit 1
}

# --- Build environment overrides for WSL2 -----------------------------------
$envOverrides = @()
if ($Model) {
    $envOverrides += "MODEL_ID=$Model"
    Write-Host "Model override: $Model" -ForegroundColor Cyan
}
if ($Port) {
    $envOverrides += "VLLM_PORT=$Port"
    Write-Host "Port override:  $Port" -ForegroundColor Cyan
}

# Convert Windows path to WSL path.
# Done manually to avoid wslpath quoting issues in PowerShell.
$driveLetter = $ScriptDir.Substring(0, 1).ToLower()
$wslScriptDir = "/mnt/$driveLetter" + ($ScriptDir.Substring(2) -replace '\\', '/')
Write-Host "WSL path: $wslScriptDir" -ForegroundColor DarkGray

# --- Launch in WSL2 ---------------------------------------------------------
$envPrefix = if ($envOverrides.Count -gt 0) {
    ($envOverrides -join " ") + " "
} else {
    ""
}

Write-Host ""
Write-Host "Starting vLLM server via WSL2..." -ForegroundColor Cyan
Write-Host "Script: $wslScriptDir/start.sh" -ForegroundColor DarkGray
Write-Host ""

# Make start.sh executable, strip Windows CRLF line endings, and run it
wsl -d $Distribution -- bash -c "chmod +x '$wslScriptDir/start.sh' && cd '$wslScriptDir' && sed -i 's/\r$//' start.sh setup_vllm.sh 2>/dev/null; ${envPrefix}./start.sh"
