<#
.SYNOPSIS
    Start the vLLM OpenAI-compatible server with Docker Desktop.
.DESCRIPTION
    Reads .env.vllm, maps Windows cache/media directories into the Linux
    container, and starts the official vLLM image with NVIDIA GPU access.
.PARAMETER Model
    Override MODEL_ID from .env.vllm.
.PARAMETER Port
    Override VLLM_PORT from .env.vllm.
.PARAMETER SharedMediaDir
    Override the Windows host directory used for file:// media.
.PARAMETER DryRun
    Validate prerequisites and print the Docker command without starting it.
.EXAMPLE
    .\start-docker.ps1 -Model "Qwen/Qwen2.5-0.5B-Instruct"
.EXAMPLE
    .\start-docker.ps1 -DryRun
#>
param(
    [string]$Model,
    [int]$Port,
    [string]$SharedMediaDir,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        $value = $parts[1].Trim()
        if ($value.Length -ge 2 -and (
            ($value.StartsWith("'") -and $value.EndsWith("'")) -or
            ($value.StartsWith('"') -and $value.EndsWith('"'))
        )) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $values[$parts[0].Trim()] = $value
    }

    return $values
}

function Get-Setting {
    param(
        [hashtable]$Values,
        [string]$Name,
        [string]$Default
    )

    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($processValue)) {
        return $processValue
    }
    if ($Values.ContainsKey($Name) -and -not [string]::IsNullOrWhiteSpace($Values[$Name])) {
        return $Values[$Name]
    }
    return $Default
}

function Convert-WslPathToWindows {
    param([string]$Path)

    if ($Path -notmatch '^/mnt/([a-zA-Z])(?:/(.*))?$') {
        return $null
    }

    $drive = $Matches[1].ToUpper()
    $remainder = $Matches[2] -replace '/', '\'
    return "${drive}:\$remainder"
}

function Convert-WindowsPathToWsl {
    param([string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if ($fullPath -notmatch '^([a-zA-Z]):\\(.*)$') {
        throw "Expected a drive-qualified Windows path, got: $Path"
    }

    $drive = $Matches[1].ToLower()
    $remainder = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$remainder"
}

$scriptDir = $PSScriptRoot
$settings = Read-DotEnv (Join-Path $scriptDir ".env.vllm")

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install and start Docker Desktop first."
}

$dockerOsOutput = try {
    & docker version --format '{{.Server.Os}}' 2>$null
} catch {
    $null
}
if ($LASTEXITCODE -ne 0 -or -not $dockerOsOutput) {
    throw "Docker Desktop is not running. Start it and wait for the engine to become ready."
}
$dockerOs = "$dockerOsOutput".Trim()
if ($dockerOs -ne "linux") {
    throw "Docker Desktop must be using Linux containers; current engine OS: $dockerOs"
}

$dockerRuntimes = & docker info --format '{{json .Runtimes}}' 2>$null
if ($LASTEXITCODE -ne 0 -or $dockerRuntimes -notmatch 'nvidia') {
    throw "Docker does not expose the NVIDIA runtime. Enable WSL 2 GPU support in Docker Desktop."
}

$modelId = if ($Model) { $Model } else { Get-Setting $settings "MODEL_ID" "google/gemma-4-E2B-it" }
$serverPort = if ($Port) { $Port } else { [int](Get-Setting $settings "VLLM_PORT" "8000") }
$maxModelLen = Get-Setting $settings "MAX_MODEL_LEN" "8192"
$gpuMemoryUtilization = Get-Setting $settings "GPU_MEMORY_UTILIZATION" "0.90"
$dtype = Get-Setting $settings "DTYPE" "bfloat16"
$quantization = Get-Setting $settings "QUANTIZATION" "none"
$tensorParallelSize = Get-Setting $settings "TENSOR_PARALLEL_SIZE" "1"
$kvCacheDtype = Get-Setting $settings "KV_CACHE_DTYPE" "auto"
$limitMmPerPrompt = Get-Setting $settings "LIMIT_MM_PER_PROMPT" '{"image": 24, "video": 1}'
$image = Get-Setting $settings "VLLM_IMAGE" "vllm/vllm-openai:latest"
$hfToken = Get-Setting $settings "HF_TOKEN" ""

if (-not $DryRun) {
    $portListener = Get-NetTCPConnection -LocalPort $serverPort -State Listen -ErrorAction SilentlyContinue
    if ($portListener) {
        throw "Port $serverPort is already in use. Stop that process or pass -Port with a free port."
    }
}

$hfCache = if ($env:HF_HOME) {
    [System.IO.Path]::GetFullPath($env:HF_HOME)
} else {
    Join-Path $HOME ".cache\huggingface"
}
New-Item -ItemType Directory -Path $hfCache -Force | Out-Null

$configuredMediaPath = Get-Setting $settings "SHARED_MEDIA_DIR" ""
if ($SharedMediaDir) {
    $mediaHostPath = [System.IO.Path]::GetFullPath($SharedMediaDir)
    $mediaContainerPath = Convert-WindowsPathToWsl $mediaHostPath
} elseif ($configuredMediaPath) {
    $mediaHostPath = Convert-WslPathToWindows $configuredMediaPath
    if (-not $mediaHostPath) {
        throw "SHARED_MEDIA_DIR must use a /mnt/<drive>/... WSL path for Docker Desktop: $configuredMediaPath"
    }
    $mediaContainerPath = $configuredMediaPath
} else {
    $mediaHostPath = Join-Path (Split-Path $scriptDir -Parent) "shared-media"
    $mediaContainerPath = Convert-WindowsPathToWsl $mediaHostPath
}
New-Item -ItemType Directory -Path $mediaHostPath -Force | Out-Null

$modelLower = $modelId.ToLowerInvariant()
if ($modelLower.Contains("awq")) {
    $dtype = "float16"
    if ($quantization -eq "none") {
        $quantization = "awq"
    }
}

$dockerArgs = @(
    "run", "--rm",
    "--gpus", "all",
    "--ipc", "host",
    "-p", "${serverPort}:${serverPort}",
    "-v", "${hfCache}:/root/.cache/huggingface",
    "-v", "${mediaHostPath}:${mediaContainerPath}:ro",
    "--name", "vllm-server"
)

if ($hfToken) {
    $dockerArgs += @("-e", "HUGGING_FACE_HUB_TOKEN=$hfToken")
}

$dockerArgs += @(
    $image,
    "--model", $modelId,
    "--host", "0.0.0.0",
    "--port", "$serverPort",
    "--max-model-len", $maxModelLen,
    "--gpu-memory-utilization", $gpuMemoryUtilization,
    "--dtype", $dtype,
    "--tensor-parallel-size", $tensorParallelSize,
    "--limit-mm-per-prompt", $limitMmPerPrompt,
    "--allowed-local-media-path", $mediaContainerPath
)

if ($quantization -ne "none") {
    $dockerArgs += @("--quantization", $quantization)
}
if ($kvCacheDtype -ne "auto" -and $kvCacheDtype) {
    $dockerArgs += @("--kv-cache-dtype", $kvCacheDtype)
}
if ($modelLower.Contains("mistral")) {
    if ($modelLower.Contains("awq")) {
        $dockerArgs += @("--tokenizer_mode", "mistral")
    } else {
        $dockerArgs += @(
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral"
        )
    }
}

Write-Host ""
Write-Host "vLLM Model Server (Docker Desktop)" -ForegroundColor Cyan
Write-Host "  Image:       $image"
Write-Host "  Model:       $modelId"
Write-Host "  API:         http://localhost:$serverPort"
Write-Host "  HF cache:    $hfCache"
Write-Host "  Media host:  $mediaHostPath"
Write-Host "  Media mount: $mediaContainerPath"
Write-Host ""

if ($DryRun) {
    $displayArgs = $dockerArgs | ForEach-Object {
        if ($_ -like "HUGGING_FACE_HUB_TOKEN=*") { "HUGGING_FACE_HUB_TOKEN=<redacted>" } else { $_ }
    }
    Write-Host "Dry run; Docker command:" -ForegroundColor Yellow
    Write-Host ("docker " + ($displayArgs -join " "))
    exit 0
}

Write-Host "First launch downloads the vLLM image and model weights. Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""
& docker @dockerArgs
exit $LASTEXITCODE