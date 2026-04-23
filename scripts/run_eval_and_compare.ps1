param(
    [string]$ModelPath = "checkpoints/dqn_traffic_20260223_124456/best_model.pt",
    [string]$EvalConfig = "config/eval_model_windows.yaml",
    [int]$CompareEpisodes = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found at $python"
}

$sumoHome = "C:\Program Files (x86)\Eclipse\Sumo"
$sumoBin = Join-Path $sumoHome "bin"
if (-not (Test-Path $sumoBin)) {
    throw "SUMO bin not found at $sumoBin"
}

$env:SUMO_HOME = $sumoHome
if (-not (($env:Path -split ';') -contains $sumoBin)) {
    $env:Path = "$sumoBin;$env:Path"
}

Write-Host "============================================================"
Write-Host "Running Full Evaluation"
Write-Host "============================================================"
Write-Host "Model: $ModelPath"
Write-Host "Config: $EvalConfig"

& $python .\evaluate.py --config $EvalConfig --model $ModelPath
$evalExit = $LASTEXITCODE
Write-Host "evaluate.py exit code: $evalExit"

Write-Host ""
Write-Host "============================================================"
Write-Host "Running Policy Comparison"
Write-Host "============================================================"

& $python .\test_agent.py --model $ModelPath --episodes $CompareEpisodes
$compareExit = $LASTEXITCODE
Write-Host "test_agent.py (model) exit code: $compareExit"

if ($compareExit -ne 0) {
    Write-Host ""
    Write-Host "Model comparison failed (likely model format mismatch for SB3)."
    Write-Host "Running baselines-only comparison fallback..."
    & $python .\test_agent.py --baselines-only --episodes $CompareEpisodes
    $baselineExit = $LASTEXITCODE
    Write-Host "test_agent.py (baselines-only) exit code: $baselineExit"
    exit $baselineExit
}

exit $compareExit
