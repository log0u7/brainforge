# Setup bench llama-server instances as nssm services on Windows/gpu3080
# Usage: powershell -ExecutionPolicy Bypass -File setup_bench_servers.ps1 [-Apply]

param(
    [switch]$Apply
)

$NSSM = 'C:\Users\eppgr\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe'
$LLAMA = 'C:\llama.cpp\llama-server.exe'
$BASE_MODEL = 'C:\Users\eppgr\.cache\huggingface\hub\models--Qwen--Qwen3-1.7B-GGUF\snapshots\90862c4b9d2787eaed51d12237eafdfe7c5f6077\Qwen3-1.7B-Q8_0.gguf'
$TRAINED_MODEL = 'C:\Users\eppgr\brainforge\models\export-mr6-v2\model-q4_k_m.gguf'
$PYTHON = 'C:\Users\eppgr\brainforge\.venv\Scripts\python.exe'
$BF_EXE = 'C:\Users\eppgr\brainforge\.venv\Scripts\brainforge.exe'
$LOG_DIR = 'C:\llama.cpp\logs'
$COMMON = '-c 16384 -ngl 99 -fa on --parallel 1 --jinja --metrics'

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

function Show-Cmd($cmd) {
    Write-Host $cmd
}

if (-not $Apply) {
    Write-Host "DRY RUN - use -Apply to make changes"
    Write-Host ""
    Write-Host "# Stop/reconfigure existing services"
    Show-Cmd "sc stop bf-train"
    Show-Cmd "sc config bf-train start= demand"
    Show-Cmd "sc stop bf-llama"
    Show-Cmd "sc config bf-llama start= demand"
    Write-Host ""
    Write-Host "# Fix bf-train"
    Show-Cmd "& '$NSSM' set bf-train Application '$PYTHON'"
    Show-Cmd "& '$NSSM' set bf-train AppParameters '-u slice_run.py'"
    Show-Cmd "& '$NSSM' set bf-train AppDirectory 'C:\Users\eppgr\brainforge'"
    Show-Cmd "& '$NSSM' set bf-train AppExit Default Exit"
    Show-Cmd "& '$NSSM' set bf-train Start SERVICE_DEMAND_START"
    Write-Host ""
    Write-Host "# Create bench services"
    Show-Cmd "& '$NSSM' install bf-bench-base '$LLAMA'"
    Show-Cmd "& '$NSSM' set bf-bench-base AppParameters \"-m `"$BASE_MODEL`" --host 0.0.0.0 --port 9932 --alias base-qwen3-1.7b $COMMON\""
    Show-Cmd "& '$NSSM' set bf-bench-base AppDirectory C:\llama.cpp"
    Show-Cmd "& '$NSSM' set bf-bench-base AppExit Default Restart"
    Show-Cmd "& '$NSSM' set bf-bench-base AppStdout '$LOG_DIR\bf-bench-base.log'"
    Show-Cmd "& '$NSSM' set bf-bench-base AppStderr '$LOG_DIR\bf-bench-base-error.log'"
    Show-Cmd "& '$NSSM' set bf-bench-base AppRotateFiles 1; & '$NSSM' set bf-bench-base AppRotateOnline 1; & '$NSSM' set bf-bench-base AppRotateBytes 10485760"
    Show-Cmd "& '$NSSM' set bf-bench-base ObjectName LocalSystem"
    Show-Cmd "& '$NSSM' set bf-bench-base Start SERVICE_DEMAND_START"
    Write-Host ""
    Show-Cmd "& '$NSSM' install bf-bench-trained '$LLAMA'"
    Show-Cmd "& '$NSSM' set bf-bench-trained AppParameters \"-m `"$TRAINED_MODEL`" --host 0.0.0.0 --port 9933 --alias trained-mr6 $COMMON\""
    Show-Cmd "& '$NSSM' set bf-bench-trained AppDirectory C:\llama.cpp"
    Show-Cmd "& '$NSSM' set bf-bench-trained AppExit Default Restart"
    Show-Cmd "& '$NSSM' set bf-bench-trained AppStdout '$LOG_DIR\bf-bench-trained.log'"
    Show-Cmd "& '$NSSM' set bf-bench-trained AppStderr '$LOG_DIR\bf-bench-trained-error.log'"
    Show-Cmd "& '$NSSM' set bf-bench-trained AppRotateFiles 1; & '$NSSM' set bf-bench-trained AppRotateOnline 1; & '$NSSM' set bf-bench-trained AppRotateBytes 10485760"
    Show-Cmd "& '$NSSM' set bf-bench-trained ObjectName LocalSystem"
    Show-Cmd "& '$NSSM' set bf-bench-trained Start SERVICE_DEMAND_START"
    exit 0
}

# Apply changes
Write-Host "Applying changes..."

# Stop problematic services
& sc.exe stop bf-train 2>&1 | Out-Null
& sc.exe config bf-train start= demand 2>&1 | Out-Null
& sc.exe stop bf-llama 2>&1 | Out-Null
& sc.exe config bf-llama start= demand 2>&1 | Out-Null

# Fix bf-train
& $NSSM set bf-train Application $PYTHON 2>&1 | Out-Null
& $NSSM set bf-train AppParameters "-u slice_run.py" 2>&1 | Out-Null
& $NSSM set bf-train AppDirectory "C:\Users\eppgr\brainforge" 2>&1 | Out-Null
& $NSSM set bf-train AppExit Default Exit 2>&1 | Out-Null
& $NSSM set bf-train Start SERVICE_DEMAND_START 2>&1 | Out-Null

function New-BenchService($name, $model, $port, $alias, $logBase) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Host "Service $name exists, removing..."
        & $NSSM stop $name 2>&1 | Out-Null
        & $NSSM remove $name confirm 2>&1 | Out-Null
    }
    Write-Host "Installing $name..."
    & $NSSM install $name $LLAMA
    & $NSSM set $name AppParameters "-m `"$model`" --host 0.0.0.0 --port $port --alias $alias $COMMON"
    & $NSSM set $name AppDirectory C:\llama.cpp
    & $NSSM set $name AppExit Default Restart
    & $NSSM set $name AppStdout "$LOG_DIR\$logBase.log"
    & $NSSM set $name AppStderr "$LOG_DIR\$logBase-error.log"
    & $NSSM set $name AppRotateFiles 1
    & $NSSM set $name AppRotateOnline 1
    & $NSSM set $name AppRotateBytes 10485760
    & $NSSM set $name ObjectName LocalSystem
    & $NSSM set $name Start SERVICE_DEMAND_START
}

New-BenchService "bf-bench-base" $BASE_MODEL 9932 "base-qwen3-1.7b" "bf-bench-base"
New-BenchService "bf-bench-trained" $TRAINED_MODEL 9933 "trained-mr6" "bf-bench-trained"

Write-Host "Done. Start with: nssm start bf-bench-base; nssm start bf-bench-trained"
