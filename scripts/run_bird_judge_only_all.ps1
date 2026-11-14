$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Rejudge = Join-Path $RepoRoot "agents\toolcall_rejudge.py"
$OutputDir = Join-Path $RepoRoot "runs\toolcall_judges"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$AttackInputs = @(
    "--input", (Join-Path $RepoRoot "results\bird_mini_dev__agentdojo__simguard__attack_postexec_judgefix_v2.json"),
    "--input", (Join-Path $RepoRoot "results\bird_mini_dev__agentdojo__simguard__attack_errors_rerun_v2.json"),
    "--input", (Join-Path $RepoRoot "results\bird_mini_dev__agentdojo__simguard__attack_errors_rerun_round2_v2.json"),
    "--input", (Join-Path $RepoRoot "results\bird_mini_dev__agentdojo__simguard__attack_errors_rerun_round3_v2.json")
)
$BenignInputs = @(
    "--input", (Join-Path $RepoRoot "results\bird_mini_dev__agentdojo__simguard__benign_postexec_judgefix_v2.json")
)
$Judges = @(
    @{ Name = "gpt_oss_120b";      Model = "openrouter/openai/gpt-oss-120b" },
    @{ Name = "gpt_oss_20b";       Model = "openrouter/openai/gpt-oss-20b" },
    @{ Name = "deepseek_v4_pro";   Model = "openrouter/deepseek/deepseek-v4-pro" },
    @{ Name = "deepseek_v4_flash"; Model = "openrouter/deepseek/deepseek-v4-flash" },
    @{ Name = "llama31_70b";       Model = "openrouter/meta-llama/llama-3.1-70b-instruct" },
    @{ Name = "llama31_8b";        Model = "openrouter/meta-llama/llama-3.1-8b-instruct" }
)

foreach ($Judge in $Judges) {
    $AttackOutput = Join-Path $OutputDir "bird_attack_merged_v2__judge_$($Judge.Name).json"
    Write-Output "[$(Get-Date -Format s)] START attack $($Judge.Model)"
    & $Python $Rejudge @AttackInputs `
        --dataset bird `
        --label attack `
        --judge-model $Judge.Model `
        --openai-base-url "https://api.ssunlp.co.kr/v1" `
        --reasoning-effort low `
        --workers 8 `
        --max-attempts 3 `
        --timeout 300 `
        --output $AttackOutput
    Write-Output "[$(Get-Date -Format s)] END attack $($Judge.Model) exit=$LASTEXITCODE"

    $BenignOutput = Join-Path $OutputDir "bird_benign_v2__judge_$($Judge.Name).json"
    Write-Output "[$(Get-Date -Format s)] START benign $($Judge.Model)"
    & $Python $Rejudge @BenignInputs `
        --dataset bird `
        --label benign `
        --judge-model $Judge.Model `
        --openai-base-url "https://api.ssunlp.co.kr/v1" `
        --reasoning-effort low `
        --workers 8 `
        --max-attempts 3 `
        --timeout 300 `
        --output $BenignOutput
    Write-Output "[$(Get-Date -Format s)] END benign $($Judge.Model) exit=$LASTEXITCODE"
}

Write-Output "[$(Get-Date -Format s)] ALL BIRD JUDGE-ONLY RUNS FINISHED"
