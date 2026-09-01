# Register the GPU runner on the Windows host (RTX 3080)

How-to for the host that owns the RTX 3080 10GB. After this guide, the
`GPU smoke` and `GPU train` workflows run there, and the `GPU train via SSH`
workflow can reach the host over SSH.

Prerequisites:

- NVIDIA driver installed and up to date (`nvidia-smi` works in PowerShell).
  No CUDA Toolkit is needed: the torch wheels ship the CUDA runtime.
- Git for Windows installed (provides the `bash` shell used by every GPU step).
- At least 15 GB free disk (models + datasets; Qwen3-8B in 4-bit is ~6 GB).
- Admin PowerShell open.

## 1. Register the GitHub self-hosted runner

1. On GitHub, open the repo: Settings > Actions > Runners > New self-hosted
   runner > Windows > x64.
2. In PowerShell on the GPU host, run the three commands GitHub shows
   (download the runner zip, `config.cmd` with the short-lived token, `run.cmd`).
   In `config.cmd`, when asked for labels, add:

   ```
   gpu,self-hosted
   ```

3. Keep `run.cmd` in an interactive console for the first runs. Once stable,
   install it as a service (`svc install`, `svc start`) so it survives reboots.

4. Sanity check: on the host, `mise install` inside the repo clone
   (creates `mise.toml`-pinned uv), then trigger the `GPU smoke` workflow with
   job = `smoke` from the Actions tab. A first run downloads
   `Qwen/Qwen3-0.6B` (~1.5 GB) into the runner HF cache.

## 2. Enable SSH access (for the `GPU train via SSH` workflow)

In admin PowerShell:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic
Start-Service sshd
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server" -Enabled True `
  -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "C:\Program Files\PowerShell\7\pwsh.exe" -PropertyType String -Force
```

(`DefaultShell` falls back to `cmd.exe` if PowerShell 7 is not installed.)

Authorize your key: append the public key to
`C:\ProgramData\ssh\administrators_authorized_keys` (for admin users) and run:

```powershell
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "SYSTEM:F" /grant "BUILTIN\Administrators:F"
```

Clone the repo in the user's home (`cd ~; git clone
git@github.com:log0u7/brainforge.git`), run `mise install` inside it once, and
set `mise.toml`-pinned tools on PATH via `mise activate` in your profile.

## 3. GitHub repository secrets (SSH workflow)

Settings > Secrets and variables > Actions > New repository secret:

- `SSH_HOST`: GPU host address.
- `SSH_USER`: Windows user with the authorized key.
- `SSH_KEY`: private key whose public half is authorized on the host.

## 4. Variant: GitLab or Forgejo runner on the same host

Same principle, different registration:

- GitLab: install `gitlab-runner` for Windows, then
  `gitlab-runner register --url <instance> --token <token> --executor shell --tag-list gpu`.
  The `.gitlab-ci.yml` `train-smoke`/`train` jobs pick it up via `tags: [gpu]`.
- Forgejo: download `act_runner` for Windows, register with the instance and
  add the label `gpu` (`.forgejo/workflows/gpu-*.yml` use
  `runs-on: [self-hosted, gpu]`).

## 5. First real training run

1. Generate a dataset (mock provider is free):
   `uv run brainforge pipeline run security_dataset --input examples/cases --provider mock`.
2. Trigger `GPU` workflow with job = `train`, dataset =
   `datasets/security_dataset.jsonl`, epochs = `1`.
3. Watch the run; artifacts contain the adapter (`experiments/`),
   `train_summary.json` (loss curve) and `eval.json` (perplexity).
4. Scale up: replace the mock dataset with a real one (ROADMAP phase 2) and
   use the `TrainingConfig` defaults (Qwen3-8B, 4-bit, rank 16).
