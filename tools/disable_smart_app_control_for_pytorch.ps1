$ErrorActionPreference = "Stop"

$log = "D:\workspace\pv_missing_modility\tools\disable_smart_app_control_for_pytorch.log"
"[$(Get-Date -Format s)] Starting Smart App Control / CI policy repair" | Out-File -FilePath $log -Encoding utf8

function Write-Log($message) {
    "[$(Get-Date -Format s)] $message" | Tee-Object -FilePath $log -Append
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script must run as Administrator."
}

Write-Log "Running elevated."

$ciPolicyKey = "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy"
$srpKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Srp\Gp"

if (Test-Path $ciPolicyKey) {
    Write-Log "Setting VerifiedAndReputablePolicyState=0"
    Set-ItemProperty -Path $ciPolicyKey -Name VerifiedAndReputablePolicyState -Type DWord -Value 0
}

if (Test-Path $srpKey) {
    Write-Log "Setting LastSmartlockerEnabled=0"
    Set-ItemProperty -Path $srpKey -Name LastSmartlockerEnabled -Type DWord -Value 0
}

$policyIds = @(
    "{0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}",
    "{1283AC0F-FFF1-49AE-ADA1-8A933130CAD6}"
)

function Invoke-CiTool($arguments) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "$env:SystemRoot\System32\CiTool.exe"
    $psi.Arguments = $arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = [System.Diagnostics.Process]::Start($psi)
    $process.StandardInput.WriteLine()
    $process.StandardInput.Close()

    if (-not $process.WaitForExit(15000)) {
        $process.Kill()
        Write-Log "CiTool timed out: $arguments"
        return
    }

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if ($stdout) { $stdout | Tee-Object -FilePath $log -Append }
    if ($stderr) { $stderr | Tee-Object -FilePath $log -Append }
    Write-Log "CiTool exit code: $($process.ExitCode)"
}

foreach ($policyId in $policyIds) {
    Write-Log "Attempting CiTool remove-policy $policyId"
    try {
        Invoke-CiTool "--remove-policy $policyId"
    } catch {
        Write-Log "CiTool remove-policy failed for ${policyId}: $($_.Exception.Message)"
    }
}

Write-Log "Refreshing Code Integrity policy"
try {
    Invoke-CiTool "--refresh"
} catch {
    Write-Log "CiTool refresh failed: $($_.Exception.Message)"
}

Write-Log "Current relevant registry values:"
reg query HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy /v VerifiedAndReputablePolicyState | Tee-Object -FilePath $log -Append
reg query HKLM\SYSTEM\CurrentControlSet\Control\Srp\Gp /v LastSmartlockerEnabled | Tee-Object -FilePath $log -Append

Write-Log "Repair complete. Reboot is usually required."
