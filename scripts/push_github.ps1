$ErrorActionPreference = 'Stop'
$token = $env:GITHUB_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
    throw 'GITHUB_TOKEN is not set in this PowerShell session. Set it securely, then run this script again.'
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$git = Join-Path $root '.tools\mingit\cmd\git.exe'
if (-not (Test-Path -LiteralPath $git)) { throw "Portable Git not found: $git" }

$gitRoot = Split-Path (Split-Path $git -Parent) -Parent
$env:PATH = "$(Join-Path $gitRoot 'ucrt64\bin');$(Split-Path $git -Parent);$env:PATH"
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:ALL_PROXY = ''
$env:GIT_HTTP_PROXY = ''
$env:GIT_HTTPS_PROXY = ''
$basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("ZiyuanWang24:$token"))
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'http.extraheader'
$env:GIT_CONFIG_VALUE_0 = "AUTHORIZATION: basic $basic"

& $git --git-dir=(Join-Path $root '.upload-git') --work-tree=$root -c http.sslBackend=openssl push -u origin main
if ($LASTEXITCODE -ne 0) { throw "Git push failed with exit code $LASTEXITCODE" }
Write-Output 'GitHub push completed.'
