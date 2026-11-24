param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$gpuPython = 'C:\AI_APP\minimax_H3\ComfyUI\.venv\Scripts\python.exe'
$gpuSite = 'C:\AI_APP\minimax_H3\ComfyUI\.venv\Lib\site-packages'
$systemSite = 'C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages'
if (-not (Test-Path -LiteralPath $gpuPython)) { throw "CUDA Python environment not found: $gpuPython" }
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:ALL_PROXY = ''
$env:PYTHONPATH = "$gpuSite;$systemSite"
& $gpuPython -S scripts/run_ui.py @Args
exit $LASTEXITCODE
