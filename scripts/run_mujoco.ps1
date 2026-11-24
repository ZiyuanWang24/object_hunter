param(
    [ValidateSet('generate', 'run')]
    [string]$Mode = 'run',
    [string]$TargetClass = 'cup',
    [switch]$Headless
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$systemSite = 'C:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

$env:PYTHONPATH = "$repo;$systemSite"
if ($Mode -eq 'generate') {
    & $python (Join-Path $repo 'scripts\generate_simulation.py') --output (Join-Path $repo 'assets\mujoco_scene.mp4') --ground-truth (Join-Path $repo 'assets\mujoco_ground_truth.jsonl') --xml (Join-Path $repo 'assets\mujoco_tabletop.xml') --seconds 16
} else {
    $args = @((Join-Path $repo 'scripts\run_simulation.py'), '--source', (Join-Path $repo 'assets\mujoco_scene.mp4'), '--ground-truth', (Join-Path $repo 'assets\mujoco_ground_truth.jsonl'), '--target-class', $TargetClass, '--record', (Join-Path $repo 'outputs\mujoco_dashboard.mp4'))
    if ($Headless) { $args += '--headless' }
    & $python @args
}
exit $LASTEXITCODE
