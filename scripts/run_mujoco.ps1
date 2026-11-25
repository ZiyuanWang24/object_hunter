param(
    [ValidateSet('generate', 'run', 'ui', 'viewer', 'room', 'room-ui')]
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
} elseif ($Mode -eq 'ui') {
    & $python (Join-Path $repo 'scripts\run_mujoco_ui.py')
} elseif ($Mode -eq 'viewer') {
    & $python -m mujoco.viewer --mjcf (Join-Path $repo 'assets\mujoco_tabletop.xml')
} elseif ($Mode -eq 'room') {
    & $python (Join-Path $repo 'scripts\run_living_room.py')
} elseif ($Mode -eq 'room-ui') {
    & $python (Join-Path $repo 'scripts\run_room_ui.py')
} else {
    $args = @((Join-Path $repo 'scripts\run_simulation.py'), '--source', (Join-Path $repo 'assets\mujoco_scene.mp4'), '--ground-truth', (Join-Path $repo 'assets\mujoco_ground_truth.jsonl'), '--target-class', $TargetClass, '--record', (Join-Path $repo 'outputs\mujoco_dashboard.mp4'))
    if ($Headless) { $args += '--headless' }
    & $python @args
}
exit $LASTEXITCODE
