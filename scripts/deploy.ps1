$scriptPath = $MyInvocation.MyCommand.Path
$scriptDirectory = Split-Path -Parent $scriptPath
Set-Location (Split-Path -Parent $scriptDirectory)

Write-Output "Initializing and updating submodules..."
&git submodule update --init --recursive

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "The 'uv' command was not found. See https://docs.astral.sh/uv/getting-started/installation/ for instructions."
}

Write-Output "Creating a Python environment..."
if (Test-Path -Path ./.venv) {
    Remove-Item ./.venv -Recurse -Force
}
&uv venv
.\.venv\Scripts\Activate.ps1
Write-Output "Synchronizing environment..."
&uv sync

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw "The 'dotnet' command was not found. Install the .NET SDK: https://dotnet.microsoft.com/download"
}

Write-Output "Restoring dotnet tools..."
&dotnet tool restore

if (-not (Test-Path -Path src\DataSchemas)) {
    New-Item -ItemType Directory -Path src\DataSchemas | Out-Null
}
if (-not (Test-Path -Path src\Extensions)) {
    New-Item -ItemType Directory -Path src\Extensions | Out-Null
}

Write-Output "Regenerating JSON schema and Bonsai extension classes from pydantic models..."
&uv run regenerate-schemas

Write-Output "Creating a Bonsai environment and installing packages..."
if (Test-Path -Path "bonsai") {
    Set-Location "bonsai"
    .\setup.ps1
} elseif (Test-Path -Path ".bonsai") {
    Set-Location ".bonsai"
    .\setup.ps1
} else {
    throw "Neither 'bonsai' nor '.bonsai' directory found."
}
Set-Location ..
