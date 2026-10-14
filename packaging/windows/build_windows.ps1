[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12.12",
    [string]$AppVersion = "",
    [string]$OutputDirectory = "",
    [string]$PackagingCacheDirectory = "",
    [string]$IsccPath = "",
    [switch]$StageOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackagingDirectory = Split-Path -Parent $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $PackagingDirectory "..")).Path

if ($env:OS -ne "Windows_NT") {
    throw "The Windows package must be built on Windows x64."
}
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "X64") {
    throw "Only the Windows x64 package is currently supported."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Install uv before running this script."
}

if (-not $AppVersion) {
    $AppVersion = (& uv version --short).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the application version from pyproject.toml."
    }
}

$NumericVersion = ($AppVersion -split "[-+]")[0]
$VersionParts = @($NumericVersion.Split("."))
while ($VersionParts.Count -lt 4) {
    $VersionParts += "0"
}
$FileVersion = ($VersionParts[0..3] -join ".")

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot "dist\windows"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $PackagingCacheDirectory) {
    $PackagingCacheDirectory = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "NicoNaviPackaging"
}
$PackagingCacheDirectory = [System.IO.Path]::GetFullPath($PackagingCacheDirectory)

$BuildRoot = Join-Path $ProjectRoot "build\packaging\windows"
$StageDirectory = Join-Path $BuildRoot "stage"
$PythonStore = Join-Path $PackagingCacheDirectory "python"
$RequirementsFile = Join-Path $BuildRoot "requirements.lock.txt"
$RuntimeDirectory = Join-Path $StageDirectory "runtime"
$LibraryDirectory = Join-Path $StageDirectory "lib"

if (Test-Path $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $RuntimeDirectory, $LibraryDirectory, $OutputDirectory | Out-Null

$PreviousPythonInstallDirectory = $env:UV_PYTHON_INSTALL_DIR
$PreviousPythonInstallBin = $env:UV_PYTHON_INSTALL_BIN
$PreviousPythonPath = $env:PYTHONPATH
$PreviousFletUploadDirectory = $env:FLET_UPLOAD_DIR
$PreviousLocalAppData = $env:LOCALAPPDATA

try {
    $env:UV_PYTHON_INSTALL_DIR = $PythonStore
    $env:UV_PYTHON_INSTALL_BIN = "0"

    Write-Host "==> Installing python-build-standalone CPython $PythonVersion"
    & uv python install --managed-python --no-bin --install-dir $PythonStore $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed." }

    $ManagedPython = (& uv python find --no-project --managed-python $PythonVersion).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $ManagedPython) {
        throw "Could not locate the managed Python installation."
    }
    $ManagedRuntimeDirectory = Split-Path -Parent $ManagedPython
    Copy-Item -Path (Join-Path $ManagedRuntimeDirectory "*") -Destination $RuntimeDirectory -Recurse -Force

    $StagePython = Join-Path $RuntimeDirectory "python.exe"
    if (-not (Test-Path $StagePython)) {
        throw "The standalone runtime does not contain python.exe."
    }

    Write-Host "==> Exporting dependencies from uv.lock"
    Push-Location $ProjectRoot
    try {
        & uv export --quiet --frozen --no-dev --no-emit-project --python $StagePython --output-file $RequirementsFile
        if ($LASTEXITCODE -ne 0) { throw "uv export failed." }

        Write-Host "==> Installing locked Windows dependencies"
        & uv pip install --quiet --python $StagePython --target $LibraryDirectory --require-hashes --only-binary :all: --requirements $RequirementsFile
        if ($LASTEXITCODE -ne 0) { throw "Installing locked dependencies failed." }

        Write-Host "==> Installing NicoNavi"
        & uv pip install --quiet --python $StagePython --target $LibraryDirectory --no-deps $ProjectRoot
        if ($LASTEXITCODE -ne 0) { throw "Installing NicoNavi failed." }
    }
    finally {
        Pop-Location
    }

    Copy-Item (Join-Path $PackagingDirectory "desktop_launcher.py") (Join-Path $StageDirectory "launcher.pyw")
    Copy-Item (Join-Path $PackagingDirectory "windows_taskbar.py") (Join-Path $StageDirectory "windows_taskbar.py")
    Copy-Item (Join-Path $ProjectRoot "src\niconavi_app\assets\favicon.png") (Join-Path $StageDirectory "niconavi.png")
    Copy-Item (Join-Path $ProjectRoot "LICENSE") (Join-Path $StageDirectory "LICENSE.txt")

    Write-Host "==> Creating the multi-resolution Windows icon"
    $env:PYTHONPATH = $LibraryDirectory
    & $StagePython -c 'import sys; from PIL import Image; image = Image.open(sys.argv[1]).convert("RGBA"); image.save(sys.argv[2], format="ICO", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])' (Join-Path $StageDirectory "niconavi.png") (Join-Path $StageDirectory "niconavi.ico")
    if ($LASTEXITCODE -ne 0) { throw "Creating niconavi.ico failed." }

    Write-Host "==> Running an import smoke test"
    $SmokeData = Join-Path $BuildRoot "smoke-data"
    New-Item -ItemType Directory -Force -Path $SmokeData | Out-Null
    $env:FLET_UPLOAD_DIR = Join-Path $SmokeData "uploads"
    & $StagePython -c 'import niconavi_app.main; print("NicoNavi import: OK")'
    if ($LASTEXITCODE -ne 0) { throw "The packaged application failed its import test." }
    $env:LOCALAPPDATA = $SmokeData
    $env:PYTHONPATH = $LibraryDirectory
    & $StagePython -c 'import flet, runpy, sys; flet.app = lambda **kwargs: print("Packaged launcher: OK"); runpy.run_path(sys.argv[1], run_name="__main__")' (Join-Path $StageDirectory "launcher.pyw")
    if ($LASTEXITCODE -ne 0) { throw "The packaged launcher failed its smoke test." }

    if ($StageOnly) {
        Write-Host "Stage created at: $StageDirectory"
        return
    }

    if (-not $IsccPath) {
        $IsccCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        if ($IsccCommand) {
            $IsccPath = $IsccCommand.Source
        }
        else {
            $ProgramFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
            $Candidate = Join-Path $ProgramFilesX86 "Inno Setup 6\ISCC.exe"
            if (Test-Path $Candidate) {
                $IsccPath = $Candidate
            }
        }
    }
    if (-not $IsccPath -or -not (Test-Path $IsccPath)) {
        throw "Inno Setup 6 ISCC.exe was not found. Install Inno Setup or pass -IsccPath. The staged app remains at $StageDirectory"
    }

    Write-Host "==> Compiling the Inno Setup installer"
    $InnoScript = Join-Path $PSScriptRoot "niconavi.iss"
    & $IsccPath "/DMyAppVersion=$AppVersion" "/DMyFileVersion=$FileVersion" "/DSourceDir=$StageDirectory" "/DOutputDir=$OutputDirectory" $InnoScript
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

    Write-Host "Done: $(Join-Path $OutputDirectory "NicoNavi-$AppVersion-Setup-x64.exe")"
}
finally {
    $env:UV_PYTHON_INSTALL_DIR = $PreviousPythonInstallDirectory
    $env:UV_PYTHON_INSTALL_BIN = $PreviousPythonInstallBin
    $env:PYTHONPATH = $PreviousPythonPath
    $env:FLET_UPLOAD_DIR = $PreviousFletUploadDirectory
    $env:LOCALAPPDATA = $PreviousLocalAppData
}
