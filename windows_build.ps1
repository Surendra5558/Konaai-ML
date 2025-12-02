# Write a function to check last exit code and stop the script if it is not 0
$LASTEXITCODE = 0
Function CheckLastExitCode(){
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Script failed!" -ForegroundColor White -BackgroundColor Red
        # print stack trace
        Write-Host "StackTrace: $($_.Exception.StackTrace)" -ForegroundColor Red

        # reset the last exit code
        $LASTEXITCODE = 0

        # exit the script
        exit 1
    }
}

function InstallPython() {
    # check if pyenv is already installed
    $pyenv = & pyenv --version
    $cwd = (Get-Location).Path
    if ($null -ne $pyenv){
        Write-Host "Pyenv is already installed!" -ForegroundColor Yellow
        Write-Host "Installing Python..."
        # install python version
        pyenv install 3.13.1 > setup.log
        python -m pip install --upgrade pip >> setup.log

        Write-Host "Setting Python 3.13.1 as local version..."
        pyenv local 3.13.1
        CheckLastExitCode
    }
    else {
        Write-Host "Installing Pyenv for Windows..."
        # check if there is a .pyenv directory in current directory and remove it
        Remove-Directory "$cwd\.pyenv"

        # Install pyenv-win
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
        InstallPython
    }
    CheckLastExitCode

    ###################################################################################################
    # Environment Variable Validation
    ###################################################################################################
    # check if PYENV environment variable is set
    if (-not $env:PYENV) {
        Write-Host "PYENV environment variable is not set!" -ForegroundColor Red
        Write-Host "Please restart powershell as admin and rerun this script!!!!" -ForegroundColor Red
        exit 1
    }
    CheckLastExitCode

}

function CheckPython {
    param([int]$Attempt = 1)

    # check if python version is already installed
    # $python = Get-Command python -ErrorAction SilentlyContinue
    $python = & python --version
    $versionNumber = $python.Split(" ")[1]
    # Check if the version matches 3.12.x pattern
    if ($versionNumber -match "^3\.12\.\d+$") {
        Write-Host "Python $versionNumber is already installed!" -ForegroundColor Yellow
    } else {
        if ($Attempt -eq 1) {
            Write-Output "Python version is not 3.12.x" -ForegroundColor Red
            Write-Host "Python version check output: $python" -ForegroundColor Red
            # install python
            InstallPython
        } else {
            Write-Host "Python is not installed!" -ForegroundColor Red
            # exit the script
            exit 1
        }
    }
}

Function Remove-Directory($path){
    # check if the directory exists
    if (Test-Path -Path $path) {
        # Assign ownership of the directory to the current user
        $acl = Get-Acl $path
        $acl.SetOwner([System.Security.Principal.NTAccount] $env:UserName)

        # Set the ACL on the directory
        Set-Acl -Path $path -AclObject $acl

        # Grant full control to the current user
        $acl = Get-Acl $path
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($env:UserName, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
        $acl.SetAccessRule($rule)
        Set-Acl -Path $path -AclObject $acl

        # remove the directory
        Write-Host "Removing $path..."
        Remove-Item -Path $path -Recurse -Force
    }
}

Write-Host "Starting build process..." -ForegroundColor Green

###################################################################################################
# Build Process Starts Here
###################################################################################################
CheckPython

# check again if python is installed with attemp 2
CheckPython -Attempt 2

# Install uv
pip install uv

# Install dependencies
uv sync --no-dev --no-install-project --frozen

# Install pyinstaller
uv add pyinstaller

# find this script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# change directory
Set-Location $scriptDir

# delete everything in the dist directory
$distDir = Join-Path $scriptDir "dist"
Remove-Directory $distDir

# run build script
$buildScript = Join-Path $scriptDir "build_exe.py"
# check if build script exists
if (-not (Test-Path $buildScript)) {
    Write-Host "Build script not found!" -ForegroundColor Red
    exit 1
}
# run build script
uv run $buildScript

# copy the contents of windows_deployment to the dist directory
$deploymentDirectoryName = "windows_deployment"
$distDirWindowsDeployment = Join-Path $distDir $deploymentDirectoryName
# check if windows_deployment directory exists
if (-not (Test-Path $distDirWindowsDeployment)) {
    # create the directory
    New-Item -ItemType Directory -Path $distDirWindowsDeployment
}
# copy the contents of windows_deployment from scriptDir to distDirWindowsDeployment
$sourceDir = Join-Path $scriptDir $deploymentDirectoryName
# check if source directory exists
if (-not (Test-Path $sourceDir)) {
    Write-Host "$deploymentDirectoryName directory not found!" -ForegroundColor Red
    exit 1
}

# copy the contents of windows_deployment from scriptDir to distDirWindowsDeployment
Copy-Item -Path $sourceDir\* -Destination $distDirWindowsDeployment -Recurse
