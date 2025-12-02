###################################################################################################
# Helper Functions
###################################################################################################
# Adding cli binding for the port when deployment script is run automatically as part of CI/CD pipeline
# Call script with -port <port_number>
[CmdletBinding(PositionalBinding = $true)]
param(
    [Parameter(Mandatory = $false,
               HelpMessage = 'Port number (1-65535).')]
    [ValidateRange(1,65535)]
    [int]$Port
)


Function Remove-Services{
    param([string]$nssmPath)

    # list all services that have the word konaai in them irrespective of case
    $services = @(
        "KonaAI Intelligence Worker",
        "KonaAI Intelligence Web Server",
        "KonaAI Intelligence Scheduler",
        "KonaAICelery",
        "KonaAIWeb",
        "KonaAIStreamlit",
        "KonaAIScheduler"
    )

    foreach ($serviceName in $services) {
        # Write-Host "Service with name $($service) already exists!" -ForegroundColor Yellow
        $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue

        if ($null -ne $service) {
            Write-Host "Service '$serviceName' exists."
            try{
                # stop the service
                Write-Host "Stopping $($serviceName) service..."
                Stop-Service -DisplayName $serviceName -Force

                # remove the service
                Write-Host "Removing $($serviceName) service..."
                & "$nssmPath" remove $($serviceName) confirm
            }
            catch{
                Write-Host "Can not remove service $($serviceName)!" -ForegroundColor Yellow
            }
        }

    }
}

# Write a function to remove a directory with a given path
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

###################################################################################################
# Main Workflow
###################################################################################################
# check if the environment variable is set, if not error out
$environmentVariable = "INTELLIGENCE_PATH"
${env:$environmentVariable} = [System.Environment]::GetEnvironmentVariable($environmentVariable, "Machine")
$IntelligencePath = ${env:$environmentVariable}
if (-not $IntelligencePath) {
    Write-Host "$environmentVariable environment variable is not set!" -ForegroundColor Red
    Write-Host "Set the environment variable and restart powershell as admin and rerun this script!!!!" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "$environmentVariable environment variable is set to $IntelligencePath!" -ForegroundColor Green
}

# check for the admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "Please run this script as an administrator!" -ForegroundColor Red
    exit 1
}

# find this script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Current directory: $scriptDir"

# show a message that setup is starting
Write-Host "Starting setup..."

# reset the last exit code
CheckLastExitCode

###################################################################################################
# Set NSSM Path
###################################################################################################
# chose nssm based on the system architecture
if ([Environment]::Is64BitOperatingSystem) {
    # 64-bit system
    Write-Host "64-bit system detected!"
    $nssmPath = $scriptDir + "\nssm\nssm_64.exe"
}
else {
    # 32-bit system
    Write-Host "32-bit system detected!"
    $nssmPath = $scriptDir + "\nssm\nssm_32.exe"
}

# Remove old services
Remove-Services $nssmPath

# create service logs directory
$serviceLogsPath = "$IntelligencePath\service_logs"
if (-not (Test-Path -Path $serviceLogsPath)) {
    Write-Host "Creating service logs directory..."
    New-Item -ItemType Directory -Path $serviceLogsPath
}

$executableName = "KonaAI Intelligence Server.exe"
$parentPath = Split-Path -Parent $scriptDir
# executable in parent path to this script, get executable path
$executablePath = Join-Path -Path $parentPath -ChildPath "$executableName"
###################################################################################################
# Run Python celery as a service
###################################################################################################
$serviceName = "KonaAI Intelligence Worker"
Write-Host "Installing $serviceName service..."
& "$nssmPath" install "$serviceName" "$executablePath"
& "$nssmPath" set "$serviceName" AppParameters "worker"
& "$nssmPath" set "$serviceName" AppStdout "$IntelligencePath\service_logs\worker.log"
& "$nssmPath" set "$serviceName" AppStderr "$IntelligencePath\service_logs\worker.log"
# set log rotation to every 1 day
& "$nssmPath" set "$serviceName" AppRotateFiles 1

# start the service
& "$nssmPath" start "$serviceName"
# check if service is running
Write-Host "Service status: $(& "$nssmPath" status "$serviceName")"
CheckLastExitCode

###################################################################################################
# Run Celery Scheduler as a service
###################################################################################################
$serviceName = "KonaAI Intelligence Scheduler"
Write-Host "Installing $serviceName service..."
& "$nssmPath" install "$serviceName" "$executablePath"
& "$nssmPath" set "$serviceName" AppParameters "scheduler"
& "$nssmPath" set "$serviceName" AppStdout "$IntelligencePath\service_logs\scheduler.log"
& "$nssmPath" set "$serviceName" AppStderr "$IntelligencePath\service_logs\scheduler.log"
# set log rotation to every 1 day
& "$nssmPath" set "$serviceName" AppRotateFiles 1

# start the service
& "$nssmPath" start "$serviceName"
# check if service is running
Write-Host "Service status: $(& "$nssmPath" status "$serviceName")"
CheckLastExitCode

###################################################################################################
# Ask for web server port
###################################################################################################
# ask for port number for the web server
if (-not $PSBoundParameters.ContainsKey('Port')) {
    # if port number is not provided, ask user for input
    Write-Host "Enter the port number for the web server: " -ForegroundColor Cyan -NoNewline
    [int]$user_port = Read-Host # read user input
    if (-not $user_port) {
        Write-Host "No port number entered" -ForegroundColor Red
        # exit the script
        exit 1
    }
    else {
        # validate the port number
        if ($user_port -lt 1 -or $user_port -gt 65535) {
            Write-Host "Invalid port number entered," -ForegroundColor Red
            # exit the script
            exit 1
        }
    }
}
else {
    $user_port = $Port
}


###################################################################################################
# Run Web Server as a service
###################################################################################################
$serviceName = "KonaAI Intelligence Web Server"
Write-Host "Installing $serviceName service..."
& "$nssmPath" install "$serviceName" "$executablePath"
& "$nssmPath" set "$serviceName" AppParameters "web --port $user_port"
& "$nssmPath" set "$serviceName" AppStdout "$IntelligencePath\service_logs\web.log"
& "$nssmPath" set "$serviceName" AppStderr "$IntelligencePath\service_logs\web.log"
# set log rotation to every 1 day
& "$nssmPath" set "$serviceName" AppRotateFiles 1

# start the service
& "$nssmPath" start "$serviceName"
# check if service is running
Write-Host "Service status: $(& "$nssmPath" status "$serviceName")"
CheckLastExitCode

###################################################################################################
# Install ODBC Driver 18 for SQL Server
###################################################################################################
# check of ODBC Driver 18 for SQL Server is already installed
if (Get-ODBCDriver -Name "ODBC Driver 18 for SQL Server" -ErrorAction SilentlyContinue) {
    Write-Host "ODBC Driver 18 for SQL Server is already installed!" -ForegroundColor Yellow
    Write-Host "Skipping ODBC Driver 18 for SQL Server installation..." -ForegroundColor Yellow
}
else {
    Write-Host "Installing ODBC Driver 18 for SQL Server..."
    # check if its a 32-bit or 64-bit system
    if ([Environment]::Is64BitOperatingSystem) {
        # 64-bit system
        # download ODBC Driver 18 for SQL Server
        Write-Host "Downloading ODBC Driver 18 for SQL Server..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://go.microsoft.com/fwlink/?linkid=2242886" -OutFile "./msodbcsql.msi"
        # install ODBC Driver 18 for SQL Server
        Write-Host "Installing ODBC Driver 18 for SQL Server..."
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i msodbcsql.msi /qn /norestart" -Wait
        Remove-Item -Path "./msodbcsql.msi"
    }
    else {
        # 32-bit system
        # download ODBC Driver 18 for SQL Server
        Write-Host "Downloading ODBC Driver 18 for SQL Server..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://go.microsoft.com/fwlink/?linkid=2242980" -OutFile "./msodbcsql.msi"
        # install ODBC Driver 18 for SQL Server
        Write-Host "Installing ODBC Driver 18 for SQL Server..."
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i msodbcsql.msi /qn /norestart" -Wait
        Remove-Item -Path "./msodbcsql.msi"
    }
    CheckLastExitCode
}
