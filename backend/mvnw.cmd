@echo off
setlocal
set "MAVEN_VERSION=3.9.11"
set "MAVEN_HOME=%USERPROFILE%\.m2\wrapper\dists\apache-maven-%MAVEN_VERSION%"
set "MAVEN_BIN=%MAVEN_HOME%\bin\mvn.cmd"
set "MAVEN_ZIP=%TEMP%\apache-maven-%MAVEN_VERSION%-bin.zip"

if not exist "%MAVEN_BIN%" (
    curl.exe -fsSL "https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/%MAVEN_VERSION%/apache-maven-%MAVEN_VERSION%-bin.zip" -o "%MAVEN_ZIP%"
    if errorlevel 1 exit /b 1
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%MAVEN_ZIP%' -DestinationPath '%MAVEN_HOME%\..' -Force"
    if errorlevel 1 exit /b 1
)

call "%MAVEN_BIN%" %*
exit /b %ERRORLEVEL%
