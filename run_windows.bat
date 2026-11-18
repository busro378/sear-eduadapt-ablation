@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 pipeline_scientsbank.py) || (python pipeline_scientsbank.py)
if errorlevel 9009 echo Python not found. Install from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
pause
