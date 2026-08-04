@echo off
REM spt-generated endpoint shortcut — safe to regenerate (do not remove this line)
REM cc-claudette: launch the `claudette` endpoint with the baked selection.
REM   cmd.exe (project dir):  cc-claudette      PowerShell:  .\cc-claudette
spt endpoint run --adapter claude-spt --id claudette --create %*
