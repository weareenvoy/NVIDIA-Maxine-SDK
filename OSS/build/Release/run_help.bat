@echo off
SETLOCAL
SET PATH=%PATH%;..\..\source\external\opencv\bin;..\..\bin;
BodyTrack.exe --help
@pause