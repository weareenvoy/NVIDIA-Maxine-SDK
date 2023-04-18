@echo off
SETLOCAL
SET PATH=%PATH%;..\..\source\external\opencv\bin;..\..\bin;
BodyTrack.exe --video_source=2 --shared_mem_name="TOPtest"
@pause