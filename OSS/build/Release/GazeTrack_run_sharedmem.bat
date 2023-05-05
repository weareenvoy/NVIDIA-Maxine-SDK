@echo off
SETLOCAL
SET PATH=%PATH%;..\..\source\external\opencv\bin;..\..\bin;
GazeRedirect.exe --video_source=2 --shared_mem_name="TOPShm" --vector_scale=100
@pause