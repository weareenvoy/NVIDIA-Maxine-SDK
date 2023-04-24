@echo off
SETLOCAL
SET PATH=%PATH%;..\..\source\external\opencv\bin;..\..\bin;
GazeRedirect.exe --video_source=1 --in_file_path=D:\Leviathan\_Clients\_Envoy\NVIDIA-Maxine-SDK\_testvideos\GazeTrackingTest.mov --out_file_path=D:\Leviathan\_Clients\_Envoy\NVIDIA-Maxine-SDK\_testvideos\GazeTrackingTest_Output.mov --capture_outputs=true
@pause