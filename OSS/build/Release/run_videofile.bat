@echo off
SETLOCAL
SET PATH=%PATH%;..\..\source\external\opencv\bin;..\..\bin;
BodyTrack.exe --video_source=1 --in_file_path=D:\Leviathan\_Clients\_Envoy\_testvideos\Stretching.mp4 --out_file_prefix=D:\Leviathan\_Clients\_Envoy\_testvideos\Stretching --capture_outputs=true
@pause