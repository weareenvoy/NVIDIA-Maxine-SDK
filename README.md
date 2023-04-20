# NVIDIA-Maxine-SDK

## Software Downloads
- [Visual Studio Community 2022 @ 17.5.4](https://visualstudio.microsoft.com/vs/community/)
- [NVIDIA CUDA @ 12.1.0](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA AR Installer @ 0.8.3](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/)
    - If NVIDIA is still on 0.8.2, use the downloads on [Dropbox](https://www.dropbox.com/home/Projects/Universal/UNV00006_HollywoodInstallation/09_Dev/_installers/nvidia_maxine_feb23)
- [CMAKE @ 3.26.3](https://cmake.org/download/)
- [OSC Library](https://code.google.com/archive/p/oscpack/downloads)
- [TouchDesigner @ 2022.32660](https://download.derivative.ca/TouchDesigner.2022.32660.exe)

## Documentation
- [NVIDIA Maxine AR SDK Github @ 0.8.2](https://github.com/NVIDIA/MAXINE-AR-SDK)
- [AR SDK System Guide](https://docs.nvidia.com/deeplearning/maxine/ar-sdk-system-guide/index.html)
- [AR SDK Programming Guide](https://docs.nvidia.com/deeplearning/maxine/ar-sdk-programming-guide/index.html)

## Shared Memory
- When using Shared Memory as video input source, you must use **TD @ 2022.32660 or greater** with this repo, or else the system will not be able to properly access frame data due to the updates to the SharedMem C++ code within TouchDesigner. The SharedMem code in `OSS/source/utils/SharedMem` must match the SharedMem code in the Samples folder for your version of TouchDesigner.

## BodyTrack

Arguments are all prefixed with `--`. Ex, `BodyTrack.exe --help` or `BodyTrack.exe --video_source=0`.

| Argument              | Data Type           | Description |
| -----------           | -----------         | ----------- |
| `mode`                | [0, 1]              | Model Mode. 0: High Quality, 1: High Performance. Default is 1. |
| `draw_tracking`       | bool                | Draw tracking information (joints, bbox) on top of frame. Default is true. |
| `draw_window`         | bool                | Draw video feed to window on desktop. Default is true. |
| `draw_fps`            | bool                | Draw FPS debug information on top of frame. Default is true. |
| `use_cuda_graph`      | bool                | Enable faster execution by using cuda graph to capture engine execution. Default is true. |
| `chosen_gpu`          | [0, 1, 2, 3, ...]   | GPU index for running the Maxine instance. Default is 0. |
| `video_source `       | [0, 1, 2]           | Specify video source. 0: Webcam, 1: Video File, 2: Shared Mem (TD). Default is 0. |
| `cam_index`           | [0, 1, 2, 3, ...]   | Specify the webcam index we want to use for the video feed. Default is 0. |
| `shared_mem_name`     | string              | Specify the string name for Shared Memory from TouchDesigner. Default is "TOPShm". |
| `capture_outputs`     | bool                | Enables video/image capture and writing data to file. Default is false. |
| `cam_res`             | [WWWx]HHH (string)  | Specify webcam resolution as height or width x height (`--cam_res=640x480` or `--cam_res=480`). Default is empty string. |
| `in_file_path`        | filepath            | Specify the input file. Default is empty string. |
| `out_file_prefix`     | string              | Specify the output file name (no extension like .mp4). Default is empty string. |
| `codec`               | fourcc              | FOURCC code for the desired codec. Default is H264 (avc1). |
| `model_path`          | filepath            | Specify the directory containing the TRT models. Default is `C:/Program Files/NVIDIA Corporation/NVIDIA AR SDK/models`. |
| `shadow_tracking_age` | int                 | Shadow Tracking Age (frames), after which tracking info of a person is removed. Default is 90. |
| `probation_age`       | int                 | Length of probationary period (frames), after which tracking info of a person is added. Default is 10. |
| `max_targets_tracked` | int                 | Maximum number of targets to be tracked. Default is 30, min value is 1. |
| `send_osc`            | bool                | Enables sending of OSC data to TouchDesigner. Default is true. |
| `status_port`         | int                 | Port for sending out status data (fps, pulse, pid) over OSC. Default is 7000. |
| `keypoints_port`      | int                 | Port for sending out keypoint data (for all 8 possible users) over OSC. Default is 7001. |
| `debug`               | bool                | Report debugging timing info to console. Default is false. |
| `verbose`             | bool                | Report keypoint joint info to console. Default is false. |