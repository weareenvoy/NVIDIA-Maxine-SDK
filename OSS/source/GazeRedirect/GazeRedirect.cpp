/*###############################################################################
#
# Copyright 2020 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
###############################################################################*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <process.h>	// for PID

using namespace std;

// nvidia maxine
#include "GazeEngine.h"
#include "GazeRedirect.h"
#include "RenderingUtils.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"

// touchdesigner shared memory
#include "SharedMem/TOP_SharedMemHeader.h"
#include "SharedMem/UT_Mutex.h"
#include "SharedMem/UT_SharedMem.h"

// sending over osc
#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"
#include "ip/IpEndpointName.h"

// using cuda for gpu distribution
#include "cuda/cuda.h"
#include "cuda/cuda_device_runtime_api.h"
#include "cuda/cuda_runtime_api.h"
#include "cuda/cudart_platform.h"

// helper classes
#include "Timer.h"

/********************************************************************************
 * constant global values
 ********************************************************************************/

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
#define M_2PI 6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI ((float)M_PI)
#define F_PI_2 ((float)M_PI_2)
#define F_2PI ((float)M_2PI)
#define DEGREES_PER_RADIAN (180.0 / M_PI)

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

enum {
	myErrNone = 0,
	myErrShader = -1,
	myErrProgram = -2,
	myErrTexture = -3,
};

// video source for frame data
enum {
	webcam = 0,
	videoFile = 1,
	sharedMemory = 2
};

// for fps calculation and display
const float TIME_CONSTANT = 16.f;
const float FRAMETIME_THRESHOLD = 100.f;
const float FPS_DELAY_THRESHOLD = 0.9f;
const auto DEBUG_TEXT_COLOR = cv::Scalar(255, 255, 255);	// white
const int DEBUG_TEXT_SCALE = 1;
const int DEBUG_FONT_THICKNESS = 1;
const int DEBUG_FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;

// shared memory
UT_SharedMem* shm;						// shared memory data
TOP_SharedMemHeader* shmTopHeader;		// TOP_SharedMemHeader.h header data
cv::Mat shmTempFrame;					// for converting touch shared mem to opencv mat
int shmWidth, shmHeight;				// width and height of image from shared memory

// specifies whether to use offline video or an online camera video as the input. 
// when offline mode is true --> Disables webcam, reads video from file, and writes output video results to file
bool offlineMode = false;				// False: Webcam or SharedMem, True: Video file

// OSC communication
#define OUTPUT_BUFFER_SIZE 1024
UdpTransmitSocket* transmitSocket_keypoints;
UdpTransmitSocket* transmitSocket_status;
const std::string USER_STR = "/user";
const std::string FPS_STR = "/fps";
const std::string PULSE_STR = "/pulse";
const std::string PID_STR = "/pid";

// process id
int pid = _getpid();

const bool GAZE_REDIRECT = false;		// we only care about gaze tracking data and not adjusting the eye position

// for setting level of detail for face tracking
const unsigned int LANDMARKS_126 = 126;
const unsigned int LANDMARKS_68 = 68;


/********************************************************************************
 * command-line arguments
 ********************************************************************************/

bool
FLAG_sendOsc = true,					// Send OSC data out of Maxine
FLAG_drawTracking = true,				// Draw keypoint and bounding box data to window
FLAG_drawWindow = true,					// Draw window with video feed to desktop
FLAG_drawFPS = true,					// Write FPS debug information to window
FLAG_captureOutputs = false,			// Enables video/image capture and writing body detection/keypoints outputs to file. If input is video file, gets set to true
FLAG_useCudaGraph = true,				// Uses CUDA Graphs to improve performance. CUDA graph reduces the overhead of the GPU operation submission of 3D body tracking
FLAG_temporalSmoothing = true,			// Temporally optimize face rect bounding box and facial landmarks
FLAG_useDetailedLandmarks = false;		// Display 68 (false) or 126 (true) facial landmarks

std::string
FLAG_inFilePath,						// input file path on disk for video source (path + name + prefix)
FLAG_outFilePath,						// output file path for writing data to disk (path + name + prefix)
FLAG_camRes,							// If offlineMode=false, specifies the cam res. If width omitted, width is computed from height to give an aspect ratio of 4:3.
FLAG_captureCodec = "avc1",				// avc1 = h264
FLAG_modelPath = "C:/Program Files/NVIDIA Corporation/NVIDIA AR SDK/models",	// default installation location
FLAG_sharedMemName = "TOPShm";

unsigned int
FLAG_videoSource = sharedMemory,		// Specify video source. 0: Webcam, 1: Video File, 2: Shared Mem (TouchDesigner).
FLAG_camIndex = 0,						// Index of webcam connected to the PC
FLAG_chosenGPU = 0,
FLAG_eyeSizeSensitivity = 3,
FLAG_keypointsPort = 7002,				// Sets the port on which we send out all keypoint data (for all 8 users)
FLAG_statusPort = 7003,					// Sets the port on which we send out FPS, Pulse, and PID data
FLAG_vectorScale = 50.0f;			// scale of the head translation and gaze direction vector. Default is 50.

/********************************************************************************
 * parsing command line args
 ********************************************************************************/

static void Usage() {
	printf(
		"GazeRedirect [<args> ...]\n"
		"where <args> is\n"
		" --landmarks_detailed[=(true|false)]   Set the number of facial landmark points to 126 (true), otherwise 68 (false). Default is false.\n"
		" --eyesize_sensitivity[=(2|3|4|5|6)]	Set the eye size sensitivity parameter, an integer value between 2 and 6. Default is 3.\n"
		" --vector_scale[=(float)]				Scale of the head translation and gaze direction vector. Default is 50.\n"
		" --temporal[=(true|false)]				Temporally optimize face rect and landmarks. Default is true.\n"
		" --draw_tracking[=(true|false)]		Draw tracking information (joints, bbox) on top of frame. Default is true.\n"
		" --draw_window[=(true|false)]			Draw video feed to window on desktop. Default is true.\n"
		" --draw_fps[=(true|false)]				Draw FPS debug information on top of frame. Default is true.\n"
		" --use_cuda_graph[=(true|false)]		Enable faster execution by using cuda graph to capture engine execution. Default is true.\n"
		" --chosen_gpu[=(0|1|2|3|..)]			GPU index for running the Maxine instance. Default is 0.\n"
		" --video_source[=(0|1|2)]				Specify video source. 0: Webcam, 1: Video File, 2: Shared Mem (TD). Default is 0.\n"
		" --cam_index[=(0|1|2|3|..)]			Specify the webcam index we want to use for the video feed. Default is 0.\n"
		" --shared_mem_name=<string>			Specify the string name for Shared Memory from TouchDesigner. Default is 'TOPshm'.\n"
		" --cam_res=[WWWx]HHH					Specify webcam resolution as height or width x height (--cam_res=640x480 or --cam_res=480). Default is empty string.\n"
		" --in_file_path=<file>					Specify the input file path. Default is empty string.\n"
		" --out_file_path=<file>				Specify the output file path. Default is empty string.\n"
		" --capture_outputs[=(true|false)]	Enables video/image capture and writing data to file. Default is false.\n"
		" --codec=<fourcc>						FOURCC code for the desired codec. Default is H264 (avc1).\n"
		" --model_path=<path>					Specify the directory containing the TRT models.\n"
		" --send_osc[=(true|false)]				Enables sending of OSC data to TouchDesigner. Default is true."
		" --status_port=<0000>					Port for sending out status data (fps, pulse, pid) over OSC. Default is 7000.\n"
		" --keypoints_port=<0000>				Port for sending out keypoint data (for all 8 possible users) over OSC. Default is 7001.\n"
	);
}

static bool GetFlagArgVal(const char* flag, const char* arg, const char** val) {
	if (*arg != '-') {
		return false;
	}
	while (*++arg == '-') {
		continue;
	}
	const char* s = strchr(arg, '=');
	if (s == NULL) {
		if (strcmp(flag, arg) != 0) {
			return false;
		}
		*val = NULL;
		return true;
	}
	unsigned n = (unsigned)(s - arg);
	if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) {
		return false;
	}
	*val = s + 1;
	return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, std::string* val) {
	const char* valStr;
	if (!GetFlagArgVal(flag, arg, &valStr)) return false;
	val->assign(valStr ? valStr : "");
	return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, bool* val) {
	const char* valStr;
	bool success = GetFlagArgVal(flag, arg, &valStr);
	if (success) {
		*val = (valStr == NULL || strcasecmp(valStr, "true") == 0 || strcasecmp(valStr, "on") == 0 ||
			strcasecmp(valStr, "yes") == 0 || strcasecmp(valStr, "1") == 0);
	}
	return success;
}

bool GetFlagArgVal(const char* flag, const char* arg, long* val) {
	const char* valStr;
	bool success = GetFlagArgVal(flag, arg, &valStr);
	if (success) {
		*val = strtol(valStr, NULL, 10);
	}
	return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, unsigned* val) {
	long longVal;
	bool success = GetFlagArgVal(flag, arg, &longVal);
	if (success) {
		*val = (unsigned)longVal;
	}
	return success;
}

static int ParseMyArgs(int argc, char** argv) {
	// query NVAR_MODEL_DIR environment variable first before checking the command line arguments
	const char* modelPath = getenv("NVAR_MODEL_DIR");
	if (modelPath) {
		FLAG_modelPath = modelPath;
	}

	int errs = 0;
	for (--argc, ++argv; argc--; ++argv) {
		bool help;
		const char* arg = *argv;
		if (arg[0] != '-') {
			continue;
		}
		else if ((arg[1] == '-') && (
			GetFlagArgVal("landmarks_detailed", arg, &FLAG_useDetailedLandmarks) ||
			GetFlagArgVal("eyesize_sensitivity", arg, &FLAG_eyeSizeSensitivity) ||
			GetFlagArgVal("vector_scale", arg, &FLAG_vectorScale) ||
			GetFlagArgVal("temporal", arg, &FLAG_temporalSmoothing) ||
			GetFlagArgVal("draw_tracking", arg, &FLAG_drawTracking) ||
			GetFlagArgVal("draw_window", arg, &FLAG_drawWindow) ||
			GetFlagArgVal("draw_fps", arg, &FLAG_drawFPS) ||
			GetFlagArgVal("use_cuda_graph", arg, &FLAG_useCudaGraph) ||
			GetFlagArgVal("chosen_gpu", arg, &FLAG_chosenGPU) ||
			GetFlagArgVal("video_source", arg, &FLAG_videoSource) ||
			GetFlagArgVal("cam_index", arg, &FLAG_camIndex) ||
			GetFlagArgVal("shared_mem_name", arg, &FLAG_sharedMemName) ||
			GetFlagArgVal("cam_res", arg, &FLAG_camRes) ||
			GetFlagArgVal("in_file_path", arg, &FLAG_inFilePath) ||
			GetFlagArgVal("out_file_path", arg, &FLAG_outFilePath) ||
			GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||
			GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||
			GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
			GetFlagArgVal("send_osc", arg, &FLAG_sendOsc) ||
			GetFlagArgVal("status_port", arg, &FLAG_statusPort) ||
			GetFlagArgVal("keypoints_port", arg, &FLAG_keypointsPort) 
			)) {
			continue;
		}
		else if (GetFlagArgVal("help", arg, &help)) {
			Usage();
		}
		else {
			printf("Unknown flag: \"%s\"\n", arg);
		}
	}
	return errs;
}

/********************************************************************************
 * error codes
 ********************************************************************************/

const char* GazeTrack::errorStringFromCode(GazeTrack::Err code) {
	struct LUTEntry {
		Err code;
		const char* str;
	};
	static const LUTEntry lut[] = {
	  {errNone, "no error"},
	  {errGeneral, "an error has occured"},
	  {errRun, "an error has occured while the feature is running"},
	  {errInitialization, "Initializing Gaze Engine failed"},
	  {errRead, "an error has occured while reading a file"},
	  {errEffect, "an error has occured while creating a feature"},
	  {errParameter, "an error has occured while setting a parameter for a feature"},
	  {errUnimplemented, "the feature is unimplemented"},
	  {errMissing, "missing input parameter"},
	  {errVideo, "no video source has been found"},
	  {errImageSize, "the image size cannot be accommodated"},
	  {errNotFound, "the item cannot be found"},
	  {errGLFWInit, "GLFW initialization failed"},
	  {errGLInit, "OpenGL initialization failed"},
	  {errRendererInit, "renderer initialization failed"},
	  {errGLResource, "an OpenGL resource could not be found"},
	  {errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
	  {errSDK, "an SDK error has occurred"},
	  {errCuda, "a CUDA error has occurred"},
	  {errCancel, "the user cancelled"},
	  {errCamera, "unable to connect to the camera"},
	};
	for (const LUTEntry* p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
		if (p->code == code) return p->str;
	static char msg[18];
	snprintf(msg, sizeof(msg), "error #%d", code);
	return msg;
}

/********************************************************************************
 * osc communication
 ********************************************************************************/

namespace osc {

	void InitOSC()
	{
		transmitSocket_keypoints = new UdpTransmitSocket(IpEndpointName("127.0.0.1", FLAG_keypointsPort));
		transmitSocket_status = new UdpTransmitSocket(IpEndpointName("127.0.0.1", FLAG_statusPort));
	}

	void SendGazeData(float head3D_X, float head3D_Y, float head3D_Z, cv::Point headPose2D, NvAR_Point3f* gazeDirection3D, std::vector<cv::Point2f> gazeDirection2D)
	{
		char buffer_keypoints[OUTPUT_BUFFER_SIZE] = {};
		osc::OutboundPacketStream packet_keypoints(buffer_keypoints, OUTPUT_BUFFER_SIZE);

		// convert data to readable strings
		std::string oscAddyHead3DX = "/head_3D_translation_x";
		std::string oscAddyHead3DY = "/head_3D_translation_y";
		std::string oscAddyHead3DZ = "/head_3D_translation_z";
		std::string oscAddyGaze3D_OriginX = "/gaze_3D_origin_x";
		std::string oscAddyGaze3D_OriginY = "/gaze_3D_origin_y";
		std::string oscAddyGaze3D_OriginZ = "/gaze_3D_origin_z";
		std::string oscAddyGaze3D_TargetX = "/gaze_3D_target_x";
		std::string oscAddyGaze3D_TargetY = "/gaze_3D_target_y";
		std::string oscAddyGaze3D_TargetZ = "/gaze_3D_target_z";
		std::string oscAddyHead2D_X = "/head_2D_x";
		std::string oscAddyHead2D_Y = "/head_2D_y";
		std::string oscAddyGaze2D_OriginX = "/gaze_2D_origin_x";
		std::string oscAddyGaze2D_OriginY = "/gaze_2D_origin_y";
		std::string oscAddyGaze2D_TargetX = "/gaze_2D_target_x";
		std::string oscAddyGaze2D_TargetY = "/gaze_2D_target_y";
		std::string oscVectorScale = "/vector_scale";

		// send OSC bundle -- each user has their own keypoint osc endpoint
		// send x position data
		packet_keypoints.Clear();
		packet_keypoints << osc::BeginBundleImmediate
			<< osc::BeginMessage(oscAddyHead3DX.c_str()) << head3D_X << osc::EndMessage
			<< osc::BeginMessage(oscAddyHead3DY.c_str()) << head3D_Y << osc::EndMessage
			<< osc::BeginMessage(oscAddyHead3DZ.c_str()) << head3D_Z << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_OriginX.c_str()) << gazeDirection3D[0].x << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_OriginY.c_str()) << gazeDirection3D[0].y << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_OriginZ.c_str()) << gazeDirection3D[0].z << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_TargetX.c_str()) << gazeDirection3D[1].x << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_TargetY.c_str()) << gazeDirection3D[1].y << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze3D_TargetZ.c_str()) << gazeDirection3D[1].z << osc::EndMessage
			<< osc::BeginMessage(oscAddyHead2D_X.c_str()) << headPose2D.x << osc::EndMessage
			<< osc::BeginMessage(oscAddyHead2D_Y.c_str()) << headPose2D.y << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze2D_OriginX.c_str()) << gazeDirection2D[0].x << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze2D_OriginY.c_str()) << gazeDirection2D[0].y << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze2D_TargetX.c_str()) << gazeDirection2D[1].x << osc::EndMessage
			<< osc::BeginMessage(oscAddyGaze2D_TargetY.c_str()) << gazeDirection2D[1].y << osc::EndMessage
			<< osc::BeginMessage(oscVectorScale.c_str()) << float(FLAG_vectorScale) << osc::EndMessage
			<< osc::EndBundle;
		transmitSocket_keypoints->Send(packet_keypoints.Data(), packet_keypoints.Size());	
	}

	void SendStatusData(float fps)
	{
		char buffer_status[OUTPUT_BUFFER_SIZE] = {};
		osc::OutboundPacketStream packet_status(buffer_status, OUTPUT_BUFFER_SIZE);

		int pulse = int(fps) % 2;

		packet_status.Clear();
		packet_status << osc::BeginBundleImmediate
			<< osc::BeginMessage(FPS_STR.c_str()) << fps << osc::EndMessage
			<< osc::BeginMessage(PULSE_STR.c_str()) << pulse << osc::EndMessage
			<< osc::BeginMessage(PID_STR.c_str()) << pid << osc::EndMessage
			<< osc::EndBundle;
		transmitSocket_status->Send(packet_status.Data(), packet_status.Size());
	}
}

/********************************************************************************
 * helper functions
 ********************************************************************************/

static int StringToFourcc(const std::string& str) {
	union chint {
		int i;
		char c[4];
	};
	chint x = { 0 };
	for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
	return x.i;
}

std::string getCalendarTime() {
	// Get the current time
	std::chrono::system_clock::time_point currentTimePoint = std::chrono::system_clock::now();
	// Convert to time_t from time_point
	std::time_t currentTime = std::chrono::system_clock::to_time_t(currentTimePoint);
	// Convert to tm to get structure holding a calendar date and time broken down into its components.
	std::tm brokenTime = *std::localtime(&currentTime);
	std::ostringstream calendarTime;

	// std::put_time has not been implemented under GCC5. In order to support centOS7(GCC4.8), we use strftime.
	// (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
	// Get the time since epoch 0(Thu Jan  1 00:00:00 1970) and the remainder after division is our milliseconds
	char foo[24];
	if (0 < strftime(foo, sizeof(foo), "%Y-%m-%d-%H-%M-%S", &brokenTime)) {
		//cout << foo << endl;
		std::string fooStr(foo);
		calendarTime << fooStr;
	}

	std::chrono::milliseconds currentMilliseconds =
		std::chrono::duration_cast<std::chrono::milliseconds>(currentTimePoint.time_since_epoch()) % 1000;
	// Append the milliseconds to the stream
	calendarTime << "-" << std::setfill('0') << std::setw(3) << currentMilliseconds.count();  // milliseconds
	return calendarTime.str();
}

/********************************************************************************
 * GazeTrack class
 ********************************************************************************/

GazeTrack* gApp = nullptr;
const char GazeTrack::windowTitle[] = "Envoy Maxine GazeTrack";
char* g_nvARSDKPath = NULL;

GazeTrack::GazeTrack() {
	// Make sure things are initialized properly
	gApp = this;
	frameTime = 0;
	frameIndex = 0;
	nvErr = GazeEngine::errNone;
}

GazeTrack::~GazeTrack() {}

/********************************************************************************
 * draw tracking data and bounding boxes to window
 ********************************************************************************/

void GazeTrack::drawBBoxes(const cv::Mat& src, NvAR_Rect* output_bbox) {
	float x, y, width, height;
	cv::Point rectPoint1, rectPoint2;
	cv::Scalar rectColor = cv::Scalar(255, 0, 0);	// red
	int rectThickness = 2;

	// get frame data
	cv::Mat frm = (offlineMode) ? src.clone() : src;

	if (output_bbox) {
		// extract bounding box data
		x = output_bbox->x;
		y = output_bbox->y;
		width = output_bbox->width;
		height = output_bbox->height;
		rectPoint1 = cv::Point(lround(x), lround(y));
		rectPoint2 = cv::Point(lround(x + width), lround(y + height));

		// draw bounding box
		cv::rectangle(frm, rectPoint1, rectPoint2, rectColor, rectThickness);
	}
}

void GazeTrack::DrawLandmarkPoints(const cv::Mat& src, NvAR_Point2f* facial_landmarks, int numLandmarks, cv::Scalar* color) {
	if (!facial_landmarks)
		return;

	NvAR_Point2f* pt, * endPt;
	cv::Point circleCenter;
	float x, y;
	int circleRadius = 1.5;
	int circleThickness = -1;	// filled

	// get frame data
	cv::Mat frm = (offlineMode) ? src.clone() : src;

	// draw a circle for each landmark point
	for (endPt = (pt = (NvAR_Point2f*)facial_landmarks) + numLandmarks; pt < endPt; ++pt) {
		x = pt->x;
		y = pt->y;
		circleCenter = cv::Point(lround(x), lround(y));
		cv::circle(frm, circleCenter, circleRadius, *color, circleThickness);
	}
}

/********************************************************************************
 * write data to file
 ********************************************************************************/

GazeTrack::Err GazeTrack::writeVideo(const cv::Mat& frm) {
	if (FLAG_captureOutputs) {
		if (!capturedVideo.isOpened()) {
			//Assign the filename for capturing video
			const std::string currentCalendarTime = getCalendarTime();
			const std::string capturedOutputFileName = currentCalendarTime + ".mp4";
			getFPS();

			if (frameTime) {
				float fps = (float)(1.0 / frameTime);
				// Open the video for writing.
				capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps, cv::Size(frm.cols, frm.rows));

				if (!capturedVideo.isOpened()) {
					std::cout << "Error: Could not open video for writing: \"" << capturedOutputFileName << "\"\n";
					return errVideo;
				}

				std::cout << "Capturing video started" << std::endl;

				capturedVideo << frm;
			}
			else {  // If frameTime is 0.f, returns without writing the frame to the Video
				return errNone;
			}
		}
		else {
			// Write each frame to the Video
			capturedVideo << frm;
		}
	}
	else {
		if (capturedVideo.isOpened()) {
			std::cout << "Capturing video ended" << std::endl;
			capturedVideo.release();
		}
	}
	return errNone;
}

/********************************************************************************
 * draw stats to window
 ********************************************************************************/

void GazeTrack::getFPS() {
	frameTimer.stop();
	float t = (float)frameTimer.elapsedTimeFloat();
	if (t < FRAMETIME_THRESHOLD) {
		if (frameTime)
			frameTime += (t - frameTime) * (1.f / TIME_CONSTANT);  // 1 pole IIR filter
		else
			frameTime = t;
	}
	else {            // Ludicrous time interval; reset
		frameTime = 0.f;  // WAKE UP
	}
	frameTimer.start();
}

void GazeTrack::drawFPS(cv::Mat& img) {
	//getFPS();	// pulling this out so it only gets called once on a global var
	if (frameTime && FLAG_drawFPS) {
		char buf[32];
		snprintf(buf, sizeof(buf), "%.1f", 1. / frameTime);
		cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), DEBUG_FONT_FACE, DEBUG_TEXT_SCALE, DEBUG_TEXT_COLOR, DEBUG_FONT_THICKNESS);
	}
}

void GazeTrack::sendFPS() {
	if (FLAG_sendOsc) {
		fpsSendDelayTimer.pause();
		float fpsDelayTime = (float)fpsSendDelayTimer.elapsedTimeFloat();
		if (fpsDelayTime >= FPS_DELAY_THRESHOLD) {
			osc::SendStatusData(1. / frameTime);
			fpsSendDelayTimer.stop();
			fpsSendDelayTimer.start();
		}
		else {
			fpsSendDelayTimer.resume();
		}
	}
}

/********************************************************************************
 * acquire joints keypoints and bounding boxes
 ********************************************************************************/

GazeTrack::Err GazeTrack::acquireWebcamOrVideoFrame() {
	Err err = errNone;

	// If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
	// frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
	// resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
	// done here) as well as reallocate memory accordingly with GazeEngine::initGazeRedirectionIOParams()
	cap >> frame;  // get a new frame from camera into the class variable frame.
	if (frame.empty()) {
		// if in Offline mode, this means end of video,so we return
		if (offlineMode) return errVideo;
		// try Init one more time if reading frames from camera
		err = initCamera(FLAG_camRes.c_str(), FLAG_camIndex);
		if (err != errNone) return err;
		cap >> frame;
		if (frame.empty()) return errVideo;
	}
	return err;
}

GazeTrack::Err GazeTrack::acquireSharedMemFrame()
{
	/*
	* !! IMPORTANT !!
	* Your video input must be flipped along the Y before getting to the Shared Mem Out TOP
	* Use a Flip TOP before the Shared Mem Out TOP and toggle "Flip Y" to ON.
	*/

	// Before you read or write to the memory, you need to lock it.
	// If it's able to lock the memory then you can get the pointer to the memory and use it.		
	if (shm == NULL) {
		printf("ERROR shared memory is null when trying to acquire frame\n");
		return errSharedMem;
	}
	if (!shm->tryLock(1000))
	{
		printf("Failed to get a lock on shared memory!\n");
		return errNone;		// keep moving forward, try again next loop
	}
	else
	{
		shmTopHeader = (TOP_SharedMemHeader*)shm->getMemory();
		if (shmTopHeader == NULL || shm->getErrorState() != UT_SHM_ERR_NONE)
		{
			// idk if we still need to do this null check but doing it just in case for clarity
			printf("ERROR: No shared memory when trying to acquire frame\n");
			shm->unlock();
			return errSharedMem;
		}
		else // if (topHeader && (topHeader != NULL)) 
		{
			shmTempFrame.data = (unsigned char*)shmTopHeader->getImage();
			shm->unlock();

			if (shmTempFrame.empty()) {
				// could not read frame from shared memory
				printf("ERROR: Frame is empty and does not contain SharedMem image data\n");
				return errSharedMemVideo;
			}

			frame = shmTempFrame;
			return errNone;
		}
	}
}

GazeTrack::Err GazeTrack::acquireGazeRedirection() {
	GazeTrack::Err doErr = errNone;

	// we don't want to shift the gaze, we only care about gaze tracking data
	nvErr = gaze_ar_engine.acquireGazeRedirection(frame, gazeRedirectFrame);

	if (GazeEngine::Err::errNone == nvErr) {

		// pick the most prominent user
		frameTimer.pause();
		NvAR_Rect* bbox = gaze_ar_engine.getLargestBox();

		// Check for valid bounding box in case confidence check fails
		if (bbox) {

			char buf[64];
			float gazePitch, gazeYaw;
			float head3D_X, head3D_Y, head3D_Z, gaze3D_X, gaze3D_Y, gaze3D_Z;
			float fontScale = (inputHeight <= 720) ? 0.5 : .5;
			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			int fontThickness = 1;
			cv::Point textCenter;
			cv::Scalar textColor = cv::Scalar(0, 255, 0);	// green
			cv::Scalar landmarks_color(0, 0, 255);			// blue

			// get gaze direction and head rotation
			NvAR_Quaternion* headPose3D = gaze_ar_engine.getPose();
			NvAR_Point3f* gazeDirection3D = gaze_ar_engine.getGazeDirectionPoints();
			std::vector<cv::Point2f> gazeDirection2D = { {0,0}, {0,0} };	// {{ origin.x, origin.y}, {direction.x, direction.y}}
			cv::Point headPose2D = { 0,0 };									// { origin.x, origin.y }

			gazePitch = gaze_ar_engine.gaze_angles_vector[0] * DEGREES_PER_RADIAN;
			gazeYaw = gaze_ar_engine.gaze_angles_vector[1] * DEGREES_PER_RADIAN;

			gaze3D_X = gaze_ar_engine.gaze_direction[0].x;
			gaze3D_Y = gaze_ar_engine.gaze_direction[0].y;
			gaze3D_Z = gaze_ar_engine.gaze_direction[0].z;

			head3D_X = gaze_ar_engine.head_translation[0];
			head3D_Y = gaze_ar_engine.head_translation[1];
			head3D_Z = gaze_ar_engine.head_translation[2];

			// get gaze information in a 2D plane using data in 3D space
			// only write to frame if the drawTracking flag is set to true
			gazeDirection2D = gaze_ar_engine.DrawEstimatedGaze(frame, FLAG_drawTracking, float(FLAG_vectorScale));
			if (headPose3D) headPose2D = gaze_ar_engine.DrawPose(frame, headPose3D, FLAG_drawTracking, float(FLAG_vectorScale));

			// send data to TouchDesigner over osc
			if (FLAG_sendOsc) osc::SendGazeData(head3D_X, head3D_Y, head3D_Z, headPose2D, gazeDirection3D, gazeDirection2D);

			// draw keypoints, bbox, and gaze info to window on top of video
			if (FLAG_drawTracking) {
				// display head translation data
				textCenter = cv::Point(80, 80);
				snprintf(buf, sizeof(buf), "head translation: (%.1f, %.1f, %.1f)", head3D_X, head3D_Y, head3D_Z);
				cv::putText(frame, buf, textCenter, fontFace, fontScale, textColor, fontThickness);

				// display gaze angle data
				textCenter = cv::Point(80, 110);
				snprintf(buf, sizeof(buf), "gaze angles: %.1f (pitch), %.1f (yaw)", gazePitch, gazeYaw);
				cv::putText(frame, buf, textCenter, fontFace, fontScale, textColor, fontThickness);

				// Display landmarks and bounding box 
				DrawLandmarkPoints(frame, gaze_ar_engine.getLandmarks(), gaze_ar_engine.getNumLandmarks(), &landmarks_color);
				drawBBoxes(frame, gaze_ar_engine.getLargestBox());
			}
		}
	}
	if (offlineMode) {
		gazeRedirectOutputVideo.write(frame);
	}
	frameTimer.resume();

	// If captureOutputs has been set to true, save the output frames.
	if (FLAG_captureOutputs) {
		writeVideo(frame);
	}
	return doErr;
}

/********************************************************************************
 * init processes
 ********************************************************************************/

GazeTrack::Err GazeTrack::initGazeEngine(const char* modelPath, bool isNumLandmarks126, bool gazeRedirect, unsigned eyeSizeSensitivity, bool useCudaGraph) {
	// exit if there is no camera or video file for video source
	if (!(FLAG_videoSource == sharedMemory) && !cap.isOpened())
		return errVideo;

	int numLandmarkPoints = isNumLandmarks126 ? LANDMARKS_126 : LANDMARKS_68;
	gaze_ar_engine.setNumLandmarks(numLandmarkPoints);
	gaze_ar_engine.setGazeRedirect(gazeRedirect);
	gaze_ar_engine.setUseCudaGraph(useCudaGraph);
	gaze_ar_engine.setEyeSizeSensitivity(eyeSizeSensitivity);
	nvErr = gaze_ar_engine.createGazeRedirectionFeature(modelPath);

	if (!offlineMode)
		cv::namedWindow(windowTitle, 1);

	frameIndex = 0;

	return doAppErr(nvErr);
}

GazeTrack::Err GazeTrack::initSharedMemory()
{
	// Convert string name to wchar_t
	int strLen = (int)FLAG_sharedMemName.length() + 1;
	int wideCharLen = MultiByteToWideChar(CP_ACP, 0, FLAG_sharedMemName.c_str(), strLen, 0, 0);
	wchar_t* wideCharName = new wchar_t[wideCharLen];
	MultiByteToWideChar(CP_ACP, 0, FLAG_sharedMemName.c_str(), strLen, wideCharName, wideCharLen);
	std::wstring smName(wideCharName);
	delete[] wideCharName;
	printf("Shared Mem Name: %s\n\n", FLAG_sharedMemName.c_str());

	// init touchdesigner shared memory w/ name from command line
	shm = new UT_SharedMem(ShmString(smName));
	// make sure we can get at shared memory
	if (shm->getErrorState() != UT_SHM_ERR_NONE) {
		printf("No shared memory in initialization\n");
		return errSharedMem;
	}
	else printf("Created a UT_SharedMem successfully!\n");

	// lock test
	if (!shm->tryLock(5000)) {
		printf("Lock test FAILED to get a lock on shared memory!\n");
		printf("Try closing and reopening TouchDesigner to restart shared mem\n");
		return errSharedMem;
	}
	else {
		printf("Lock test succeded!\n");
		shmTopHeader = (TOP_SharedMemHeader*)shm->getMemory();

		shmWidth = shmTopHeader->width;
		shmHeight = shmTopHeader->height;

		printf("Shared memory (w,h) = (%d,%d)\n\n", shmWidth, shmHeight);
		shm->unlock();

		shmTempFrame = cv::Mat(shmHeight, shmWidth, CV_8UC4);			// for converting image data from shared mem to frame

		inputHeight = shmHeight;
		inputWidth = shmWidth;
		gaze_ar_engine.setInputImageWidth(shmWidth);
		gaze_ar_engine.setInputImageHeight(shmHeight);

		return errNone;
	}
}

GazeTrack::Err GazeTrack::initCamera(const char* camRes, unsigned int camID) {
	if (cap.open(camID)) {
		if (camRes) {
			int n;
			n = sscanf(camRes, "%d%*[xX]%d", &inputWidth, &inputHeight);
			switch (n) {
			case 2:
				break;  // We have read both width and height
			case 1:
				inputHeight = inputWidth;
				inputWidth = (int)(inputHeight * (4. / 3.) + .5);
				break;
			default:
				inputHeight = 0;
				inputWidth = 0;
				break;
			}

			if (inputWidth) cap.set(CV_CAP_PROP_FRAME_WIDTH, inputWidth);
			if (inputHeight) cap.set(CV_CAP_PROP_FRAME_HEIGHT, inputHeight);

			inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
			inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

			//// openCV API(CAP_PROP_FRAME_WIDTH) to get camera resolution is not always reliable with some cameras
			//cap >> frame;
			//if (frame.empty()) return errCamera;
			//if (inputWidth != frame.cols || inputHeight != frame.rows) {
			//	std::cout << "!!! warning: openCV API(CAP_PROP_FRAME_WIDTH/CV_CAP_PROP_FRAME_HEIGHT) to get camera resolution is not trustable. Using the resolution from the actual frame" << std::endl;
			//	inputWidth = frame.cols;
			//	inputHeight = frame.rows;
			//}

			gaze_ar_engine.setInputImageWidth(inputWidth);
			gaze_ar_engine.setInputImageHeight(inputHeight);
		}
	}
	else
		return errCamera;
	return errNone;
}

GazeTrack::Err GazeTrack::initOfflineMode(const char* inputFilename, const char* outputFilename) {
	if (cap.open(inputFilename)) {
		inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
		inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		gaze_ar_engine.setInputImageWidth(inputWidth);
		gaze_ar_engine.setInputImageHeight(inputHeight);
	}
	else {
		printf("ERROR: Unable to open the input video file \"%s\" \n", inputFilename);
		return Err::errVideo;
	}
	std::string fdOutputVideoName, fldOutputVideoName, ffOutputVideoName, fgzOutputVideoName;
	std::string outputFilePrefix;
	if (outputFilename && strlen(outputFilename) != 0) {
		outputFilePrefix = outputFilename;
	}
	else {
		size_t lastindex = std::string(inputFilename).find_last_of(".");
		outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
		outputFilePrefix = outputFilePrefix + "_gaze.mp4";
	}
	fgzOutputVideoName = outputFilePrefix;

	if (!gazeRedirectOutputVideo.open(fgzOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
		cv::Size(inputWidth, inputHeight))) {
		printf("ERROR: Unable to open the output video file \"%s\" \n", fgzOutputVideoName.c_str());
		return Err::errGeneral;
	}

	FLAG_captureOutputs = true;
	return Err::errNone;
}

void GazeTrack::initCudaGPU() {
	// If the system has multiple supported GPUs then the application
	// should use CUDA driver APIs or CUDA runtime APIs to enumerate
	// the GPUs and select one based on the application's requirements

	if (FLAG_useCudaGraph) {
		printf("Chosen GPU: %d\n", FLAG_chosenGPU);
		cudaError_t err = cudaSetDevice(FLAG_chosenGPU);
	}
}

/********************************************************************************
 * main
 ********************************************************************************/

void GazeTrack::printArgsToConsole() {
	// print args to console for debug
	printf("Use CUDA Graph: %s\n", FLAG_useCudaGraph ? "true" : "false");
	printf("Chosen GPU: %d\n", FLAG_chosenGPU);
	printf("Enable temporal optimizations: %d\n", FLAG_temporalSmoothing);
	printf("Eye Size Sensitivity: %d\n", FLAG_eyeSizeSensitivity);
	printf("Vector Scale: %d\n", FLAG_vectorScale);
	switch (FLAG_videoSource) {
	case webcam:
		printf("Video Source: Webcam\n");
		printf("Webcam Index: %d\n", FLAG_camIndex);
		printf("Webcam Resolution: %s\n", FLAG_camRes.c_str());
		printf("Draw Window: %s\n", FLAG_drawWindow ? "true" : "false");
		printf("Draw Tracking Info: %s\n", FLAG_drawTracking ? "true" : "false");
		printf("Draw FPS: %s\n", FLAG_drawFPS ? "true" : "false");
		break;
	case sharedMemory:
		printf("Video Source: Shared Memory\n");
		printf("Draw Window: %s\n", FLAG_drawWindow ? "true" : "false");
		printf("Draw Tracking Info: %s\n", FLAG_drawTracking ? "true" : "false");
		printf("Draw FPS: %s\n", FLAG_drawFPS ? "true" : "false");
		break;
	case videoFile:
		printf("Video Source: Video File\n");
		printf("Capture Outputs: %s\n", FLAG_captureOutputs ? "true" : "false");
		printf("Capture Codec: %s\n", FLAG_captureCodec.c_str());
		break;
	default:
		printf("Video Source: UNKNOWN\n");
		break;
	}
	printf("Send OSC data: %s\n", FLAG_sendOsc ? "true" : "false");
	if (FLAG_sendOsc) {
		printf("Status Port: %d\n", FLAG_statusPort);
		printf("Keypoints Port: %d\n", FLAG_keypointsPort);
	}
}

int main(int argc, char** argv) {
	// Parse the arguments
	if (0 != ParseMyArgs(argc, argv)) return -100;

	GazeTrack app;
	GazeTrack::Err doErr = GazeTrack::Err::errNone;

	// start initializing the tracking
	app.gaze_ar_engine.setFaceStabilization(FLAG_temporalSmoothing);

	// debug args read from batch file
	app.printArgsToConsole();

	// initialize video source 
	offlineMode = FLAG_videoSource == videoFile;
	if (offlineMode) {
		// offline mode is only when video source is a file on disk
		if (FLAG_inFilePath.empty()) {
			doErr = GazeTrack::errMissing;
			printf("ERROR: %s, please specify input file using --in \n", app.errorStringFromCode(doErr));
			goto bail;
		}
		doErr = app.initOfflineMode(FLAG_inFilePath.c_str(), FLAG_outFilePath.c_str());
	}
	else {
		// initialize webcam or shared memory
		if (FLAG_videoSource == webcam)
			doErr = app.initCamera(FLAG_camRes.c_str(), FLAG_camIndex);
		else if (FLAG_videoSource == sharedMemory)
			doErr = app.initSharedMemory();
	}
	BAIL_IF_ERR(doErr);
	printf(".\nInitialized video stream!\n");

	// initialize GazeEngine class based on model and landmark detailed
	doErr = app.initGazeEngine(FLAG_modelPath.c_str(), FLAG_useDetailedLandmarks, GAZE_REDIRECT, FLAG_eyeSizeSensitivity, FLAG_useCudaGraph);
	BAIL_IF_ERR(doErr);
	printf("Initialized GazeEngine!\n");

	// initialize osc communication
	if (FLAG_sendOsc) {
		osc::InitOSC();
		printf("Initialized OSC communication!\n");
	}

	// start analyzing the video each frame
	printf("Starting to run...\n");
	doErr = app.run();
	BAIL_IF_ERR(doErr);

bail:
	if (doErr) printf("ERROR: %s\n", app.errorStringFromCode(doErr));
	app.stop();
	return (int)doErr;
}

GazeTrack::Err GazeTrack::run() {
	GazeTrack::Err doErr = errNone;
	GazeEngine::Err err = gaze_ar_engine.initGazeRedirectionIOParams();
	if (err != GazeEngine::Err::errNone) return doAppErr(err);

	while (1) {
		// get frame based on desired video source
		// "frame" is a global variable shared among functions
		if (FLAG_videoSource == webcam || FLAG_videoSource == videoFile)
			doErr = acquireWebcamOrVideoFrame();
		else if (FLAG_videoSource == sharedMemory)
			doErr = acquireSharedMemFrame();

		// We have reached the end of the video so return without any error.
		if (frame.empty() && offlineMode) return GazeTrack::errNone;
		else if (doErr != GazeTrack::errNone) return doErr;

		// get joint keypoint data and bounding box data
		gazeRedirectFrame.create(inputHeight, inputWidth, frame.type());
		doErr = acquireGazeRedirection();
		if (GazeTrack::errCancel == doErr || GazeTrack::errVideo == doErr) return doErr;

		// evaluate fps and send out fps over oscc
		getFPS();
		sendFPS();

		// write frame to output file or to window
		if (FLAG_drawWindow && !offlineMode) {
			if (!frame.empty()) {
				if (FLAG_drawFPS)
					drawFPS(frame);
				cv::imshow(windowTitle, frame);
			}
			// we need this cv::waitKey to allow the window to refresh
			int n = cv::waitKey(1);
			/*
				if (n >= 0) {
					static const int ESC_KEY = 27;
					switch (n) {
					case ESC_KEY:
						break;
					case '3':
						gaze_ar_engine.destroyGazeRedirectionFeature();
						gaze_ar_engine.createGazeRedirectionFeature(FLAG_modelPath.c_str());
						gaze_ar_engine.initGazeRedirectionIOParams();
						break;
					default:
						break;
					}
				}
			*/
		}
	}
	return doErr;
}

void GazeTrack::stop() {
	gaze_ar_engine.destroyGazeRedirectionFeature();

	if (offlineMode) {
		gazeRedirectOutputVideo.release();
	}
	cap.release();

	cv::destroyAllWindows();
}