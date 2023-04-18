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
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>

// nvidia maxine
#include "BodyEngine.h"
#include "BodyTrack.h"
#include "RenderingUtils.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"

// touchdesigner shared memory
#include "SharedMem/TOP_SharedMemHeader.h"
#include "SharedMem/UT_Mutex.h"
#include "SharedMem/UT_SharedMem.h"

// helper classes
#include "Timer.h"

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#endif

/********************************************************************************
 * constant global values
 ********************************************************************************/

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

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

const int PELVIS = 0;
const int LEFT_HIP = 1;
const int RIGHT_HIP = 2;
const int TORSO = 3;
const int LEFT_KNEE = 4;
const int RIGHT_KNEE = 5;
const int NECK = 6;
const int LEFT_ANKLE = 7;
const int RIGHT_ANKLE = 8;
const int LEFT_BIG_TOE = 9;
const int RIGHT_BIG_TOE = 10;
const int LEFT_SMALL_TOE = 11;
const int RIGHT_SMALL_TOE = 12;
const int LEFT_HEEL = 13;
const int RIGHT_HEEL = 14;
const int NOSE = 15;
const int LEFT_EYE = 16;
const int RIGHT_EYE = 17;
const int LEFT_EAR = 18;
const int RIGHT_EAR = 19;
const int LEFT_SHOULDER = 20;
const int RIGHT_SHOULDER = 21;
const int LEFT_ELBOW = 22;
const int RIGHT_ELBOW = 23;
const int LEFT_WRIST = 24;
const int RIGHT_WRIST = 25;
const int LEFT_PINKY_KNUCKLE = 26;
const int RIGHT_PINKY_KNUCKLE = 27;
const int LEFT_MIDDLE_TIP = 28;
const int RIGHT_MIDDLE_TIP = 29;
const int LEFT_INDEX_KNUCKLE = 30;
const int RIGHT_INDEX_KNUCKLE = 31;
const int LEFT_THUMB_TIP = 32;
const int RIGHT_THUMB_TIP = 33;

const int NUM_KEYPOINTS = 34;

// Batch Size has to be 8 when people tracking is enabled
const unsigned int PEOPLE_TRACKING_BATCH_SIZE = 8;

enum {
	myErrNone = 0,
	myErrShader = -1,
	myErrProgram = -2,
	myErrTexture = -3,
};

// colors for drawing to window
static const cv::Scalar cv_colors[] = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) };
enum {
	kColorRed = 0,
	kColorGreen = 1,
	kColorBlue = 2
};

// video source for frame data
enum {
	webcam = 0,
	videoFile = 1,
	sharedMemory = 2
};

// mode
enum {
	highQuality = 0,
	highPerformance = 1
};

// app mode
enum {
	bodyDetection = 0,			// bounding box only
	bodyPoseDetection = 1		// bounding box and keypoints
};

// style guide for drawing bounding box, skeleton joints, and text
const auto CIRCLE_COLOR = cv::Scalar(180, 180, 180);		// grey
const int CIRCLE_RADIUS = 4;
const int CIRCLE_THICKNESS = -1;							// filled
const auto RECT_COLOR = cv::Scalar(255, 0, 0);				// red
const int RECT_THICKNESS = 2;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int ID_FONT_THICKNESS = 2;
const double ID_FONT_SCALE = 0.5;
const int ID_TEXT_OFFSET_Y = -10;
const auto DEBUG_TEXT_COLOR = cv::Scalar(255, 255, 255);	// white
const int DEBUG_TEXT_SCALE = 1;
const int DEBUG_FONT_THICKNESS = 1;

// for fps calculation
const float TIME_CONSTANT = 16.f;
const float FRAMETIME_THRESHOLD = 100.f;

// Made the executive decision that we only want to track multiple people and not limit outselves to only one user at a time
const bool ENABLE_MULTI_PEOPLE_TRACKING = true;
// Made the executive decision that we default to keypoint detection, which includes body detection
const unsigned int APP_MODE = bodyPoseDetection;
// Optimizes the results for temporal input frames. Not possible for multi-person tracking, so must be false.
const bool TEMPORAL_SMOOTHING = false;

// shared memory
UT_SharedMem* shm;						// shared memory data
TOP_SharedMemHeader* shmTopHeader;		// TOP_SharedMemHeader.h header data
cv::Mat shmTempFrame;					// for converting touch shared mem to opencv mat
int shmWidth, shmHeight;				// width and height of image from shared memory

// for writing frame and video data to file on disk
bool captureFrame = false, captureVideo = false;
// specifies whether to use offline video or an online camera video as the input. 
// when offline mode is true --> Disables webcam, reads video from file, and writes output video results to file
bool offlineMode = false;				// False: Webcam or SharedMem, True: Video file

// designates model selection within modelPath folder based on mode and appMode
std::string bodyModel;					// set automatically based on PC hardware

/********************************************************************************
 * command-line arguments
 ********************************************************************************/

bool
FLAG_drawTracking = true,				// Draw keypoint and bounding box data to window
FLAG_drawWindow = true,					// Draw window with video feed to desktop
FLAG_drawFPS = true,					// Write FPS debug information to window
FLAG_captureOutputs = false,			// Enables video/image capture and writing body detection/keypoints outputs to file. If input is video file, gets set to true
FLAG_debug = false,						// Print debugging information to the console
FLAG_verbose = false,					// Print extra information to the console
FLAG_useCudaGraph = true;				// Uses CUDA Graphs to improve performance. CUDA graph reduces the overhead of the GPU operation submission of 3D body tracking

std::string
FLAG_inFilePath,						// input file path on disk for video source (path + name + prefix)
FLAG_outFilePrefix,						// output file prefix for writing data to disk (path + name but should not include file time, like .mp4)
FLAG_camRes,							// If offlineMode=false, specifies the cam res. If width omitted, width is computed from height to give an aspect ratio of 4:3.
FLAG_captureCodec = "avc1",				// avc1 = h264
FLAG_modelPath = "C:/Program Files/NVIDIA Corporation/NVIDIA AR SDK/models",	// default installation location
FLAG_sharedMemName = "TOPshm";

unsigned int
FLAG_videoSource = webcam,				// Specify video source. 0: Webcam, 1: Video File, 2: Shared Mem (TouchDesigner).
FLAG_mode = highPerformance,			// 0: High Quality, 1: High Performance -> default to high performance
FLAG_chosenGPU = 0,						// Index of GPU to run the Maxine executable
FLAG_camIndex = 0,						// Index of webcam connected to the PC
FLAG_shadowTrackingAge = 90,			// Sets the Shadow Tracking Age, after which tracking information for a person is removed. The default value is 90 (frames).
FLAG_probationAge = 10,					// Sets the Probation Age, after which tracking information for a person is addedd. The default value is 10 (frames).
FLAG_maxTargetsTracked = 30;			// Sets the Maxinum Targets Tracked. The default value is 30, and the minimum is value is 1.

/********************************************************************************
 * parsing command line args
 ********************************************************************************/

static void Usage() {
	printf(
		"BodyTrack [<args> ...]\n"
		"where <args> is:\n"
		" --mode[=0|1]							Model Mode. 0: High Quality, 1: High Performance. Default is 1.\n"
		" --draw_tracking[=(true|false)]		Draw tracking information (joints, bbox) on top of frame. Default is true.\n"
		" --draw_window[=(true|false)]			Draw video feed to window on desktop. Default is true.\n"
		" --draw_fps[=(true|false)]				Draw FPS debug information on top of frame. Default is true.\n"
		" --use_cuda_graph[=(true|false)]		Enable faster execution by using cuda graph to capture engine execution. Default is true.\n"
		" --chosen_gpu[=(0|1|2|3|..)]			GPU index for running the Maxine instance. Default is 0.\n"
		" --video_source[=(0|1|2)]				Specify video source. 0: Webcam, 1: Video File, 2: Shared Mem (TD). Default is 0.\n"
		" --cam_index[=(0|1|2|3|..)]			Specify the webcam index we want to use for the video feed. Default is 0.\n"
		" --shared_mem_name=<string>			Specify the string name for Shared Memory from TouchDesigner. Default is 'TOPshm'.\n"
		" --capture_outputs[=(true|false)]		Enables video/image capture and writing data to file. Default is false.\n"
		" --cam_res=[WWWx]HHH					Specify webcam resolution as height or width x height (--cam_res=640x480 or --cam_res=480). Default is empty string.\n"
		" --in_file_path=<file>					Specify the input file. Default is empty string.\n"
		" --out_file_prefix=<file>				Specify the output file name (no extension like .mp4). Default is empty string.\n"
		" --codec=<fourcc>						FOURCC code for the desired codec. Default is H264 (avc1).\n"
		" --model_path=<path>					Specify the directory containing the TRT models.\n"
		" --shadow_tracking_age=<int>			Shadow Tracking Age (frames), after which tracking info of a person is removed. Default is 90.\n"
		" --probation_age=<int>					Length of probationary period (frames), after which tracking info of a person is added. Default is 10.\n"
		" --max_targets_tracked=<int>			Maximum number of targets to be tracked. Default is 30, min value is 1.\n"
		" --debug[=(true|false)]				Report debugging timing info to console. Default is false.\n"
		" --verbose[=(true|false)]				Report keypoint joint info to console. Default is false.\n"
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
	int errs = 0;
	for (--argc, ++argv; argc--; ++argv) {
		bool help;
		const char* arg = *argv;
		if (arg[0] != '-') {
			continue;
		}
		else if ((arg[1] == '-') && (
			GetFlagArgVal("mode", arg, &FLAG_mode) ||
			GetFlagArgVal("draw_tracking", arg, &FLAG_drawTracking) ||
			GetFlagArgVal("draw_window", arg, &FLAG_drawWindow) ||
			GetFlagArgVal("draw_fps", arg, &FLAG_drawFPS) ||
			GetFlagArgVal("use_cuda_graph", arg, &FLAG_useCudaGraph) ||
			GetFlagArgVal("chosen_gpu", arg, &FLAG_chosenGPU) ||
			GetFlagArgVal("video_source", arg, &FLAG_videoSource) ||
			GetFlagArgVal("cam_index", arg, &FLAG_camIndex) ||
			GetFlagArgVal("shared_mem_name", arg, &FLAG_sharedMemName) ||
			GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||
			GetFlagArgVal("cam_res", arg, &FLAG_camRes) ||
			GetFlagArgVal("in_file_path", arg, &FLAG_inFilePath) ||
			GetFlagArgVal("out_file_prefix", arg, &FLAG_outFilePrefix) ||
			GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||
			GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
			GetFlagArgVal("shadow_tracking_age", arg, &FLAG_shadowTrackingAge) ||
			GetFlagArgVal("probation_age", arg, &FLAG_probationAge) ||
			GetFlagArgVal("max_targets_tracked", arg, &FLAG_maxTargetsTracked) ||
			GetFlagArgVal("debug", arg, &FLAG_debug) ||
			GetFlagArgVal("verbose", arg, &FLAG_verbose)
			)) {
			continue;
		}
		else if (GetFlagArgVal("help", arg, &help)) {
			Usage();
		}
		else if (arg[1] != '-') {
			for (++arg; *arg; ++arg) {
				if (*arg == 'v') {
					FLAG_verbose = true;
				}
				else {
					// printf("Unknown flag: \"-%c\"\n", *arg);
				}
			}
			continue;
		}
		else {
			// printf("Unknown flag: \"%s\"\n", arg);
		}
	}
	return errs;
}

/********************************************************************************
 * error codes
 ********************************************************************************/

const char* BodyTrack::errorStringFromCode(BodyTrack::Err code) {
	struct LUTEntry {
		Err code;
		const char* str;
	};
	static const LUTEntry lut[] = {
		{errNone, "no error (errNone)"},
		{errGeneral, "an error has occured (errGeneral)"},
		{errRun, "an error has occured while the feature is running (errRun)"},
		{errInitialization, "Initializing Body Engine failed (errInitialization)"},
		{errRead, "an error has occured while reading a file (errRead)"},
		{errEffect, "an error has occured while creating a feature (errEffect)"},
		{errParameter, "an error has occured while setting a parameter for a feature (errParameter)"},
		{errUnimplemented, "the feature is unimplemented (errUnimplemented)"},
		{errMissing, "missing input parameter (errMissing)"},
		{errVideo, "no video source has been found (errVideo)"},
		{errImageSize, "the image size cannot be accommodated (errImageSize)"},
		{errNotFound, "the item cannot be found (errNotFound)"},
		{errBodyModelInit, "body model initialization failed (errBodyModelInit)"},
		{errGLFWInit, "GLFW initialization failed (errGLFWInit)"},
		{errGLInit, "OpenGL initialization failed (errGLInit)"},
		{errRendererInit, "renderer initialization failed (errRendererInit)"},
		{errGLResource, "an OpenGL resource could not be found (errGLResource)"},
		{errGLGeneric, "an otherwise unspecified OpenGL error has occurred (errGLGeneric)"},
		{errBodyFit, "an error has occurred while body fitting (errBodyFit)"},
		{errNoBody, "no body has been found (errNoBody)"},
		{errSDK, "an SDK error has occurred (errSDK)"},
		{errCuda, "a CUDA error has occurred (errCuda)"},
		{errCancel, "the user cancelled (errCancel)"},
		{errCamera, "unable to connect to the camera (errCamera)"},
		{errSharedMemLock, "could not lock shared memory (errSharedMemLock)"},
		{errSharedMemSeg, "invalid shared memory segment (errSharedMemSeg)"},
		{errSharedMemVideo, "could not read frame from shared memory (errSharedMemVideo)"}
	};
	for (const LUTEntry* p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
		if (p->code == code) return p->str;
	static char msg[18];
	snprintf(msg, sizeof(msg), "error #%d", code);
	return msg;
}

/********************************************************************************
 * CUDA GPU selection
 ********************************************************************************/

int chooseGPU() {
	// If the system has multiple supported GPUs then the application
	// should use CUDA driver APIs or CUDA runtime APIs to enumerate
	// the GPUs and select one based on the application's requirements

	if (FLAG_useCudaGraph) {
		printf("Chosen GPU: %d\n", FLAG_chosenGPU);
		// TODO: set CUDA device:
		//cudaError_t err = cudaSetDevice(FLAG_chosenGPU);
		return 0;
	}
	else {
		return 0;
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
	// calendarTime << std::put_time(
	//     &brokenTime,
	//     "%Y-%m-%d-%H-%M-%S");  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
	char time_string[24];
	if (0 < strftime(time_string, sizeof(time_string), "%Y-%m-%d-%H-%M-%S] ", &brokenTime))
		calendarTime << time_string;  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
	// Get the time since epoch 0(Thu Jan  1 00:00:00 1970) and the remainder after division is
	// our milliseconds
	std::chrono::milliseconds currentMilliseconds =
		std::chrono::duration_cast<std::chrono::milliseconds>(currentTimePoint.time_since_epoch()) % 1000;
	// Append the milliseconds to the stream
	calendarTime << "-" << std::setfill('0') << std::setw(3) << currentMilliseconds.count();  // milliseconds
	return calendarTime.str();
}

/********************************************************************************
 * BodyTrack class
 ********************************************************************************/

BodyTrack* gApp = nullptr;
const char BodyTrack::windowTitle[] = "Envoy Maxine BodyTrack";
char* g_nvARSDKPath = NULL;

BodyTrack::BodyTrack() {
	// Make sure things are initialized properly
	gApp = this;
	frameTime = 0;
	frameIndex = 0;
	nvErr = BodyEngine::errNone;
	scaleOffsetXY[0] = scaleOffsetXY[2] = 1.f;
	scaleOffsetXY[1] = scaleOffsetXY[3] = 0.f;
}

BodyTrack::~BodyTrack() {}

/********************************************************************************
 * draw joints, bones, boxes, data to window
 ********************************************************************************/

void BodyTrack::DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_TrackingBBoxes* output_bbox) {
	NvAR_Point2f* pt, * endPt;
	NvAR_Point2f* keypointsBatch8 = keypoints;
	cv::Point circleCenter, rectPoint1, rectPoint2, textCenter;
	float x, y, width, height;
	int trackingID;

	// get frame and user data
	cv::Mat frm = (offlineMode) ? src.clone() : src;
	int numTrackedUsers = body_ar_engine.output_tracking_bboxes.num_boxes;

	// draw bounding box and id number for each tracked user 
	for (int i = 0; i < numTrackedUsers; i++) {

		// get a user's batch of 34 joints
		keypoints = keypointsBatch8 + (i * NUM_KEYPOINTS);

		for (endPt = (pt = (NvAR_Point2f*)keypoints) + numKeyPoints; pt < endPt; ++pt) {
			circleCenter = cv::Point(lround(pt->x), lround(pt->y));
			cv::circle(frm, circleCenter, CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS);
		}

		if (output_bbox) {
			// extract bounding box data
			x = output_bbox->boxes[i].bbox.x;
			y = output_bbox->boxes[i].bbox.y;
			width = output_bbox->boxes[i].bbox.width;
			height = output_bbox->boxes[i].bbox.height;
			trackingID = output_bbox->boxes[i].tracking_id;

			while (colorCodes.size() <= trackingID)
				colorCodes.push_back(cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF));

			// draw bounding box
			auto rectColor = colorCodes[trackingID];
			rectPoint1 = cv::Point(lround(x), lround(y));
			rectPoint2 = cv::Point(lround(x + width), lround(y + height));
			cv::rectangle(frm, rectPoint1, rectPoint2, rectColor, RECT_THICKNESS);

			// draw id number
			std::string text = "ID: " + std::to_string(trackingID);
			textCenter = cv::Point(lround(x), lround(y) + ID_TEXT_OFFSET_Y);
			cv::putText(frm, text, textCenter, FONT_FACE, ID_FONT_SCALE, rectColor, ID_FONT_THICKNESS);
		}

		// draw joint + bone data
		DrawKeyPointLine(frm, keypoints);
	}

	// write output video to file
	if (offlineMode)
		keyPointsOutputVideo.write(frm);
}

void BodyTrack::DrawKeyPointLine(cv::Mat& frm, NvAR_Point2f* keypoints) {
	// center body
	drawKeyPointLine(frm, keypoints, PELVIS, TORSO, kColorGreen);
	drawKeyPointLine(frm, keypoints, TORSO, NECK, kColorGreen);
	drawKeyPointLine(frm, keypoints, NECK, PELVIS, kColorGreen);

	// right side
	drawKeyPointLine(frm, keypoints, RIGHT_ANKLE, RIGHT_KNEE, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_KNEE, RIGHT_HIP, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_HIP, PELVIS, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_HIP, RIGHT_SHOULDER, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_ELBOW, RIGHT_WRIST, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_SHOULDER, NECK, kColorRed);

	// right side hand and feet
	drawKeyPointLine(frm, keypoints, RIGHT_WRIST, RIGHT_PINKY_KNUCKLE, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_WRIST, RIGHT_MIDDLE_TIP, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_WRIST, RIGHT_INDEX_KNUCKLE, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_WRIST, RIGHT_THUMB_TIP, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_ANKLE, RIGHT_HEEL, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_ANKLE, RIGHT_BIG_TOE, kColorRed);
	drawKeyPointLine(frm, keypoints, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, kColorRed);

	//left side
	drawKeyPointLine(frm, keypoints, LEFT_ANKLE, LEFT_KNEE, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_KNEE, LEFT_HIP, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_HIP, PELVIS, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_HIP, LEFT_SHOULDER, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_SHOULDER, LEFT_ELBOW, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_ELBOW, LEFT_WRIST, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_SHOULDER, NECK, kColorBlue);

	// left side hand and feet
	drawKeyPointLine(frm, keypoints, LEFT_WRIST, LEFT_PINKY_KNUCKLE, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_WRIST, LEFT_MIDDLE_TIP, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_WRIST, LEFT_INDEX_KNUCKLE, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_WRIST, LEFT_THUMB_TIP, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_ANKLE, LEFT_HEEL, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_ANKLE, LEFT_BIG_TOE, kColorBlue);
	drawKeyPointLine(frm, keypoints, LEFT_BIG_TOE, LEFT_SMALL_TOE, kColorBlue);

	// head
	drawKeyPointLine(frm, keypoints, NECK, NOSE, kColorGreen);
	drawKeyPointLine(frm, keypoints, NOSE, RIGHT_EYE, kColorGreen);
	drawKeyPointLine(frm, keypoints, RIGHT_EYE, RIGHT_EAR, kColorGreen);
	drawKeyPointLine(frm, keypoints, NOSE, LEFT_EYE, kColorGreen);
	drawKeyPointLine(frm, keypoints, LEFT_EYE, LEFT_EAR, kColorGreen);
}

void BodyTrack::drawKeyPointLine(const cv::Mat& src, NvAR_Point2f* keypoints, int joint1, int joint2, int color) {
	// extract line start + end data
	NvAR_Point2f point1_pos = *(keypoints + joint1);
	NvAR_Point2f point2_pos = *(keypoints + joint2);
	cv::Point point1 = cv::Point((int)point1_pos.x, (int)point1_pos.y);
	cv::Point point2 = cv::Point((int)point2_pos.x, (int)point2_pos.y);
	// draw line
	cv::line(src, point1, point2, cv_colors[color], RECT_THICKNESS);
}

/********************************************************************************
 * write data to file
 ********************************************************************************/

void BodyTrack::writeEstResults(std::ofstream& outputFile, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
	/**
	 * Output File Format :
	 * BodyDetectOn, KeyPointDetectOn
	 * kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumKPs, [j_x, j_y]{kNumKPs}
	 */

	int bodyDetectOn = (body_ar_engine.appMode == BodyEngine::mode::bodyDetection ||
		body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
		? 1
		: 0;
	int keyPointDetectOn = (body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
		? 1
		: 0;
	outputFile << bodyDetectOn << "," << keyPointDetectOn << "\n";

	if (bodyDetectOn && output_bboxes.num_boxes) {
		// Append number of bodies detected in the current frame
		outputFile << unsigned(output_bboxes.num_boxes) << ",";
		// write outputbboxes to outputFile
		for (size_t i = 0; i < output_bboxes.num_boxes; i++) {
			int x1 = (int)output_bboxes.boxes[i].bbox.x, y1 = (int)output_bboxes.boxes[i].bbox.y,
				width = (int)output_bboxes.boxes[i].bbox.width, height = (int)output_bboxes.boxes[i].bbox.height;
			unsigned int tracking_id = output_bboxes.boxes[i].tracking_id;
			outputFile << x1 << "," << y1 << "," << width << "," << height << "," << tracking_id << ",";
		}
	}
	else {
		outputFile << "0,";
	}
	if (keyPointDetectOn && output_bboxes.num_boxes) {
		int numKeyPoints = body_ar_engine.getNumKeyPoints();
		// Append number of keypoints
		outputFile << numKeyPoints << ",";
		// Append 2 * number of keypoint values
		NvAR_Point2f* pt, * endPt;
		for (endPt = (pt = (NvAR_Point2f*)keypoints) + numKeyPoints; pt < endPt; ++pt)
			outputFile << pt->x << "," << pt->y << ",";
	}
	else {
		outputFile << "0,";
	}

	outputFile << "\n";
}

void BodyTrack::writeVideoAndEstResults(const cv::Mat& frm, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
	if (captureVideo) {
		if (!capturedVideo.isOpened()) {
			const std::string currentCalendarTime = getCalendarTime();
			const std::string capturedOutputFileName = currentCalendarTime + ".mp4";
			getFPS();
			if (frameTime) {
				float fps = (float)(1.0 / frameTime);
				capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps, cv::Size(frm.cols, frm.rows));
				if (!capturedVideo.isOpened()) {
					std::cout << "Error: Could not open video: \"" << capturedOutputFileName << "\"\n";
					return;
				}
				printf("Capturing video started...\n");
			}
			else {  // If frameTime is 0.f, returns without writing the frame to the Video
				return;
			}
			const std::string outputsFileName = currentCalendarTime + ".txt";
			bodyEngineVideoOutputFile.open(outputsFileName, std::ios_base::out);
			if (!bodyEngineVideoOutputFile.is_open()) {
				std::cout << "Error: Could not open file: \"" << outputsFileName << "\"\n";
				return;
			}
			std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
			bodyEngineVideoOutputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n ";
			bodyEngineVideoOutputFile << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
		}
		// Write each frame to the Video
		capturedVideo << frm;
		writeEstResults(bodyEngineVideoOutputFile, output_bboxes, keypoints);
	}
	else {
		if (capturedVideo.isOpened()) {
			printf("Capturing video ended!\n");
			capturedVideo.release();
			if (bodyEngineVideoOutputFile.is_open()) bodyEngineVideoOutputFile.close();
		}
	}
}

void BodyTrack::writeFrameAndEstResults(const cv::Mat& frm, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
	if (captureFrame) {
		const std::string currentCalendarTime = getCalendarTime();
		const std::string capturedFrame = currentCalendarTime + ".png";
		cv::imwrite(capturedFrame, frm);
		if (FLAG_verbose) {
			std::cout << "Captured the frame" << std::endl;
		}
		// Write Body Engine Outputs
		const std::string outputFilename = currentCalendarTime + ".txt";
		std::ofstream outputFile;
		outputFile.open(outputFilename, std::ios_base::out);
		if (!outputFile.is_open()) {
			std::cout << "Error: Could not open file: \"" << outputFilename << "\"\n";
			return;
		}
		std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
		outputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n";
		outputFile << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
		writeEstResults(outputFile, output_bboxes, keypoints);
		if (outputFile.is_open()) outputFile.close();
		captureFrame = false;
	}
}

/********************************************************************************
 * draw stats to window
 ********************************************************************************/

void BodyTrack::getFPS() {
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

void BodyTrack::drawFPS(cv::Mat& img) {
	getFPS();
	if (frameTime && FLAG_drawFPS) {
		char buf[32];
		snprintf(buf, sizeof(buf), "%.1f", 1. / frameTime);
		cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), FONT_FACE, DEBUG_TEXT_SCALE, DEBUG_TEXT_COLOR, DEBUG_FONT_THICKNESS);
	}
}

/********************************************************************************
 * acquire joints keypoints and bounding boxes
 ********************************************************************************/

BodyTrack::Err BodyTrack::acquireWebcamOrVideoFrame() {
	Err err = errNone;

	// If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
	// frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
	// resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
	// done here) as well as reallocate memory accordingly with BodyEngine::initFeatureIOParams()
	cap >> frame;  // get a new frame from camera into the class variable frame.
	if (frame.empty()) {
		// if in Offline mode, this means end of video,so we return
		if (offlineMode) return errVideo;
		// try Init one more time if reading frames from camera
		err = initCamera(FLAG_camRes.c_str());
		if (err != errNone)
			return err;
		cap >> frame;
		if (frame.empty()) return errVideo;
	}
	return err;
}

BodyTrack::Err BodyTrack::acquireSharedMemFrame()
{
	/*
	*  !! IMPORTANT !!
	* Your video input must be flipped along the X before getting to the Shared Mem Out TOP
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

BodyTrack::Err BodyTrack::acquireBodyBoxAndKeyPoints() {
	Err err = errNone;
	NvAR_BBoxes output_bbox;
	NvAR_TrackingBBoxes output_tracking_bbox;
	std::chrono::steady_clock::time_point start, end;

	int numKeyPoints = body_ar_engine.getNumKeyPoints();
	std::vector<NvAR_Point2f> keypoints2D(numKeyPoints * PEOPLE_TRACKING_BATCH_SIZE);
	std::vector<NvAR_Point3f> keypoints3D(numKeyPoints * PEOPLE_TRACKING_BATCH_SIZE);
	std::vector<NvAR_Quaternion> jointAngles(numKeyPoints * PEOPLE_TRACKING_BATCH_SIZE);

	try {
		if (FLAG_debug) {
			start = std::chrono::high_resolution_clock::now();
		}

		unsigned n;
		// get keypoints in original image resolution coordinate space
		n = body_ar_engine.acquireBodyBoxAndKeyPoints(frame, keypoints2D.data(), keypoints3D.data(), jointAngles.data(), &output_tracking_bbox, 0);

		if (FLAG_debug) {
			end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "box+keypoints time: " << duration.count() << " microseconds" << std::endl;
		}

		if (n && FLAG_verbose) {
			printf("2D KeyPoints: [\n");
			for (const auto& pt : keypoints2D) {
				printf("%7.1f%7.1f\n", pt.x, pt.y);
			}
			printf("]\n");

			printf("3d KeyPoints: [\n");
			for (const auto& pt : keypoints3D) {
				printf("%7.1f%7.1f%7.1f\n", pt.x, pt.y, pt.z);
			}
			printf("]\n");
		}
		if (n == 0) {
			return errNoBody;
		}

		if (FLAG_captureOutputs) {
			writeFrameAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
			writeVideoAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
		}

		if (FLAG_drawTracking) {
			DrawKeyPointsAndEdges(frame, keypoints2D.data(), numKeyPoints, &output_tracking_bbox);
		}

		frameIndex++;

		return err;
	}
	catch (...) {
		printf("(BodyTrack) Could not complete body_ar_engine.acquireBodyBoxAndKeyPoints()\n");
		return errNone;
		//return errSharedMemSeg;
	}
}

/********************************************************************************
 * init processes
 ********************************************************************************/

BodyTrack::Err BodyTrack::initBodyEngine(const char* modelPath) {
	// make sure video capture has been started for webcam and video file
	if (!(FLAG_videoSource == sharedMemory) && !cap.isOpened())
		return errVideo;

	// start creating tracking features based on model and multi-track
	nvErr = body_ar_engine.createFeatures(modelPath, PEOPLE_TRACKING_BATCH_SIZE);

	// create a window on the desktop for displaying video feed and tracking
	if (!offlineMode && FLAG_drawWindow)
		cv::namedWindow(windowTitle, 1);

	return doAppErr(nvErr);
}

BodyTrack::Err BodyTrack::initSharedMemory() {
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
		body_ar_engine.setInputImageWidth(shmWidth);
		body_ar_engine.setInputImageHeight(shmHeight);

		return errNone;
	}
}

BodyTrack::Err BodyTrack::initCamera(const char* camRes) {
	if (cap.open(FLAG_camIndex)) {
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
			body_ar_engine.setInputImageWidth(inputWidth);
			body_ar_engine.setInputImageHeight(inputHeight);
		}
	}
	else
		return errCamera;
	return errNone;
}

BodyTrack::Err BodyTrack::initOfflineMode(const char* inputFilename, const char* outputFilename) {
	if (cap.open(inputFilename)) {
		printf("Input Video File: %s\n", inputFilename);
		inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
		inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		body_ar_engine.setInputImageWidth(inputWidth);
		body_ar_engine.setInputImageHeight(inputHeight);
	}
	else {
		printf("ERROR: Unable to open the input video file \"%s\" \n", inputFilename);
		return Err::errVideo;
	}

	std::string jdOutputVideoName;
	std::string outputFilePrefix;
	if (outputFilename && strlen(outputFilename) != 0) {
		outputFilePrefix = outputFilename;
	}
	else {
		size_t lastindex = std::string(inputFilename).find_last_of(".");
		outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
	}

	jdOutputVideoName = outputFilePrefix + "_BodyTrack.mp4";
	printf("Output Video: %s\n", jdOutputVideoName.c_str());

	bool canOpenOutputVideo = keyPointsOutputVideo.open(jdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS), cv::Size(inputWidth, inputHeight));
	if (!canOpenOutputVideo) {
		printf("ERROR: Unable to open the output video file \"%s\" \n", jdOutputVideoName.c_str());
		return Err::errGeneral;
	}

	return Err::errNone;
}

/********************************************************************************
 * main, run, stop
 ********************************************************************************/

void BodyTrack::printArgsToConsole() {
	// print args to console for debug
	printf("Mode: %s\n", FLAG_mode ? "High Performance" : "High Quality");
	printf("Use CUDA Graph: %s\n", FLAG_useCudaGraph ? "true" : "false");
	printf("Chosen GPU: %d\n", FLAG_chosenGPU);
	printf("Shadow Tracking Age: %d\n", FLAG_shadowTrackingAge);
	printf("Probation Age: %d\n", FLAG_probationAge);
	printf("Max Targets Tracked: %d\n", FLAG_maxTargetsTracked);

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
}

int main(int argc, char** argv) {
	// Parse the arguments
	if (0 != ParseMyArgs(argc, argv)) return -100;

	BodyTrack app;
	BodyTrack::Err doErr = BodyTrack::Err::errNone;

	// start initializing the tracking
	app.body_ar_engine.setAppMode(BodyEngine::mode(APP_MODE));
	app.body_ar_engine.setMode(FLAG_mode);
	app.body_ar_engine.setBodyStabilization(TEMPORAL_SMOOTHING);
	app.body_ar_engine.useCudaGraph(FLAG_useCudaGraph);
	app.body_ar_engine.enablePeopleTracking(ENABLE_MULTI_PEOPLE_TRACKING, FLAG_shadowTrackingAge, FLAG_probationAge, FLAG_maxTargetsTracked);

	// based on appMode and mode, set NVIDIA model used for tracking
	doErr = BodyTrack::errBodyModelInit;
	if (FLAG_modelPath.empty()) {
		printf("WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/body/models, "
			"SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
			"please restart your application after the SDK Installation. \n");
	}
	else {
		printf("Model Path: %s\n", FLAG_modelPath.c_str());
	}
	if (!bodyModel.empty()) {
		app.body_ar_engine.setBodyModel(bodyModel.c_str());
	}

	// debug args read from batch file
	app.printArgsToConsole();

	// initialize video source 
	offlineMode = FLAG_videoSource == videoFile;
	if (offlineMode) {
		// offline mode is only when video source is a file on disk
		if (FLAG_inFilePath.empty()) {
			doErr = BodyTrack::errMissing;
			printf("ERROR: %s, please specify input file using --in_file_path\n", app.errorStringFromCode(doErr));
			goto bail;
		}
		doErr = app.initOfflineMode(FLAG_inFilePath.c_str(), FLAG_outFilePrefix.c_str());
	}
	else {
		if (FLAG_videoSource == webcam)
			doErr = app.initCamera(FLAG_camRes.c_str());
		else if (FLAG_videoSource == sharedMemory)
			doErr = app.initSharedMemory();
	}
	BAIL_IF_ERR(doErr);
	printf(".\nInitialized video stream...\n");

	// initialize BodyEngine class based on model
	doErr = app.initBodyEngine(FLAG_modelPath.c_str());
	BAIL_IF_ERR(doErr);
	printf("Initialized BodyEngine...\n");

	// start analyzing the video each frame
	printf("Starting to run...\n");
	doErr = app.run();
	BAIL_IF_ERR(doErr);

bail:
	if (doErr)
		printf("ERROR: %s\n", app.errorStringFromCode(doErr));
	app.stop();
	return (int)doErr;
}

BodyTrack::Err BodyTrack::run() {

	BodyTrack::Err doErr = errNone;
	BodyEngine::Err err = body_ar_engine.initFeatureIOParams();
	if (err != BodyEngine::Err::errNone)
		return doAppErr(err);

	while (1) {
		// get frame based on desired video source
		// "frame" is a global variable shared among functions
		if (FLAG_videoSource == webcam || FLAG_videoSource == videoFile) {
			doErr = acquireWebcamOrVideoFrame();
		}
		else if (FLAG_videoSource == sharedMemory) {
			doErr = acquireSharedMemFrame();
		}

		if (offlineMode && frame.empty()) {
			// We have reached the end of the video so return without any error
			return BodyTrack::errNone;
		}
		else if (doErr != BodyTrack::errNone) {
			return doErr;
		}

		// get joint keypoint data and bounding box data
		doErr = acquireBodyBoxAndKeyPoints();
		if (doErr == BodyTrack::errCancel || doErr == BodyTrack::errVideo)
			return doErr;

		// write frame to output file or to window
		if (offlineMode) {
			if (doErr == BodyTrack::errNoBody || doErr == BodyTrack::errBodyFit)
				keyPointsOutputVideo.write(frame);
		}
		else { // if (!offlineMode) {
			if (!frame.empty() && FLAG_drawWindow) {
				if (FLAG_drawFPS)
					drawFPS(frame);
				cv::imshow(windowTitle, frame);
			}

			int n = cv::waitKey(1);
			if (n >= 0) {
				static const int ESC_KEY = 27;
				if (n == ESC_KEY) break;
				/*
				// This is leftover from parsing keys on the keyboard but might be useful
				 switch (n) {
				 case '1':
						body_ar_engine.destroyFeatures();
						body_ar_engine.setAppMode(BodyEngine::mode::bodyDetection);
						body_ar_engine.createFeatures(FLAG_modelPath.c_str(), 1);
						body_ar_engine.initFeatureIOParams();
						break;
				 case '2':
						body_ar_engine.destroyFeatures();
						body_ar_engine.setAppMode(BodyEngine::mode::keyPointDetection);
						if (FLAG_enablePeopleTracking)
							body_ar_engine.createFeatures(FLAG_modelPath.c_str());
						else
							body_ar_engine.createFeatures(FLAG_modelPath.c_str(), 1);
						body_ar_engine.initFeatureIOParams();
						break;
				}
				*/
			}
		}
	}
	return doErr;
}

void BodyTrack::stop() {
	body_ar_engine.destroyFeatures();

	if (offlineMode)
		keyPointsOutputVideo.release();
	else
		cv::destroyAllWindows();

	if (!FLAG_videoSource == sharedMemory)
		cap.release();
}