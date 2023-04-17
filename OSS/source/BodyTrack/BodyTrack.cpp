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

#define DEBUG_RUNTIME
#define PEOPLE_TRACKING_BATCH_SIZE 8

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
const unsigned int APP_MODE = 1;			
// Optimizes the results for temporal input frames. Not possible for multi-person tracking, so must be false.
const bool TEMPORAL_SMOOTHING = false;		

// shared memory
UT_SharedMem* shm;							// shared memory
TOP_SharedMemHeader* topHeader;				// TOP_SharedMemHeader.h
cv::Mat tempFrame;							// for converting touch shared mem to opencv mat
int width, height;							// size of image from shared memory

/********************************************************************************
 * command-line arguments
 ********************************************************************************/

bool
	FLAG_showFPS = true,					// Write FPS debug information to window
	FLAG_drawVisualization = true,			// Draw keypoint and bounding box data to window
	FLAG_debug = false,						// Print debugging information to the console
	FLAG_verbose = false,					// Print extra information to the console
	FLAG_captureOutputs = false,			// Enables video/image capture and writing body detection/keypoints outputs to file
	FLAG_offlineMode = false,				// False: Webcam, True: Video file --> specifies whether to use offline video or an online camera video as the input
	FLAG_useCudaGraph = true;				// Uses CUDA Graphs to improve performance. CUDA graph reduces the overhead of the GPU operation submission of 3D body tracking
bool captureFrame = false, captureVideo = false;

std::string 
	FLAG_outDir, 
	FLAG_inFile, 
	FLAG_outFile, 
	FLAG_modelPath = "C:/Program Files/NVIDIA Corporation/NVIDIA AR SDK/models",
	FLAG_captureCodec = "avc1", 
	FLAG_camRes, 
	FLAG_bodyModel,
	FLAG_sharedMemName = "TOPshm";

unsigned int
	FLAG_chosenGPU = 0,						// Index of GPU to run the Maxine executable
	FLAG_mode = 1,							// 0: High Quality, 1: High Performance -> default to high performance
	FLAG_camIndex = 0,						// Index of webcam connected to the PC
	FLAG_shadowTrackingAge = 90,			// Sets the Shadow Tracking Age for Multi-Person Tracking, the default value is 90 (frames).
	FLAG_probationAge = 10,					// Sets the Probation Age for Multi-Person tracking, the default value is 10 (frames).
	FLAG_maxTargetsTracked = 30;			// Sets the Maxinum Targets Tracked. The default value is 30, and the minimum is value is 1.

/********************************************************************************
 * parsing command line args
 ********************************************************************************/

static void Usage() {
	printf(
		"BodyTrack [<args> ...]\n"
		"where <args> is\n"
		" --debug[=(true|false)]				Report debugging info\n"
		" --verbose[=(true|false)]				Report interesting info\n"
		" --use_cuda_graph[=(true|false)]		Enable faster execution by using cuda graph to capture engine execution\n"
		" --capture_outputs[=(true|false)]		Enables video/image capture and writing body detection/keypoints outputs to file\n"
		" --offline_mode[=(true|false)]			Disables webcam, reads video from file, and writes output video results to file\n"
		" --cam_res=[WWWx]HHH					Specify resolution as height or width x height\n"
		" --cam_index[=0,1,2,3,...]				Specify the webcam index we want to use for the video feed\n"
		" --in_file=<file>						Specify the input file\n"
		" --codec=<fourcc>						FOURCC code for the desired codec (default H264)\n"
		" --in=<file>							Specify the input file\n"
		" --out_file=<file>						Specify the output file\n"
		" --out=<file>							Specify the output file\n"
		" --model_path=<path>					Specify the directory containing the TRT models\n"
		" --mode[=0|1]							Model Mode. 0: High Quality, 1: High Performance\n"
		" --enable_people_tracking[=(0|1)]		Enables people tracking\n "
		" --shadow_tracking_age					Shadow Tracking Age after which tracking information of a person is removed. Measured in frames\n"
		" --probation_age						Length of probationary period. Measured in frames\n"
		" --max_targets_tracked					Maximum number of targets to be tracked\n"
		" --benchmarks[=<pattern>]				Run benchmarks\n");
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
			GetFlagArgVal("verbose", arg, &FLAG_verbose) || 
			GetFlagArgVal("debug", arg, &FLAG_debug) ||
			GetFlagArgVal("in", arg, &FLAG_inFile) || 
			GetFlagArgVal("in_file", arg, &FLAG_inFile) ||
			GetFlagArgVal("out", arg, &FLAG_outFile) || 
			GetFlagArgVal("out_file", arg, &FLAG_outFile) ||
			GetFlagArgVal("offline_mode", arg, &FLAG_offlineMode) ||
			GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||
			GetFlagArgVal("cam_res", arg, &FLAG_camRes) || 
			GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||
			GetFlagArgVal("cam_index", arg, &FLAG_camIndex) ||
			GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
			GetFlagArgVal("mode", arg, &FLAG_mode) ||
			GetFlagArgVal("use_cuda_graph", arg, &FLAG_useCudaGraph) ||
			GetFlagArgVal("shadow_tracking_age", arg, &FLAG_shadowTrackingAge) ||
			GetFlagArgVal("probation_age", arg, &FLAG_probationAge) ||
			GetFlagArgVal("max_targets_tracked", arg, &FLAG_maxTargetsTracked) 
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

/********************************************************************************
 * error codes
 ********************************************************************************/

const char* BodyTrack::errorStringFromCode(BodyTrack::Err code) {
	struct LUTEntry {
		Err code;
		const char* str;
	};
	static const LUTEntry lut[] = {
		{errNone, "no error"},
		{errGeneral, "an error has occured"},
		{errRun, "an error has occured while the feature is running"},
		{errInitialization, "Initializing Body Engine failed"},
		{errRead, "an error has occured while reading a file"},
		{errEffect, "an error has occured while creating a feature"},
		{errParameter, "an error has occured while setting a parameter for a feature"},
		{errUnimplemented, "the feature is unimplemented"},
		{errMissing, "missing input parameter"},
		{errVideo, "no video source has been found"},
		{errImageSize, "the image size cannot be accommodated"},
		{errNotFound, "the item cannot be found"},
		{errBodyModelInit, "body model initialization failed"},
		{errGLFWInit, "GLFW initialization failed"},
		{errGLInit, "OpenGL initialization failed"},
		{errRendererInit, "renderer initialization failed"},
		{errGLResource, "an OpenGL resource could not be found"},
		{errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
		{errBodyFit, "an error has occurred while body fitting"},
		{errNoBody, "no body has been found"},
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
	cv::Mat frm = (FLAG_offlineMode) ? src.clone() : src;
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
	if (FLAG_offlineMode) 
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
				capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps,
					cv::Size(frm.cols, frm.rows));
				if (!capturedVideo.isOpened()) {
					std::cout << "Error: Could not open video: \"" << capturedOutputFileName << "\"\n";
					return;
				}
				if (FLAG_verbose) {
					std::cout << "Capturing video started" << std::endl;
				}
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
			bodyEngineVideoOutputFile
				<< "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
		}
		// Write each frame to the Video
		capturedVideo << frm;
		writeEstResults(bodyEngineVideoOutputFile, output_bboxes, keypoints);
	}
	else {
		if (capturedVideo.isOpened()) {
			if (FLAG_verbose) {
				std::cout << "Capturing video ended" << std::endl;
			}
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
	if (frameTime && FLAG_showFPS) {
		char buf[32];
		snprintf(buf, sizeof(buf), "%.1f", 1. / frameTime);
		cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), FONT_FACE, DEBUG_TEXT_SCALE, DEBUG_TEXT_COLOR, DEBUG_FONT_THICKNESS);
	}
}

/********************************************************************************
 * acquire joints keypoints and bounding boxes
 ********************************************************************************/

BodyTrack::Err BodyTrack::acquireFrame() {
	Err err = errNone;

	// If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
	// frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
	// resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
	// done here) as well as reallocate memory accordingly with BodyEngine::initFeatureIOParams()
	cap >> frame;  // get a new frame from camera into the class variable frame.
	if (frame.empty()) {
		// if in Offline mode, this means end of video,so we return
		if (FLAG_offlineMode) return errVideo;
		// try Init one more time if reading frames from camera
		err = initCamera(FLAG_camRes.c_str());
		if (err != errNone)
			return err;
		cap >> frame;
		if (frame.empty()) return errVideo;
	}

	return err;
}

BodyTrack::Err BodyTrack::acquireBodyBoxAndKeyPoints() {
	Err err = errNone;
	int numKeyPoints = body_ar_engine.getNumKeyPoints();
	NvAR_BBoxes output_bbox;
	NvAR_TrackingBBoxes output_tracking_bbox;
	std::vector<NvAR_Point2f> keypoints2D(numKeyPoints * 8);
	std::vector<NvAR_Point3f> keypoints3D(numKeyPoints * 8);
	std::vector<NvAR_Quaternion> jointAngles(numKeyPoints * 8);

#ifdef DEBUG_PERF_RUNTIME
	auto start = std::chrono::high_resolution_clock::now();
#endif

	unsigned n;
	// get keypoints in original image resolution coordinate space
	n = body_ar_engine.acquireBodyBoxAndKeyPoints(frame, keypoints2D.data(), keypoints3D.data(), jointAngles.data(), &output_tracking_bbox, 0);

#ifdef DEBUG_PERF_RUNTIME
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "box+keypoints time: " << duration.count() << " microseconds" << std::endl;
#endif

	if (n && FLAG_verbose && body_ar_engine.appMode != BodyEngine::mode::bodyDetection) {
		printf("KeyPoints: [\n");
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
	if (FLAG_captureOutputs) {
		writeFrameAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
		writeVideoAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
	}
	if (0 == n) return errNoBody;

	if (FLAG_drawVisualization)
		DrawKeyPointsAndEdges(frame, keypoints2D.data(), numKeyPoints, &output_tracking_bbox);

	frameIndex++;

	return err;
}

BodyTrack::Err BodyTrack::acquireSharedMemFrame()
{
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
		topHeader = (TOP_SharedMemHeader*)shm->getMemory();
		if (topHeader == NULL || shm->getErrorState() != UT_SHM_ERR_NONE)
		{
			// idk if we still need to do this null check but doing it just in case for clarity
			printf("No shared memory when trying to acquire frame\n");
			shm->unlock();
			return errSharedMem;
		}
		else // if (topHeader && (topHeader != NULL)) 
		{
			tempFrame.data = (unsigned char*)topHeader->getImage();
			shm->unlock();

			if (tempFrame.empty()) {
				// could not read frame from shared memory
				printf("Frame is empty and does not contain SharedMem image data\n");
				return errSharedMemVideo;
			}

			frame = tempFrame;
			return errNone;
		}
	}
}

/********************************************************************************
 * init processes
 ********************************************************************************/

BodyTrack::Err BodyTrack::initBodyEngine(const char* modelPath) {
	if (!cap.isOpened()) 
		return errVideo;

	int numKeyPoints = body_ar_engine.getNumKeyPoints();

	nvErr = body_ar_engine.createFeatures(modelPath, PEOPLE_TRACKING_BATCH_SIZE);

#ifdef DEBUG
	detector->setOutputLocation(outputDir);
#endif  // DEBUG

	if (!FLAG_offlineMode) 
		cv::namedWindow(windowTitle, 1);

	frameIndex = 0;

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
		topHeader = (TOP_SharedMemHeader*)shm->getMemory();

		width = topHeader->width;
		height = topHeader->height;

		float quarterWidth = width / 4.0;
		limitUser1 = quarterWidth;
		limitUser2 = quarterWidth * 2;
		limitUser3 = quarterWidth * 3;
		limitUser4 = width;	// quarterWidth * 4

		printf("Shared memory (w,h) = (%d,%d)\n\n", width, height);
		shm->unlock();

		tempFrame = cv::Mat(height, width, CV_8UC4);			// for converting image data from shared mem to frame
		body_ar_engine.setInputImageWidth(width);
		body_ar_engine.setInputImageHeight(height);

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
		inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
		inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		body_ar_engine.setInputImageWidth(inputWidth);
		body_ar_engine.setInputImageHeight(inputHeight);
	}
	else {
		printf("ERROR: Unable to open the input video file \"%s\" \n", inputFilename);
		return Err::errVideo;
	}

	std::string bdOutputVideoName, jdOutputVideoName;
	std::string outputFilePrefix;
	if (outputFilename && strlen(outputFilename) != 0) {
		outputFilePrefix = outputFilename;
	}
	else {
		size_t lastindex = std::string(inputFilename).find_last_of(".");
		outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
	}
	bdOutputVideoName = outputFilePrefix + "_bbox.mp4";
	jdOutputVideoName = outputFilePrefix + "_pose.mp4";

	if (!bodyDetectOutputVideo.open(bdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
		cv::Size(inputWidth, inputHeight))) {
		printf("ERROR: Unable to open the output video file \"%s\" \n", bdOutputVideoName.c_str());
		return Err::errGeneral;
	}
	if (!keyPointsOutputVideo.open(jdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
		cv::Size(inputWidth, inputHeight))) {
		printf("ERROR: Unable to open the output video file \"%s\" \n", bdOutputVideoName.c_str());
		return Err::errGeneral;
	}

	return Err::errNone;
}

/********************************************************************************
 * main, run, stop
 ********************************************************************************/

int main(int argc, char** argv) {
	// Parse the arguments
	if (0 != ParseMyArgs(argc, argv)) return -100;

	BodyTrack app;
	BodyTrack::Err doErr = BodyTrack::Err::errNone;

	app.body_ar_engine.setAppMode(BodyEngine::mode(APP_MODE));
	app.body_ar_engine.setMode(FLAG_mode);
	app.body_ar_engine.setBodyStabilization(TEMPORAL_SMOOTHING);
	app.body_ar_engine.useCudaGraph(FLAG_useCudaGraph);
	app.body_ar_engine.enablePeopleTracking(ENABLE_MULTI_PEOPLE_TRACKING, FLAG_shadowTrackingAge, FLAG_probationAge, FLAG_maxTargetsTracked);

	doErr = BodyTrack::errBodyModelInit;
	if (FLAG_modelPath.empty()) {
		printf("WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/body/models, "
			"SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
			"please restart your application after the SDK Installation. \n");
	}
	if (!FLAG_bodyModel.empty())
		app.body_ar_engine.setBodyModel(FLAG_bodyModel.c_str());

	if (FLAG_offlineMode) {
		if (FLAG_inFile.empty()) {
			doErr = BodyTrack::errMissing;
			printf("ERROR: %s, please specify input file using --in_file or --in \n", app.errorStringFromCode(doErr));
			goto bail;
		}
		doErr = app.initOfflineMode(FLAG_inFile.c_str(), FLAG_outFile.c_str());
	}
	else {
		doErr = app.initCamera(FLAG_camRes.c_str());
	}
	BAIL_IF_ERR(doErr);

	doErr = app.initBodyEngine(FLAG_modelPath.c_str());
	BAIL_IF_ERR(doErr);

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
		//printf(">> frame %d \n", framenum++);

		doErr = acquireFrame();
		
		if (FLAG_offlineMode && frame.empty()) {
			// We have reached the end of the video so return without any error
			return BodyTrack::errNone;
		}
		else if (doErr != BodyTrack::errNone) {
			return doErr;
		}

		doErr = acquireBodyBoxAndKeyPoints();

		if (FLAG_offlineMode && (doErr == BodyTrack::errNoBody || doErr == BodyTrack::errBodyFit)) {
			bodyDetectOutputVideo.write(frame);
			keyPointsOutputVideo.write(frame);
		}
		
		if (doErr == BodyTrack::errCancel || doErr == BodyTrack::errVideo) 
			return doErr;
				
		if (!FLAG_offlineMode) {
			if (!frame.empty()) {
				if (FLAG_drawVisualization)
					drawFPS(frame);
				cv::imshow(windowTitle, frame);
			}
			int n = cv::waitKey(1);
			if (n >= 0) {
				static const int ESC_KEY = 27;
				if (n == ESC_KEY) break;
				/*
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
	if (FLAG_offlineMode) {
		bodyDetectOutputVideo.release();
		keyPointsOutputVideo.release();
	}
	cap.release();
	cv::destroyAllWindows();
}