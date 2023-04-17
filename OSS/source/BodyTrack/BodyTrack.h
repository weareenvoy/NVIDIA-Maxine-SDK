#pragma once

#include "BodyEngine.h"
#include "Timer.h"

class BodyTrack {
public:
	BodyEngine body_ar_engine;
	BodyTrack();
	~BodyTrack();
	static const char windowTitle[];

	enum Err {
		errNone = BodyEngine::Err::errNone,
		errGeneral = BodyEngine::Err::errGeneral,
		errRun = BodyEngine::Err::errRun,
		errInitialization = BodyEngine::Err::errInitialization,
		errRead = BodyEngine::Err::errRead,
		errEffect = BodyEngine::Err::errEffect,
		errParameter = BodyEngine::Err::errParameter,
		errUnimplemented,
		errMissing,
		errVideo,
		errImageSize,
		errNotFound,
		errBodyModelInit,
		errGLFWInit,
		errGLInit,
		errRendererInit,
		errGLResource,
		errGLGeneric,
		errBodyFit,
		errNoBody,
		errSDK,
		errCuda,
		errCancel,
		errCamera,
		errSharedMem, 
		errSharedMemLock, 
		errSharedMemSeg, 
		errSharedMemVideo
	};
	BodyEngine::Err nvErr;
	Err doAppErr(BodyEngine::Err status) { return (Err)status; }
	static const char* errorStringFromCode(Err code);

	Err run();
	void stop();

	Err initBodyEngine(const char* modelPath = nullptr);
	Err initCamera(const char* camRes = nullptr);
	Err initOfflineMode(const char* inputFilename = nullptr, const char* outputFilename = nullptr);
	Err initSharedMemory();

	Err acquireFrame();
	Err acquireSharedMemFrame();
	Err acquireBodyBoxAndKeyPoints();

	void DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_TrackingBBoxes* output_bbox);
	void DrawKeyPointLine(cv::Mat& frm, NvAR_Point2f* keypoints);
	void drawKeyPointLine(const cv::Mat& src, NvAR_Point2f* keypoints, int point1, int point2, int color);
	
	void getFPS();
	void drawFPS(cv::Mat& img);

	void writeVideoAndEstResults(const cv::Mat& frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeFrameAndEstResults(const cv::Mat& frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeEstResults(std::ofstream& outputFile, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);

	cv::Mat frame;
	cv::VideoCapture cap{};
	cv::VideoWriter capturedVideo;
	cv::VideoWriter bodyDetectOutputVideo{}, keyPointsOutputVideo{};
	int inputWidth, inputHeight;
	int frameIndex;
	double frameTime;
	Timer frameTimer;
	// std::chrono::high_resolution_clock::time_point frameTimer;
	std::ofstream bodyEngineVideoOutputFile;
	float expr[6];
	float scaleOffsetXY[4];
	std::vector<cv::Scalar> colorCodes = { cv::Scalar(255,255,255) };

	// Batch Size has to be 8 when people tracking is enabled
	const unsigned int PEOPLE_TRACKING_BATCH_SIZE = 8; 
};