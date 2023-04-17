#pragma once

#include "BodyEngine.h"
#include "Timer.h"

class DoApp {
public:
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
		errCamera
	};
	Err doAppErr(BodyEngine::Err status) { return (Err)status; }
	BodyEngine body_ar_engine;
	DoApp();
	~DoApp();

	void stop();
	Err initBodyEngine(const char* modelPath = nullptr);
	Err initCamera(const char* camRes = nullptr);
	Err initOfflineMode(const char* inputFilename = nullptr, const char* outputFilename = nullptr);
	Err acquireFrame();
	Err acquireBodyBox();
	Err acquireBodyBoxAndKeyPoints();
	Err run();
	void drawFPS(cv::Mat& img);
	void DrawBBoxes(const cv::Mat& src, NvAR_Rect* output_bbox);
	//TODO: Look into ways of simplifying the app for these functions.
	void DrawBBoxes(const cv::Mat& src, NvAR_BBoxes* output_bbox);
	void DrawBBoxes(const cv::Mat& src, NvAR_TrackingBBoxes* output_bbox);
	void DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_BBoxes* output_bbox);
	void DrawKeyPointLine(cv::Mat& frm, NvAR_Point2f* keypoints);
	void drawKeyPointLine(const cv::Mat& src, NvAR_Point2f* keypoints, int point1, int point2, int color);
	void drawKalmanStatus(cv::Mat& img);
	void drawVideoCaptureStatus(cv::Mat& img);
	void processKey(int key);
	void writeVideoAndEstResults(const cv::Mat& frame, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeFrameAndEstResults(const cv::Mat& frame, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeEstResults(std::ofstream& outputFile, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_TrackingBBoxes* output_bbox);
	void writeVideoAndEstResults(const cv::Mat& frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeFrameAndEstResults(const cv::Mat& frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void writeEstResults(std::ofstream& outputFile, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints = NULL);
	void getFPS();
	static const char* errorStringFromCode(Err code);

	cv::VideoCapture cap{};
	cv::Mat frame;
	int inputWidth, inputHeight;
	cv::VideoWriter bodyDetectOutputVideo{}, keyPointsOutputVideo{};
	int frameIndex;
	static const char windowTitle[];
	double frameTime;
	// std::chrono::high_resolution_clock::time_point frameTimer;
	Timer frameTimer;
	cv::VideoWriter capturedVideo;
	std::ofstream bodyEngineVideoOutputFile;

	BodyEngine::Err nvErr;
	float expr[6];
	bool drawVisualization, showFPS, captureVideo, captureFrame;
	float scaleOffsetXY[4];
	std::vector<cv::Scalar> colorCodes = { cv::Scalar(255,255,255) };
	const unsigned int peopleTrackingBatchSize = 8; // Batch Size has to be 8 when people tracking is enabled
};