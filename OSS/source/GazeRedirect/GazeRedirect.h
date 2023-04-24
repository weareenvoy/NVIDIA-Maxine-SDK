#pragma once

#include "GazeEngine.h"
#include "Timer.h"

class GazeTrack {
public:
	GazeEngine gaze_ar_engine;
	GazeTrack();
	~GazeTrack();
	static const char windowTitle[];

	enum Err {
		errNone = GazeEngine::Err::errNone,
		errGeneral = GazeEngine::Err::errGeneral,
		errRun = GazeEngine::Err::errRun,
		errInitialization = GazeEngine::Err::errInitialization,
		errRead = GazeEngine::Err::errRead,
		errEffect = GazeEngine::Err::errEffect,
		errParameter = GazeEngine::Err::errParameter,
		errUnimplemented,
		errMissing,
		errVideo,
		errImageSize,
		errNotFound,
		errGLFWInit,
		errGLInit,
		errRendererInit,
		errGLResource,
		errGLGeneric,
		errSDK,
		errCuda,
		errCancel,
		errCamera,
		errSharedMem,
		errSharedMemLock,
		errSharedMemSeg,
		errSharedMemVideo
	};
	GazeEngine::Err nvErr;
	Err doAppErr(GazeEngine::Err status) { return (Err)status; }
	static const char* errorStringFromCode(Err code);
	
	Err run();
	void stop();
	
	void getFPS();
	void drawFPS(cv::Mat& img);
	void sendFPS();
	void printArgsToConsole();

	void initCudaGPU();
	Err initGazeEngine(const char* modelPath = nullptr, bool isLandmarks126 = false, bool gazeRedirect = false, unsigned eyeSizeSensitivity = 3, bool useCudaGraph = false);
	Err initCamera(const char* camRes = nullptr, unsigned int camID = 0);
	Err initOfflineMode(const char* inputFilename = nullptr, const char* outputFilename = nullptr);
	Err initSharedMemory();
	
	Err acquireWebcamOrVideoFrame();
	Err acquireSharedMemFrame();
	Err acquireGazeRedirection();

	void DrawLandmarkPoints(const cv::Mat& src, NvAR_Point2f* facial_landmarks, int numLandmarks, cv::Scalar* color);
	void drawBBoxes(const cv::Mat& src, NvAR_Rect* output_bbox);
	
	Err writeVideo(const cv::Mat& frm);

	cv::VideoCapture cap{};
	cv::Mat frame, gazeRedirectFrame;
	cv::VideoWriter gazeRedirectOutputVideo{};
	std::ofstream gazeEngineVideoOutputFile;
	cv::VideoWriter capturedVideo;

	int inputWidth, inputHeight;
	int frameIndex;
	double frameTime;
	Timer frameTimer;
	Timer fpsSendDelayTimer;
};