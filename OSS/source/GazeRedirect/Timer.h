#pragma once
#include <chrono>
using namespace std;

class Timer {
public:
	Timer();  /**< Clear the duration to 0. */
	void start();  /**< Start  the timer. */
	void pause();  /**< Pause  the timer. */
	void resume();  /**< Resume the timer. */
	void stop();  /**< Stop   the timer. */
	double elapsedTimeFloat() const;	 /**< Report the elapsed time as a float. */
private:
	std::chrono::high_resolution_clock::time_point t0;
	std::chrono::high_resolution_clock::duration dt;
};