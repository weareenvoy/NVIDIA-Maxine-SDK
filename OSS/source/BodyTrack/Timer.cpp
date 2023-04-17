#include <chrono>
#include <Timer.h>
using namespace std;

//private:
//	std::chrono::high_resolution_clock::time_point t0;
//	std::chrono::high_resolution_clock::duration dt;

Timer::Timer() {
	/**< Clear the duration to 0. */
	dt = dt.zero(); 
} 

void Timer::start() { 
	/**< Start  the timer. */
	t0 = std::chrono::high_resolution_clock::now(); 
}  

void Timer::pause() { 
	/**< Pause  the timer. */
	dt = std::chrono::high_resolution_clock::now() - t0; 
}  

void Timer::resume() { 
	/**< Resume the timer. */
	t0 = std::chrono::high_resolution_clock::now() - dt; 
}  

void Timer::stop() {
	/**< Stop the timer. */
	pause(); 
}  

double Timer::elapsedTimeFloat() const { 
	/**< Report the elapsed time as a float. */
	return std::chrono::duration<double>(dt).count(); 
}	 