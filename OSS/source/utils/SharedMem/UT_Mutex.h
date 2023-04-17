/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

/*
 * COMMENTS:
 *	This class is for inter-processes locking based on named mutex
 *
 */

#ifndef __UT_Mutex__
#define __UT_Mutex__


#ifdef _WIN32
	#include <ws2tcpip.h>
	#include <windows.h>
#else
	#include <pthread.h>
#endif

#include <string>

#ifdef _WIN32
	typedef HANDLE mutexId;
	typedef std::wstring MtxString;
#else
	typedef std::string MtxString;
#endif


// Needed so people can compile this outside the TouchDesigner build environment
#ifndef UT_DLLEXP
#define UT_DLLEXP
#endif 
	
class UT_DLLEXP UT_Mutex
{
public:
	// Note name length on macOS must be <= PSHMNAMLEN (31)
	UT_Mutex(const MtxString &name);
	~UT_Mutex();

	// These return true if the mutex lock was successfully obtained, false otherwise
	bool	lock();
	// Windows accepts INFINITE here, macOS implementation doesn't
	bool	tryLock(int timeout = 0);
	// Returns true if the mutex was previously locked, and no error occured trying to unlock it.
	bool	unlock();

	// This class is distributed to the users, so make sure it doesn't
	// rely on any internal TouchDesigner classes

private:
	bool myLockOwner{false};

#ifdef _WIN32
	mutexId	myMutex;
#endif
#ifdef __APPLE__
	struct SharedMutex
	{
		pthread_mutex_t mutex;
		pthread_cond_t  cond;
		bool            locked;
	};
	void cleanup();
	SharedMutex *myMutex{nullptr};
#endif // __APPLE__
};

#endif /* __UT_Mutex__ */
