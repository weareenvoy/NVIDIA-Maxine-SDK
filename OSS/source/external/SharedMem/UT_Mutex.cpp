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
 * Filename: UT_Mutex.C
 */

#include "UT_Mutex.h"
#include <cassert>

#ifdef __APPLE__
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/fcntl.h>
#include <sys/errno.h>
#include <unistd.h>
#endif

// We don't include the leakwatch here
// since this file is used by users

// macOS has no sem_timedwait(), which would simplify this greatly
//	- instead we use a pthread_mutex and pthread_cond in shared memory

UT_Mutex::UT_Mutex(const MtxString &name)
{
#ifdef _WIN32
	myMutex = CreateMutexW(NULL, FALSE, name.data());
#else
	int result = 0;

	// First try creating the shared mutex from scratch
	int fd = shm_open(name.data(), O_RDWR | O_CREAT | O_EXCL, 0666);
	if (fd < 0)
	{
		result = errno;
	}
	if (result == 0)
	{
		// We created the file and have exclusive access: set up the mutex and condition
		result = ftruncate(fd, sizeof(SharedMutex));
		if (result != 0)
		{
			result = errno;
		}
		myMutex = reinterpret_cast<SharedMutex *>(mmap(nullptr, sizeof(SharedMutex), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HASSEMAPHORE, fd, 0));
		if (myMutex == MAP_FAILED)
		{
			myMutex = nullptr;
			result = errno;
		}

		if (result == 0)
		{
			// set up the mutex
			pthread_mutexattr_t mattribs;
			result = pthread_mutexattr_init(&mattribs);
			if (result == 0)
			{
				result = pthread_mutexattr_setpshared(&mattribs, PTHREAD_PROCESS_SHARED);

				if (result == 0)
				{
					result = pthread_mutex_init(&myMutex->mutex, &mattribs);
				}
				pthread_mutexattr_destroy(&mattribs);
			}
		}

		if (result == 0)
		{
			// set up the condition
			pthread_condattr_t cattribs;
			result = pthread_condattr_init(&cattribs);
			if (result == 0)
			{
				result = pthread_condattr_setpshared(&cattribs, PTHREAD_PROCESS_SHARED);
				if (result == 0)
				{
					result = pthread_cond_init(&myMutex->cond, &cattribs);
				}
				pthread_condattr_destroy(&cattribs);
			}

			myMutex->locked = false;
		}

		// unmap the memory and close the file descriptor - we will open it again without exclusive access
		munmap(myMutex, sizeof(SharedMutex));
		close(fd);
		fd = -1;
	}
	else if (result == EEXIST)
	{
		// shm_open() with O_CREAT | O_EXCL failed, so it already exists, in which case we can use it
		result = 0;
	}
	if (result == 0)
	{
		int tries = 0;
		do
		{
			fd = shm_open(name.data(), O_RDWR);
			if (fd < 0)
			{
				result = errno;
			}
			if (result == EACCES)
			{
				// give the other instance a moment to finish creating it
				usleep(1000);
			}
			tries++;
		} while (result == EACCES && tries < 10);
	}
	if (result == 0)
	{
		myMutex = reinterpret_cast<SharedMutex *>(mmap(nullptr, sizeof(SharedMutex), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HASSEMAPHORE, fd, 0));
		if (myMutex == MAP_FAILED)
		{
			myMutex = nullptr;
			result = errno;
		}
	}
	if (result != 0)
	{
		cleanup();
	}
	// memory remains mapped until munmap, we can close the file descriptor
	if (fd != -1)
	{
		close(fd);
	}
#endif
}


UT_Mutex::~UT_Mutex()
{
	// You shouldn't be destroying this if you are currently the owner
	assert(!myLockOwner);
	// But for safely, we'll try unlocking.
	unlock();
#ifdef _WIN32
	CloseHandle(myMutex);
#else
	cleanup();
#endif 
}

#ifdef __APPLE__
void
UT_Mutex::cleanup()
{
	if (myMutex)
	{
		munmap(myMutex, sizeof(SharedMutex));
	}

	// Once created, the mutex lives forever - otherwise if it remains mapped in a process after the
	// creator unlinks the name, then another instance is created, two instances will effectively coexist,
	// so we never call any of pthread_mutex_destroy, pthread_cond_destroy or shm_unlink
}
#endif // __APPLE__

bool
UT_Mutex::lock()
{
#ifdef _WIN32
	if (myLockOwner)
		return true;

	DWORD result = WaitForSingleObject(myMutex, INFINITE);
	if (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED)
		myLockOwner = true;
#else
	bool didLock = false;
	if (myMutex)
	{
		int result = pthread_mutex_lock(&myMutex->mutex);
		if (result == 0)
		{
			while (myMutex->locked && result == 0) {
				result = pthread_cond_wait(&myMutex->cond, &myMutex->mutex);
			}
			if (result == 0)
			{
				myLockOwner = didLock = myMutex->locked = true;
			}
			pthread_mutex_unlock(&myMutex->mutex);
			if (result != 0)
			{
				// some error has happened in pthread_cond_wait
				assert(result == 0);
			}
		}
	}
#endif
	return myLockOwner;
}

bool
UT_Mutex::tryLock(int timeout)
{
#ifdef _WIN32
	if (myLockOwner)
		return true;
	DWORD result = WaitForSingleObject(myMutex, timeout);
	if (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED)
		myLockOwner = true;
#else
	bool didLock = false;
	if (myMutex)
	{
		int result = pthread_mutex_lock(&myMutex->mutex);
		if (result == 0)
		{
			struct timeval tv; // seconds, microseconds
			gettimeofday(&tv, nullptr);
			tv.tv_sec += timeout / 1000;
			tv.tv_usec += (timeout % 1000) * 1000;

			struct timespec ts; // seconds, nanoseconds
			ts.tv_sec = tv.tv_sec;
			ts.tv_nsec = tv.tv_usec * 1000;

			while (myMutex->locked && result == 0) {
				result = pthread_cond_timedwait(&myMutex->cond, &myMutex->mutex, &ts);
			}
			if (result == 0)
			{
				myLockOwner = didLock = myMutex->locked = true;
			}
			pthread_mutex_unlock(&myMutex->mutex);
		}
	}
#endif 
	return myLockOwner;
}

bool
UT_Mutex::unlock()
{
	if (!myLockOwner)
		return false;

#ifdef _WIN32
	myLockOwner = false;
	return ReleaseMutex(myMutex);
#else
	if (myLockOwner && myMutex && pthread_mutex_lock(&myMutex->mutex) == 0)
	{
		myMutex->locked = false;
		myLockOwner = false;
		pthread_cond_signal(&myMutex->cond);
		pthread_mutex_unlock(&myMutex->mutex);
		return true;
	}
	return false;
#endif 
}
