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
 * Filename: UT_SharedMem.C
 */

#ifdef __APPLE__
	#include <sys/mman.h>
	#include <sys/posix_shm.h>
	#include <sys/fcntl.h>
	#include <sys/stat.h>
	#include <algorithm>
	#include <unistd.h>
#endif 
#include <cassert>
#include <cstdlib>
#include <ctime>
#include "UT_SharedMem.h"

bool
UT_SharedMem::open(const ShmString &name,  unsigned int size, bool supportInfo)
{
	mySize = size;
	memset(myNamePostFix, 0, sizeof(myNamePostFix));

	myShortName = name;

	mySupportInfo = supportInfo;

	if (size > 0)
		myAmOwner = true;
	else
		myAmOwner = false;

	createName();

	ShmString m;
	m = myName;
#ifdef _WIN32
	m += L"Mutex";

	myMutex = new UT_Mutex(m);
#else
	m += "Mutex";

	myMutex = new UT_Mutex(ShmNameForName(m, myAmOwner));
#endif

	if (supportInfo)
	{
		if (!createInfo())
		{
			return false;
		}
	}
	else
	{
		mySharedMemInfo = NULL;
	}

	if (size > 0)
	{
		if (!createSharedMem())
		{
			myErrorState = UT_SHM_ERR_ALREADY_EXIST;
			return false;
		}
	}
	else
	{
		if (!openSharedMem())
		{
			myErrorState = UT_SHM_ERR_DOESNT_EXIST;
			return false;
		}
	}
	myErrorState = UT_SHM_ERR_NONE;
	return true;
}

UT_SharedMem::UT_SharedMem(const ShmString &name)
{
	open(name);
}

UT_SharedMem::UT_SharedMem(const ShmString &name, unsigned int size)
{
	open(name, size);
}

UT_SharedMem::UT_SharedMem(const ShmString &name, unsigned int size, bool supportInfo)
{
	open(name, size, supportInfo);
}

UT_SharedMem::~UT_SharedMem()
{
	detach();
	delete mySharedMemInfo;
	delete myMutex;
}

bool
UT_SharedMem::checkInfo()
{
	if (mySupportInfo)
	{
		// If we are looking for an info and can't find it,
		// then release the segment also
		if (!createInfo())
		{
			detach();
			myErrorState = UT_SHM_ERR_INFO_DOESNT_EXIST;
			return false;
		}
	}

	if (mySharedMemInfo && mySharedMemInfo->getErrorState() == UT_SHM_ERR_NONE && !myAmOwner)
	{
		if (mySharedMemInfo->tryLock(5000))
		{
			UT_SharedMemInfo *info = (UT_SharedMemInfo*)mySharedMemInfo->getMemory();

			if (info->version > 1)
			{
				if (info->detach)
				{
					mySharedMemInfo->unlock();
					detach();
					myErrorState = UT_SHM_ERR_INFO_DOESNT_EXIST;
					return false;
				}
			}

			ShmString pn;
			pn = info->namePostFix;

			if (pn != myNamePostFix)
			{
				memcpy(myNamePostFix, pn.data(), UT_SHM_MAX_POST_FIX_SIZE * sizeof(ShmChar));
				detachInternal();
			}
			mySharedMemInfo->unlock();
		}
		else
		{
			myErrorState = UT_SHM_ERR_UNABLE_TO_LOCK_INFO;
			return false;
		}

	}
	return true;
}

void
UT_SharedMem::resize(unsigned int s)
{

	// This can't be called by someone that didn't create it in the first place
	// Also you can't resize it if you arn't using the info feature
	// Finally, don't set the size to 0, just delete this object if you want to clean it
	if (mySize > 0 && mySharedMemInfo && myAmOwner)
	{
		if (mySharedMemInfo->tryLock(5000))
		{
			UT_SharedMemInfo *info = (UT_SharedMemInfo*)mySharedMemInfo->getMemory();
			if (info && info->supported)
			{
				detachInternal();
				mySize = s;
				// Keep trying until we find a name that works
				do 
				{
					randomizePostFix();
					createName();
				} while(!createSharedMem());
				memcpy(info->namePostFix, myNamePostFix, UT_SHM_MAX_POST_FIX_SIZE * sizeof(ShmChar));
			}
			else // Otherwise, just try and detach and resize, if it fails give up
			{
				detachInternal();
				mySize = s;
				if (!createSharedMem())
				{
					myErrorState = UT_SHM_ERR_ALREADY_EXIST;
				}

			}
			// May have been deleted
			if (mySharedMemInfo)
				mySharedMemInfo->unlock();
		}
		else
		{
			myErrorState = UT_SHM_ERR_UNABLE_TO_LOCK_INFO;
		}
	}
}

void
UT_SharedMem::randomizePostFix()
{
	for (int i = 0; i < UT_SHM_MAX_POST_FIX_SIZE - 1; i++)
	{
		int r = rand() % 26;
		char ch = 'a' + r;
		myNamePostFix[i] = ch;
	}
}

void
UT_SharedMem::createName()
{
#ifdef _WIN32
	myName = L"TouchSHM";
#else
	myName = "TouchSHM";
#endif

	myName += myShortName;
	myName += myNamePostFix;
#ifdef __APPLE__
	myShmName = ShmNameForName(myName, myAmOwner);
#endif
}

bool
UT_SharedMem::createSharedMem()
{
	if (myMapping)
		return true;

#ifdef _WIN32 
	myMapping = CreateFileMappingW(INVALID_HANDLE_VALUE, 
								  NULL,
								  PAGE_READWRITE,
								  0,
								  mySize,
								  myName.data());

	if (GetLastError() == ERROR_ALREADY_EXISTS)
	{
		detach();
		return false;
	}
#else
	myMapping = shm_open(myShmName.data(), O_RDWR | O_CREAT, 0666);
	if (myMapping == -1)
	{
		myMapping = 0;
	}
	if (myMapping != 0)
	{
		bool sized = false;
		struct stat stats;
		if (fstat(myMapping, &stats) == 0)
		{
			// If the memory was already created and is open elsewhere, then
			// we can't re-size it, so check for that, then size it if size
			// is currently zero
			if (stats.st_size >= mySize ||
				(stats.st_size == 0 && ftruncate(myMapping, mySize) == 0))
			{
				sized = true;
			}
		}
		if (!sized)
		{
			// couldn't size, so delete
			detach();
		}
	}
#endif 

	if (myMapping)
		return true;
	else
		return false;
}

bool
UT_SharedMem::openSharedMem()
{
	if (myMapping)
		return true;
	createName();
#ifdef _WIN32 
	myMapping = OpenFileMappingW( FILE_MAP_ALL_ACCESS, FALSE, myName.data());
#else
	myMapping = shm_open(myShmName.data(), O_RDWR);
	if (myMapping == -1)
	{
		myMapping = 0;
	}
#endif

	if (!myMapping)
		return false;


	return true;
}

bool
UT_SharedMem::detachInternal()
{
	if (myMemory)
	{
#ifdef _WIN32 
		UnmapViewOfFile(myMemory);
#else
		munmap(myMemory, mySize);
#endif 
		myMemory = 0;
	}
	if (myMapping)
	{
#ifdef _WIN32 
		CloseHandle(myMapping);
#else
		close(myMapping);
		if (myAmOwner)
		{
			shm_unlink(myShmName.data());
		}
#endif 
		myMapping = 0;
	}

#ifdef _WIN32
	// Try to open the file again, if it works then someone else is still holding onto the file
	// This behaviour is different on macOS - as soon as we shm_unlink it, it can't be opened again
	// (but already-mapped instances remain, so nor can we recreate it) - a client count
	// could be added to the info struct to give an indication for a return value
	if (openSharedMem())
	{
		CloseHandle(myMapping);
		myMapping = 0;
		return false;
	}
#endif
			
	return true;
}


bool
UT_SharedMem::detach()
{
	if (mySharedMemInfo)
	{
		if (mySharedMemInfo->getErrorState() == UT_SHM_ERR_NONE)
		{
			if (mySharedMemInfo->tryLock(5000))
			{
				UT_SharedMemInfo *info = (UT_SharedMemInfo*)mySharedMemInfo->getMemory();
				if (info && myAmOwner)
				{
					info->detach = true;
				}
				mySharedMemInfo->unlock();
			}
		}
		delete mySharedMemInfo;
		mySharedMemInfo = NULL;
	}
	memset(myNamePostFix, 0, sizeof(myNamePostFix));
	return detachInternal();
}

bool
UT_SharedMem::createInfo()
{
	if (!mySupportInfo)
		return true;
	if (mySharedMemInfo)
	{
		return mySharedMemInfo->getErrorState() == UT_SHM_ERR_NONE;
	}

	srand(time(NULL));
	ShmString infoName;
	infoName += myName;
	infoName += UT_SHM_INFO_DECORATION;

	mySharedMemInfo = new UT_SharedMem(infoName, 
									   myAmOwner ? sizeof(UT_SharedMemInfo) : 0, false);

	if (myAmOwner)
	{
		if (mySharedMemInfo->getErrorState() != UT_SHM_ERR_NONE)
		{
			myErrorState = UT_SHM_ERR_INFO_ALREADY_EXIST;
			return false;
		}
		if (mySharedMemInfo->tryLock(5000))
		{
			UT_SharedMemInfo *info = (UT_SharedMemInfo*)mySharedMemInfo->getMemory();
			if (!info)
			{
				myErrorState = UT_SHM_ERR_UNABLE_TO_MAP;
				mySharedMemInfo->unlock();
				return false;
			}
			info->magicNumber = UT_SHM_INFO_MAGIC_NUMBER;
			info->version = 2;
			info->supported = false;
			info->detach = false;
			memset(info->namePostFix, 0, UT_SHM_MAX_POST_FIX_SIZE);
			mySharedMemInfo->unlock();
		}
		else
		{
			myErrorState = UT_SHM_ERR_UNABLE_TO_MAP;
			return false;
		}
	}
	else
	{
		if (mySharedMemInfo->getErrorState() != UT_SHM_ERR_NONE)
		{
			myErrorState = UT_SHM_ERR_INFO_DOESNT_EXIST;
			return false;
		}
		if (mySharedMemInfo->tryLock(5000))
		{
			UT_SharedMemInfo *info = (UT_SharedMemInfo*)mySharedMemInfo->getMemory();
			if (!info)
			{
				myErrorState = UT_SHM_ERR_UNABLE_TO_MAP;
				mySharedMemInfo->unlock();
				return false;
			}
			if (info->magicNumber != UT_SHM_INFO_MAGIC_NUMBER)
			{
				myErrorState = UT_SHM_ERR_INFO_DOESNT_EXIST;
				mySharedMemInfo->unlock();
				return false;
			}
			// Let the other process know that we support the info
			info->supported = true;
			mySharedMemInfo->unlock();
		}
		else
		{
			myErrorState = UT_SHM_ERR_UNABLE_TO_MAP;
			return false;
		}
	}

	return true;
}

void *
UT_SharedMem::getMemory()
{
	if (!checkInfo())
	{
		return NULL;
	}

	if( myMemory == 0 )
	{
		if ((myAmOwner && createSharedMem()) || (!myAmOwner && openSharedMem()))
		{
#ifdef _WIN32 
			myMemory = MapViewOfFile(myMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
#else
			if (!myAmOwner)
			{
				struct stat stats;
				if (fstat(myMapping, &stats) == 0)
				{
					mySize = static_cast<unsigned int>(stats.st_size);
				}
			}
			myMemory = mmap(nullptr, mySize, PROT_READ | PROT_WRITE, MAP_SHARED, myMapping, 0);
			if (myMemory == MAP_FAILED)
			{
				myMemory = nullptr;
			}
#endif 
			if (!myMemory)
				myErrorState = UT_SHM_ERR_UNABLE_TO_MAP;
		}
	}
	if (myMemory)
	{
		myErrorState = UT_SHM_ERR_NONE;
	}
	return myMemory;
}

void
UT_SharedMem::lock()
{
	myMutex->lock();
}

bool
UT_SharedMem::tryLock(int timeout)
{
	return myMutex->tryLock(timeout);
}

bool
UT_SharedMem::unlock()
{
	return myMutex->unlock();
}

ShmString
UT_SharedMem::getErrorString(UT_SharedMemError e)
{
	switch (e)
	{
		default:
			return ShmString(STR_LIT("Unknown Error"));
		case UT_SHM_ERR_NONE:
			return ShmString();
		case UT_SHM_ERR_ALREADY_EXIST:
			return ShmString(STR_LIT("Shared Memory with given name already exists."));
		case UT_SHM_ERR_DOESNT_EXIST:
			return ShmString(STR_LIT("Shared Memory with given name does not exist."));
		case UT_SHM_ERR_INFO_ALREADY_EXIST:
			return ShmString(STR_LIT("Shared Memory Info with given name already exists."));
		case UT_SHM_ERR_INFO_DOESNT_EXIST:
			return ShmString(STR_LIT("Shared Memory Info with given name does not exist."));
		case UT_SHM_ERR_UNABLE_TO_MAP:
			return ShmString(STR_LIT("Unable to map Shared Memory."));
		case UT_SHM_ERR_UNABLE_TO_LOCK_INFO:
			return ShmString(STR_LIT("Unable to lock Shared Memory."));

	}
}

#ifdef __APPLE__

/*

 macOS limits shm_open names to 31 characters, so we maintain a directory in our own shared memory to
 look up tokens for longer names

 */
const char *UT_SharedMem::DirectoryName = "TD1zzSharedNameDirectory";
const char *UT_SharedMem::DirectoryLockName = "TD1zzSharedNameLock";

struct UT_SharedMemDirectory {
	uint64_t    allocated;      // only in directory, discounts header
	uint64_t    used;           // only in directory, discounts header
	uint64_t    version;        // 1
	char        directory[1];   // continues
};

static uint64_t UT_ShMDirectoryAvailableBytes(const struct UT_SharedMemDirectory &directory)
{
	return directory.allocated - directory.used;
}

static uint64_t UT_ShMDirectoryBytesRequired(const std::string &key, const std::string &value)
{
	return key.length() + value.length() + 2;
}

static uint64_t UT_ShMDirectoryHeaderBytes()
{
	return offsetof(struct UT_SharedMemDirectory, directory);
}

static bool UT_ShMDirectoryGet(const struct UT_SharedMemDirectory &directory, const std::string &key, std::string &value)
{
	bool isKey = true; // start on a key
	bool matched = false;
	value.clear();
	for (int i = 0; i < directory.used; i++)
	{
		char c = directory.directory[i];
		if (c)
		{
			// disregard values we're not matching
			if (isKey || matched)
			{
				value += c;
			}
		}
		else if (isKey)
		{
			if (value == key)
			{
				matched = true;
			}
			value.clear();
			isKey = false;
		}
		else
		{
			if (matched)
			{
				return true;
			}
			value.clear();
			isKey = true;
		}
	}
	value.clear();
	return false;
}

static bool UT_ShMDirectoryHasValue(const UT_SharedMemDirectory &directory, const std::string &value)
{
	std::string next;
	bool isKey = true; // start on a key
	for (int i = 0; i < directory.used; i++)
	{
		char c = directory.directory[i];
		if (c)
		{
			if (!isKey)
			{
				next += c;
			}
		}
		else
		{
			if (!isKey && next == value)
			{
				return true;
			}
			next.clear();
			isKey = !isKey;
		}
	}
	return false;
}

static void UT_ShMDirectoryAppendKnownFitString(UT_SharedMemDirectory &directory, const std::string &string)
{
	string.copy(&directory.directory[directory.used], string.length());
	directory.used += string.length();
	directory.directory[directory.used] = 0;
	directory.used += 1;
}

static bool UT_ShMDirectoryAppend(UT_SharedMemDirectory &directory, const std::string &key, const std::string &value)
{
	if (UT_ShMDirectoryBytesRequired(key, value) > UT_ShMDirectoryAvailableBytes(directory))
	{
		return false;
	}
	UT_ShMDirectoryAppendKnownFitString(directory, key);
	UT_ShMDirectoryAppendKnownFitString(directory, value);
	return true;
}

ShmString
UT_SharedMem::ShmNameForName(const ShmString &name, bool create)
{
	if (name.length() <= PSHMNAMLEN)
	{
		return name;
	}

	std::string found;
	bool done = false;
	UT_Mutex sharedNameLock(DirectoryLockName);
	if (sharedNameLock.tryLock(5000))
	{
		int fd = shm_open(DirectoryName, create ? (O_RDWR | O_CREAT) : O_RDONLY, 0666);
		if (fd != -1)
		{
			struct stat stats;
			UT_SharedMemDirectory *dir = nullptr;
			// First try lookup in any existing directory
			if (fstat(fd, &stats) == 0 && stats.st_size > 0)
			{
				dir = reinterpret_cast<UT_SharedMemDirectory *>(mmap(nullptr, stats.st_size, create ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd, 0));
				if (dir == MAP_FAILED)
				{
					dir = nullptr;
				}

				if (dir)
				{
					done = UT_ShMDirectoryGet(*dir, name, found);
				}
			}
			if (!done && create)
			{
				std::string newValue;
				{
					int suffix = 0;
					do {
						newValue = std::string("TD44e33-") + std::to_string(suffix);
						suffix++;
					} while (dir && UT_ShMDirectoryHasValue(*dir, newValue));
				}
				// Try append in existing space
				if (dir)
				{
					done = UT_ShMDirectoryAppend(*dir, name, newValue);
				}
				if (!done)
				{
					// Grow directory
					uint64_t current = UT_ShMDirectoryHeaderBytes() + (dir ? dir->used : 0);
					uint64_t extended = current + UT_ShMDirectoryBytesRequired(name, newValue);
					// Life's simpler working in multiples of page-size
					size_t pagesize = getpagesize();
					uint64_t overspill = extended % pagesize;
					if (overspill != 0)
					{
						extended += pagesize - overspill;
					}
					void *previous = dir ? malloc(current) : nullptr;
					// Check for malloc failure
					if (previous || !dir)
					{
						if (previous && dir)
						{
							memcpy(previous, dir, current);
						}
						if (dir)
						{
							// a macOS POSIX peculiarity: shm_open only allows ftruncate once, so close, unlink and then re-open
							// and re-write
							munmap(dir, stats.st_size);
							dir = nullptr;
						}
						close(fd);
						shm_unlink(DirectoryName);
						fd = shm_open(DirectoryName, create ? (O_RDWR | O_CREAT) : O_RDONLY, 0666);

						stats.st_size = extended;
						if (fd != -1 && ftruncate(fd, stats.st_size) == 0)
						{
							dir = reinterpret_cast<UT_SharedMemDirectory *>(mmap(nullptr, stats.st_size, create ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd, 0));
							if (dir == MAP_FAILED)
							{
								dir = nullptr;
							}
						}
						if (dir && previous)
						{
							memcpy(dir, previous, current);
						}
						if (dir)
						{
							dir->allocated = extended - UT_ShMDirectoryHeaderBytes();
							dir->used = current - UT_ShMDirectoryHeaderBytes();
							dir->version = 1;
						}
						done = UT_ShMDirectoryAppend(*dir, name, newValue);
					}
					// free is null-safe
					free(previous);
				}
				if (done)
				{
					found = newValue;
				}
			}
			if (dir)
			{
				munmap(dir, stats.st_size);
			}
			if (fd != -1)
			{
				close(fd);
			}
		}
		sharedNameLock.unlock();
	}
	return found;
}
#endif
