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

#include "UT_Mutex.h"
#include "UT_SharedMem.h"
#include "TOP_SharedMemHeader.h"

int 
wmain(int argc, wchar_t* argv[])
{
	UT_SharedMem *shm = new UT_SharedMem(ShmString(L"TOPShm"));

	UT_SharedMemError err = shm->getErrorState();
	if (err != UT_SHM_ERR_NONE)
	{
		// an error occured
	}

	while (true)
	{
		if (!shm->tryLock(5000))
		{
			// error
		}
		else
		{
			void *data = shm->getMemory();

			if (data)
			{
				TOP_SharedMemHeader *topHeader = (TOP_SharedMemHeader*)data;

				//  Check to see if it's a TOP shared memory segment
				if (topHeader->magicNumber == TOP_SHM_MAGIC_NUMBER)
				{
					// Do something with the TOP data
				}
				else
				{
					// Invalid shared memory segment
				}
			}

			// Unlock it when done
			shm->unlock();
		}
	}

	return 0;
}
