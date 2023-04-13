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
 * Filename: TOP_SharedMemHeader.h
 */

#ifndef TOP_SHARED_MEM_HEADER_H
#define TOP_SHARED_MEM_HEADER_H

#define TOP_SHM_MAGIC_NUMBER 	0xd95ef835

#define TOP_SHM_VERSION_NUMBER 	1

enum TOP_PixelFormat : uint32_t
{
	// 8-bit fixed
	TOP_FORMAT_R8_UNORM = 9,
	TOP_FORMAT_R8G8_UNORM = 16,
	TOP_FORMAT_R8G8B8A8_UNORM = 37,
    TOP_FORMAT_R8G8B8A8_SRGB = 43,
	TOP_FORMAT_B8G8R8A8_UNORM = 44,
	TOP_FORMAT_B8G8R8A8_SRGB = 50,

	// 10-bit fixed
	TOP_FORMAT_A2B10G10R10_UNORM_PACK32 = 64,

	// 11-bit float
	TOP_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,

	// 16-bit fixed
	TOP_FORMAT_R16_UNORM = 70,
	TOP_FORMAT_R16G16_UNORM = 77,
	TOP_FORMAT_R16G16B16A16_UNORM = 91,

	// 16-bit float
	TOP_FORMAT_R16_SFLOAT = 76,
	TOP_FORMAT_R16G16_SFLOAT = 83,
	TOP_FORMAT_R16G16B16A16_SFLOAT = 97,

	// 32-bit float
	TOP_FORMAT_R32_SFLOAT = 100,
	TOP_FORMAT_R32G32_SFLOAT = 103,
	TOP_FORMAT_R32G32B32A32_SFLOAT = 109,

	// Swizzled to 000A
	TOP_FORMAT_A8_UNORM = 0xF0001,
	TOP_FORMAT_A16_UNORM,
	TOP_FORMAT_A16_SFLOAT,
	TOP_FORMAT_A32_SFLOAT,

	// Swizzled to RRRA. Called Mono-Alpha to the users
	TOP_FORMAT_R8A8_UNORM,
	TOP_FORMAT_R16A16_UNORM,
	TOP_FORMAT_R16A16_SFLOAT,
	TOP_FORMAT_R32A32_SFLOAT,
};

// If you add new members to this after it's released, add them after dataOffset
class TOP_SharedMemHeader
{
public:
	// Magic number to make sure we are looking at the correct memory
	// must be set to TOP_SHM_MAGIC_NUMBER
	int							magicNumber = TOP_SHM_MAGIC_NUMBER;
	// version number of this header, must be set to TOP_SHM_VERSION_NUMBER
	int							version = TOP_SHM_VERSION_NUMBER;

	// image width
	int							width; 
	// image height
	int							height;

	// X aspect of the image
	float						aspectx;
	// Y aspect of the image
	float						aspecty;

	// The desired pixel format of the image data, tightly packed
	TOP_PixelFormat				pixelFormat; 

	// The size in bytes of the image data, not including any header memory
	int64_t						dataSize;

	// This offset (in bytes) is the diffrence between the start of this header,
	// and the start of the image data
	// The SENDER is required to set this. Unless you are doing something custom
	// you should set this to calcDataOffset();
	// If you are the RECEIVER, don't change this value.
	int							dataOffset; 

	// Both the sender and the reciever can use this to get the pointer to the actual
	// image data (as long as dataOffset is set beforehand).
	void*
	getImage()
	{
		char *c = (char*)this;
		c += dataOffset;
		return (void*)c;
	}
	
	static int
	calcDataOffset()
	{
		return sizeof(TOP_SharedMemHeader);
	}
};

#endif
