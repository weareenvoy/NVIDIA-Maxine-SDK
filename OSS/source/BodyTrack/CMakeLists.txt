set(SOURCE_FILES 
	BodyEngine.cpp 
	BodyTrack.cpp 
	Timer.cpp
	../utils/RenderingUtils.cpp 
	../utils/SharedMem/UT_Mutex.cpp
	../utils/SharedMem/UT_SharedMem.cpp
	../utils/osc/OscOutboundPacketStream.cpp
	../utils/osc/OscPrintReceivedElements.cpp
	../utils/osc/OscReceivedElements.cpp
	../utils/osc/OscTypes.cpp 
	../utils/ip/IpEndpointName.cpp
	../utils/ip/NetworkingUtils.cpp
	../utils/ip/UdpSocket.cpp
)
set(HEADER_FILES 
	BodyEngine.h
	BodyTrack.h
	Timer.h
	../utils/SharedMem/UT_Mutex.h
	../utils/SharedMem/UT_SharedMem.h
	../utils/SharedMem/TOP_SharedMemHeader.h
	../utils/SharedMem/CHOP_SharedMemHeader.h
	../utils/cuda/cuda.h
	../utils/cuda/cuda_device_runtime_api.h
	../utils/cuda/cuda_runtime_api.h
	../utils/cuda/cudart_platform.h
	../utils/osc/MessageMappingOscPacketListener.h
	../utils/osc/OscException.h
	../utils/osc/OscHostEndianness.h
	../utils/osc/OscOutboundPacketStream.h
	../utils/osc/OscPacketListener.h
	../utils/osc/OscPrintReceivedElements.h
	../utils/osc/OscReceivedElements.h
	../utils/osc/OscTypes.h
	../utils/ip/UdpSocket.h
	../utils/ip/PacketListener.h
	../utils/ip/IpEndpointName.h
	../utils/ip/NetworkingUtils.h
	../utils/ip/PacketListener.h
	../utils/ip/TimerListener.h
	../utils/ip/UdpSocket.h
)

if(MSVC)
	set(SOURCE_FILES ${SOURCE_FILES} 
		../../nvar/src/nvARProxy.cpp
		../../nvar/src/nvCVImageProxy.cpp
	)
	set(HEADER_FILES ${HEADER_FILES}
		../utils/RenderingUtils.h
	)
endif(MSVC)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

add_executable(BodyTrack ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(BodyTrack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(BodyTrack PUBLIC ${SDK_INCLUDES_PATH})

if(MSVC)
	target_link_libraries(
		BodyTrack PUBLIC
		opencv346
		opencv346d
		utils
	)
	set(ARSDK_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)
	set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
	set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
	set_target_properties(
		BodyTrack PROPERTIES 
		FOLDER SampleApps 
		VS_DEBUGGER_ENVIRONMENT "${PATH_STR}" 
		VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
	)

elseif(UNIX)
	find_package(PNG  REQUIRED)
	find_package(JPEG REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	target_link_libraries(
		BodyTrack PUBLIC
		nvARPose
		NVCVImage
		OpenCV
		utils
	)
endif(MSVC)