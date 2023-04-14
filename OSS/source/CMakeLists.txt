# Source code for apps

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/utils ${OpenCV_INCLUDE_DIR})
target_link_libraries(utils INTERFACE GLM)
add_subdirectory(external)
add_subdirectory(FaceTrack)
add_subdirectory(BodyTrack)
add_subdirectory(GazeRedirect)
