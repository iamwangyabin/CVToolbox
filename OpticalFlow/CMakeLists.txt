cmake_minimum_required( VERSION 2.8 )
# Create Project
project( OpticalFlow )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

# Find Packages
# Find OpenCV
find_package( OpenCV 3.1 REQUIRED )

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        )

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        )

add_library( OpticalFlow SHARED
#        OF.cpp
        LKOF.cpp
        )

target_link_libraries( OpticalFlow
        ${THIRD_PARTY_LIBS}
        )
#add_executable( OF OF.cpp)
#target_link_libraries( OF OpticalFlow )
add_executable( LKOF LKOF.cpp)
target_link_libraries( LKOF OpticalFlow )