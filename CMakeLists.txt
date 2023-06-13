cmake_minimum_required(VERSION 3.18)
project(bk_transfer)

# Add path to symphony.h
include_directories("/usr/include/eigen3", "include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lz -std=c++14 -L/usr/lib/openmpi -lmpi_cxx -fopenmp -march=native -DNDEBUG -O2 -fext-numeric-literals")
set(SOURCE_FILES src/main.cpp src/utils.cpp src/Pixel.cpp src/Ray.cpp
        src/Image.cpp src/ImagePlane.cpp src/Geometry.cpp src/Cone.cpp src/Cylinder.cpp src/Parabaloid.cpp
        include/MyExceptions.h include/logspace.h include/linspace.h src/Intersection.cpp src/BField.cpp
        src/NField.cpp src/VField.cpp src/Jet.cpp src/ParticlesDistribution.cpp
        src/Observation.cpp src/System.cpp src/main.cpp)
add_executable(bk_transfer ${SOURCE_FILES})


# Require MPI for this project:
find_package(MPI REQUIRED)


#find_library(FOO_LIB symphony)
#message("${FOO_LIB}")


#set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
#set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
#include_directories(${MPI_INCLUDE_PATH})
#message("${MPI_INCLUDE_PATH}")
#target_link_libraries(bk_transfer ${MPI_LIBRARIES})


set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(${MPI_INCLUDE_PATH})
message("${MPI_INCLUDE_PATH}")
target_link_libraries(bk_transfer ${MPI_LIBRARIES})