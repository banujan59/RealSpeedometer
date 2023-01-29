cmake_minimum_required(VERSION 3.3)

# MUST be delcared first to allow compiler detection
project ( RealSpeedometerEmbedded )

# cpp build information
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

file(GLOB_RECURSE source_file_list *.cpp *h) 
list(FILTER source_file_list EXCLUDE REGEX "build/.*\\.cpp$")

file(GLOB_RECURSE CLUSTER_CONTROLLER
    "ClusterController/*.cpp"
    "ClusterController/*.h"
)

file(GLOB_RECURSE CLUSTER_SERVER
    "ClusterServer/*.cpp"
    "ClusterServer/*.h"
)

source_group("ClusterController"        FILES ${CLUSTER_CONTROLLER})
source_group("ClusterServer"        FILES ${CLUSTER_SERVER})

add_executable( RealSpeedometerEmbedded ${source_file_list} )
target_link_libraries (RealSpeedometerEmbedded pthread wiringPi)