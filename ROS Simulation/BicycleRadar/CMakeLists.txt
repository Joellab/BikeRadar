cmake_minimum_required(VERSION 2.8.3)
project(BicycleRadar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        sensor_msgs
        tf
        nav_msgs
        visualization_msgs
        gazebo_ros
        dynamic_reconfigure
        )

generate_dynamic_reconfigure_options(cfg/bicycle_interaction.cfg)
catkin_package(DEPENDS CATKIN DEPENDS dynamic_reconfigure)

# Depend on system install of Gazebo
find_package(gazebo REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES autonomous_bicycle
        CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs
        #  DEPENDS system_lib
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})


install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        )

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
