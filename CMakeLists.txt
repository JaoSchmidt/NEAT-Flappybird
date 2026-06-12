# Root CMakeLists.txt for Pain Engine Game Projects This file allows developers
# to create their game project independently, then import the Pain Engine as a
# subdirectory.
#
# Project Structure:
#
# ./CMakeLists.txt (this file) #                                              #
# ./src/                (your game source files) #                            #
# ./include/            (your game headers) #                                 #
# ./resources/ (your game assets and scritps #                                #
# ./Engine/             (Pain Engine subdirectory #                           #

cmake_minimum_required(VERSION 3.25)

set(PROJECT_NAME "Painful Sprite Game Example")
project(${PROJECT_NAME})
set(TARGET_NAME "flappybird")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========== Game Executable Configuration ==================================
file(GLOB_RECURSE GAME_SOURCES CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(${TARGET_NAME} ${GAME_SOURCES})

# Include directories for your game
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# C++ Standard Configuration
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20
                                                CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${TARGET_NAME} PRIVATE /W4 /std:c++20 /Zi /bigobj)
  set_target_properties(${TARGET_NAME} PROPERTIES WIN32_EXECUTABLE TRUE)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -g -std=c++20 -Wall -Wextra)
endif()

# =========== Pain Engine Integration ==================================
# Add the Pain Engine as a subdirectory
include(FetchContent)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Engine/CMakeLists.txt")
  add_subdirectory(Engine)
else()
  message(
    STATUS
      "Pain Engine not found at ${CMAKE_CURRENT_SOURCE_DIR}/Engine/CMakeLists.txt"
  )
  message(STATUS "Adding Pain Engine from: ${CMAKE_CURRENT_SOURCE_DIR}/Engine")
  FetchContent_Declare(
    PainEngine
    GIT_REPOSITORY https://github.com/JaoSchmidt/Pain-Engine.git
    GIT_TAG master
    GIT_SUBMODULES_RECURSE TRUE)
  FetchContent_MakeAvailable(PainEngine)
endif()

# Link the engine library to your game executable
target_link_libraries(
  ${TARGET_NAME} PUBLIC#
  Pain #
  Painless #
)

# =========== Copying Resources ==================================
# Copy all files from resources/ to build directory file(COPY
# ${PROJECT_SOURCE_DIR}/resources/ DESTINATION
# ${CMAKE_CURRENT_BINARY_DIR}/resources/)

set(RESOURCES_SOURCE_DIR "${PROJECT_SOURCE_DIR}/resources")
set(RESOURCES_DEST_DIR "${CMAKE_BINARY_DIR}/resources")

# Copy resources to build directory (cross-platform)
if(UNIX OR APPLE)
  add_custom_target(
    copy_game_resources ALL
    COMMAND mkdir -p "${RESOURCES_DEST_DIR}"
    COMMAND rsync -av --chmod=F444,D775 "${RESOURCES_SOURCE_DIR}/"
            "${RESOURCES_DEST_DIR}/"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Syncing game resources to build directory")
else()
  add_custom_target(
    copy_game_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RESOURCES_SOURCE_DIR}"
            "${RESOURCES_DEST_DIR}"
    COMMENT "Copying game resources to build directory")
endif()
