cmake_minimum_required(VERSION 3.5)
project(MLForcePlugin)

find_package(Torch REQUIRED)
set(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
include_directories("${OPENMM_DIR}/include")
link_directories("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

set(LIBTORCH_DIR "/usr/local" CACHE PATH "Where LIBTORCH is installed")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Set RPATH settings to avoid cycles in the constraint graph
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${OPENMM_DIR}/lib;${OPENMM_DIR}/lib/plugins")

if(APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
endif(APPLE)

if(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    else(WIN32)
        set(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    endif(WIN32)
endif(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all executables.")
endif()
if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all libraries.")
endif()
set(${PROJECT_NAME}_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
set(${PROJECT_NAME}_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})

set(NN_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)
set(NN_LIBRARY_NAME MLForce)
set(SHARED_NN_TARGET ${NN_LIBRARY_NAME})

set(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

set(API_INCLUDE_FILES)
foreach(dir ${API_INCLUDE_DIRS})
    file(GLOB fullpaths ${dir}/*.h)
    set(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
endforeach(dir)

set(SOURCE_FILES)
set(SOURCE_INCLUDE_FILES)
foreach(subdir ${NN_PLUGIN_SOURCE_SUBDIRS})
    file(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    file(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    set(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})
    set(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
endforeach(subdir)

# add the library
add_library(${SHARED_NN_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
set_target_properties(${SHARED_NN_TARGET}
    PROPERTIES
    COMPILE_FLAGS "-DNN_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_link_libraries(${SHARED_NN_TARGET} OpenMM ${TORCH_LIBRARIES})

message(STATUS "OpenMM libraries: ${OpenMM_LIBRARIES}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

install(TARGETS ${SHARED_NN_TARGET} DESTINATION lib)

file(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
install(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
file(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
install(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

enable_testing()
add_subdirectory(serialization/tests)

file(GLOB_RECURSE TEST_FILES RELATIVE "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/*.pt"
)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
set(COPIED_TEST_FILES)
foreach(TEST_FILE ${TEST_FILES})
    set(infile "${CMAKE_SOURCE_DIR}/${TEST_FILE}")
    set(outfile "${CMAKE_BINARY_DIR}/${TEST_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(COPIED_TEST_FILES ${COPIED_TEST_FILES} "${outfile}")
endforeach()
add_custom_target(CopyTestFiles ALL DEPENDS ${COPIED_TEST_FILES})

add_subdirectory(platforms/reference)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
find_package(OpenCL QUIET)
if(OPENCL_FOUND)
    set(NN_BUILD_OPENCL_LIB ON CACHE BOOL "Build implementation for OpenCL")
else(OPENCL_FOUND)
    set(NN_BUILD_OPENCL_LIB OFF CACHE BOOL "Build implementation for OpenCL")
endif(OPENCL_FOUND)
if(NN_BUILD_OPENCL_LIB)
    add_subdirectory(platforms/opencl)
endif(NN_BUILD_OPENCL_LIB)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(NN_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
else(CUDA_FOUND)
    set(NN_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
endif(CUDA_FOUND)
if(NN_BUILD_CUDA_LIB)
    add_subdirectory(platforms/cuda)
endif(NN_BUILD_CUDA_LIB)

find_program(PYTHON_EXECUTABLE python)
find_program(SWIG_EXECUTABLE swig)
if(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    set(NN_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
else(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    set(NN_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
endif(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
if(NN_BUILD_PYTHON_WRAPPERS)
    add_subdirectory(python)
endif(NN_BUILD_PYTHON_WRAPPERS)
